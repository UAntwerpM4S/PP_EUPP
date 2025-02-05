import numpy as np
from utils import args_parser, CrpsGaussianLoss, CrpsGaussianTruncatedLoss
from Loader import *
import matplotlib.pyplot as plt
import time
import torch
from torch import nn as nn
from tqdm import tqdm
from datetime import datetime
from hanging_threads import start_monitoring
import xarray as xr
from Transformer import Tformer_prepare
import os 
from datetime import datetime

# Start timing at the beginning of the script
start_time = time.time()

scale_dict = {"t2m":(235, 304), "z": (48200, 58000), "t":(240, 299), "u10": (-13., 11.), "tcc": (0., 1.0), "w100":(0,50),"w10":(0,30), "u100": (-35,45), "u": (-45,60), "w700": (0,60), "p10fg6": (0,60), "sd": (0., 15.), "slhf6": (-10000000., +5000000.), "mx2t6":(240,310), "mn2t6":(235,310), "oro":(-400,2800)}

# Training
#this function is designed to execute one epoch of training 
torch.cuda.empty_cache()
def train(epoch, trainloader, model, optimizer, criterion, args, device):
    model.train()
    train_loss = []
    offset, scale = scale_dict[args.target_var]
    for batch_idx, (inputs, targets, scale_mean, scale_std) in tqdm(enumerate(trainloader), desc=f'Epoch {epoch}: ',
                                                                    unit="Batch", total=len(trainloader)):

        curr_iter = epoch * len(trainloader) + batch_idx
        inputs, targets = inputs.to(device), targets.to(device)
        scale_mean, scale_std = scale_mean.to(device), scale_std.to(device) 

        output = model(inputs)
        batch_size = output.shape[0]

        mu = output[...,0].mean(dim=1).view(batch_size, 20, 1056) 
        mu = mu* scale_std + scale_mean 
        sigma=output[...,0].std(dim=1).view(batch_size, 20, 1056) 
        sigma = torch.exp(sigma) * scale_std
        
        loss=criterion(mu, sigma, targets)

        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        optimizer.step()

        train_loss.append(loss.item())
    print(f'Epoch {epoch} Avg. Loss: {np.average(train_loss)}')


def test(epoch, testloader, model, criterion, args, device):
    global best_crps
    model.eval()
    test_loss = []

    with torch.no_grad():
        for batch_idx, (#dates, 
                        inputs, targets, scale_mean, scale_std) in tqdm(enumerate(testloader),
                                                                               desc=f'[Test] Epoch {epoch}: ',
                                                                               unit="Batch", total=len(testloader)):

            inputs, targets = inputs.to(device), targets.to(device)
            if torch.isnan(inputs).any() or torch.isinf(inputs).any():
                print("Input contains NaN or Inf values")
            if torch.isnan(targets).any() or torch.isinf(targets).any():
                print("Input contains NaN or Inf values")

            scale_mean, scale_std = scale_mean.to(device), scale_std.to(device)
            output = model(inputs)
            batch_size = output.shape[0]
            mu = output[...,0].mean(dim=1).view(batch_size, 20, 1056) 
            mu = mu* scale_std + scale_mean 
            sigma=output[...,0].std(dim=1).view(batch_size, 20, 1056) 
            sigma = torch.exp(sigma) * scale_std
        
            loss=criterion(mu, sigma, targets)

            test_loss.append(loss.item())


    # Save checkpoint.
    crps_loss = np.average(test_loss)
    print(crps_loss)

    
    if crps_loss < best_crps:
        print('Saving..')
        state = {
            'model': model.state_dict(),
            'crps': crps_loss,
            'epoch': epoch,
        }
        if not os.path.isdir('checkpoint'):
            os.mkdir('checkpoint')

        #comment out if you don't want to save
        # Define the checkpoint path dynamically
        checkpoint_path = f'results/Transformerweights/lr{args.lr}epochs{args.epochs}b{args.batch_size}heads{args.nheads}mlt{args.mlp_mult}Stack{args.num_blocks}{args.projection_channels}.pth'
        torch.save(state, checkpoint_path)
        print("saving, best model weights atm")
        best_crps = crps_loss

    print(
        '\ntest/Epoch_crps: ', crps_loss,
        '\ntest/Epoch: ', epoch,
        '\ntest/Best_crps: ', best_crps
    )


def train_model(args,device):
    global best_crps
    best_crps = 1e10  
    start_epoch = 0  
    
    trainloader, testloader = loader_prepare(args)

    model = eval('Tformer_prepare')(args)
    model = model.to(device)

    if torch.cuda.device_count() > 1:
        print(f"Using {torch.cuda.device_count()} GPUs!")
        model = nn.DataParallel(model)

    if args.resume:
        # Load checkpoint.
        print('==> Resuming from checkpoint..')
        assert os.path.isdir('checkpoint'), 'Error: no checkpoint directory found!'
        checkpoint = torch.load('./results/Transformerweights/lr0.0001epochs50b2heads8mlt4Stack4Projlayer64.pth')
        model.load_state_dict(checkpoint['model'])
        best_crps = checkpoint['crps']
        start_epoch = checkpoint['epoch']

    if args.loss == 'CRPS':
        criterion = CrpsGaussianLoss()
        
    else: 
        args.loss == 'CRPSTRUNC'
        criterion = CrpsGaussianTruncatedLoss()
        print("using the truncated CRPS") 


    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)


    for epoch in range(start_epoch, args.epochs):
        train(epoch, trainloader, model, optimizer, criterion, args, device)
        test(epoch, testloader, model, criterion, args, device)

    return model


def main(args):
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    model = train_model(args, device)  



if __name__ == '__main__':
    args = args_parser()
    main(args)


# End timing at the end of the script
end_time = time.time()
execution_time = end_time - start_time
completion_time = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
print(f"Execution Time: {execution_time} seconds")
print(f"Training Completed on: {completion_time}")


log_file_path = './results/Transformerweights/training_log.txt'  
os.makedirs(os.path.dirname(log_file_path), exist_ok=True)

# Information to log
batch_size = args.batch_size
mlp_mult = args.mlp_mult 
n_heads = args.nheads  
projection_channels = args.projection_channels  
num_blocks=args.num_blocks
epochs = args.epochs 
best_crps_value = best_crps  

# Log information to file
with open(log_file_path, 'a') as f:
    f.write(f"Training Summary:\n")

    f.write(f"Execution Time: {execution_time:.2f} seconds\n")
    f.write(f"Training Completion Time: {completion_time}\n")  # 
    f.write(f"Batch Size: {batch_size}\n")
    f.write(f"MLP Multiplier: {mlp_mult}\n")
    f.write(f"Number of Heads: {n_heads}\n")
    f.write(f"Number of bocks stacks: {num_blocks}\n")
    f.write(f"Projection Channels: {projection_channels}\n")
    f.write(f"Number of Epochs: {epochs}\n")
    f.write(f"Best CRPS Test Score: {best_crps_value:.4f}\n")
    f.write("-" * 40 + "\n")  
