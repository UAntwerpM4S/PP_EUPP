import numpy as np
from utils import args_parser, CrpsGaussianLoss, CrpsGaussianTruncatedLoss,  AdaptiveCrpsKernelLoss
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
torch.cuda.init()  
scale_dict = {"t2m":(235, 304), "z": (48200, 58000), "t":(240, 299), "u10": (-13., 11.),"v10": (-30,35), "tcc": (0., 1.0),"sd":(0,8),"mx2t6":(230,320),"mn2t6":(225,315),"v":(-50,55), "w100":(0,50),"w10":(0,30), "u100": (-35,45), "u": (-45,60),"v100":(-40,45), "w700": (0,60), "p10fg6": (0,60), "oro":(-400,2800)}

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

        output = model(inputs) #torch.Size([2, 11, 20, 32, 33, 1])
        batch_size = output.shape[0]
     
        
        #the distributional parameters
        mu = output[...,0].mean(dim=1).view(batch_size, 20, 1056) #20
        mu = mu* scale_std + scale_mean 
        sigma=output[...,0].std(dim=1).view(batch_size, 20, 1056) #20
        #sigma = sigma*scale_std
        sigma =torch.exp(sigma) * scale_std

        if args.loss == 'CRPS':
            loss=criterion(mu, sigma, targets)
            
        elif args.loss == 'CRPSTRUNC':
            loss=criterion(mu, sigma, targets)
        else:
            ensemble = output.squeeze(-1).view(batch_size,11,20, 1056) 
            ensemble = ensemble* scale_std + scale_mean 
           # ensemble = torch.clamp(ensemble, min=0.0) #no negative values for windspeed
            ensemble= ensemble.view(batch_size, 11,20, 32,33)
            
            kernel_target=targets.view(batch_size, 20, 32,33)
            loss=criterion(ensemble, kernel_target)


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
            scale_std = torch.where(scale_std == 0, torch.tensor(1e-6, device=device), scale_std)
            output = model(inputs)
            batch_size = output.shape[0]
            
            mu = output[...,0].mean(dim=1).view(batch_size, 20, 1056) #20
            mu = mu* scale_std + scale_mean 
            sigma=output[...,0].std(dim=1).view(batch_size, 20, 1056) #20
            #sigma = sigma*scale_std
            sigma =torch.exp(sigma)* scale_std    # Prevent zero or near-zero values

            if torch.isnan(mu).any() or torch.isnan(sigma).any():
                print(f"Warning: NaN detected in mu or sigma at epoch {epoch}")
                
            if args.loss == 'CRPS':
                loss=criterion(mu, sigma, targets)
            elif args.loss == 'CRPSTRUNC':
                loss=criterion(mu, sigma, targets)
            else:
                ensemble = output.squeeze(-1).view(batch_size,11,20, 1056) 
                ensemble = ensemble* scale_std + scale_mean 
                ensemble= ensemble.view(batch_size, 11,20, 32,33)
                kernel_target=targets.view(batch_size, 20, 32,33)
                loss=criterion(ensemble, kernel_target)
                
            if not torch.isnan(loss):
                test_loss.append(loss.item())

            # Compute CRPS safely
            if len(test_loss) == 0:
                print(f"Warning: No valid test loss at epoch {epoch}")
                crps_loss = np.nan  # Avoid using an empty list in np.average()
            else:
                crps_loss = np.average(test_loss)

             

    
    if crps_loss < best_crps:
        state = {
            'model': model.state_dict(),
            'crps': crps_loss,
            'epoch': epoch,
        }
        if not os.path.isdir('checkpoint'):
            os.mkdir('checkpoint')
        checkpoint_path = f'results/Transformerweights/{args.target_var}/epochs{args.epochs}predictors{args.num_predictors}{args.loss}lambda{args.lambda_reg}k{args.k_reg}.pth'
        #torch.save(state,checkpoint_path)
        #_lambda{args.lambda_reg}k{args.k_reg}_{rgs.decay_type}
        #lr{args.lr}epochs{args.epochs}b{args.batch_size}heads{args.nheads}mlt{args.mlp_mult}Stack{args.num_blocks}{args.projection_channels}
        # save_dir = f'results/Transformerweights/{args.target_var}/'
        # checkpoint_path = os.path.join(
        #     save_dir,
        #     f'predictors{args.num_predictors}_{args.loss}_lambda{args.lambda_reg}_k{args.k_reg}_epoch{epoch}.pth')
        # torch.save(state, checkpoint_path)
        # print(f"Saving new best model weights at epoch {epoch}: CRPS = {crps_loss:.4f}")
        #torch.save(state, checkpoint_path)
        print("saving, best model weights atm")
        best_crps = crps_loss
    print(
        'test/Epoch_crps: ', crps_loss,
        '\ntest/Epoch: ', epoch,
        '\ntest/Best_crps: ', best_crps,
        '\n',
        '\n'
    )

def train_model(args, device):
    global best_crps
    best_crps = 1e10  
    start_epoch = 0
    patience = 5  # Number of epochs to wait
    no_improve_epochs = 0  # Counter for no improvement

    trainloader, testloader = loader_prepare(args)

    model = eval('Tformer_prepare')(args)
    model = model.to(device)

    if torch.cuda.device_count() > 1:
        print(f"Using {torch.cuda.device_count()} GPUs!")
        model = nn.DataParallel(model)

    if args.resume:
        print('==> Resuming from checkpoint..')
        assert os.path.isdir('checkpoint'), 'Error: no checkpoint directory found!'
        checkpoint = torch.load('./results/Transformerweights/t2m/epochs50predictors12CRPSlambda0.02k3.3_TIMETEST.pth', weights_only=False)
        model.load_state_dict(checkpoint['model'])
        best_crps = checkpoint['crps']
        start_epoch = checkpoint['epoch']

    if args.loss == 'CRPS':
        criterion = CrpsGaussianLoss()
    elif args.loss == 'CRPSTRUNC':
        criterion = CrpsGaussianTruncatedLoss()
        print("using the truncated CRPS")
    elif args.loss == 'CRPSKERNELSTEP':
        criterion = AdaptiveCrpsKernelLoss(
            mode='mean',
            initial_lambda=args.lambda_reg,
            k=args.k_reg
           # decay_epochs=args.decay_epochs,
          #  min_lambda=args.min_lambda
        )
        print("using regularization ensemble crps")

    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)

    actual_epochs_trained = 0

    for epoch in range(start_epoch, args.epochs):
        if hasattr(criterion, "step"):
            criterion.step()
        
        train(epoch, trainloader, model, optimizer, criterion, args, device)

        prev_best = best_crps
        test(epoch, testloader, model, criterion, args, device)

        actual_epochs_trained += 1

        if best_crps < prev_best:
            no_improve_epochs = 0
        else:
            no_improve_epochs += 1
            print(f"No improvement for {no_improve_epochs} epochs.")

        if no_improve_epochs >= patience:
            print(f"Early stopping triggered after {actual_epochs_trained} epochs. No improvement for {patience} consecutive epochs.")
            break

    return model, actual_epochs_trained



def main(args):
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    if device == 'cuda':
        torch.cuda.init() 
        print("CUDA Initialized")
    model, actual_epochs_trained = train_model(args, device)
    return actual_epochs_trained



if __name__ == '__main__':
    args = args_parser()
    actual_epochs_trained = main(args)


# End timing at the end of the script
end_time = time.time()
execution_time = end_time - start_time
completion_time = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
print(f"Execution Time: {execution_time} seconds")
print(f"Training Completed on: {completion_time}")


log_file_path = f'./results/Transformerweights/{args.target_var}/training_log.txt'  
os.makedirs(os.path.dirname(log_file_path), exist_ok=True)

# Information to log
batch_size = args.batch_size
mlp_mult = args.mlp_mult 
n_heads = args.nheads  
projection_channels = args.projection_channels  
num_blocks=args.num_blocks
epochs = args.epochs 
best_crps_value = best_crps  
predictors=args.num_predictors
Lossfunction=args.loss

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
    f.write(f"Actual Epochs Trained: {actual_epochs_trained}\n")
    f.write(f"Best CRPS Test Score: {best_crps_value:.4f}\n")
    f.write(f"Number of predictors: {predictors}\n")
    f.write(f"Loss function: {args.loss}\n")
    f.write(f"Lambda_reg: {args.lambda_reg}\n")
    f.write(f"k_reg: {args.k_reg}\n")
   # f.write(f"Decay epochs: {args.decay_epochs}\n")
  #  f.write(f"Min lambda: {args.min_lambda}\n")
    f.write("-" * 40 + "\n")
