import os 
import torch
from torch.utils.data import Dataset, DataLoader
import numpy as np
import xarray as xr
import cf2cdm
import glob
from datetime import datetime
import random
import pickle

class EUPPFullEnsembleDataset(Dataset):
    def __init__(self, nsample, target_var,data_path="data",num_ensemble=11, dataset_type="train", normalized=True, return_time=False):
        self.num_ensemble = num_ensemble
        self.normalized = normalized
        self.return_time = return_time
        self.data_path = data_path
        if target_var in ["t2m","w10","w100"]:
            self.variables=['w100','t2m','w10','u10','tcc','u100','z','u','w700','t','p10fg6']#,'oro']
            #tcc','mx2t6','mn2t6', 'sd','slhf6']
            self.target_var = target_var
            self.value_range = {"t2m":(235, 304), "z": (48200, 58000), "t":(240, 299), "u10": (-13., 11.), "tcc": (0., 1.0), "w100":(0,50),"w10":(0,30), "u100": (-35,45), "u": (-45,60), "w700": (0,60), "p10fg6": (0,60), "oro":(-400,2800)}

        
        
        self.train_eupp_files = []
        self.train_era5_files = []
        self.val_eupp_files = []
        self.val_era5_files = []
        self.test_eupp_files = []
        self.test_era5_files = []
    
            
        eupp_files = glob.glob("./data/EUPP/output.sfc.*.nc")
        era5_files = glob.glob("/data/ERA5/era.sfc.*.nc")
        
        eupp_file_train_path = "./TrainValTestSplit/train_eupp_files.pkl"
        eupp_file_val_path = "./TrainValTestSplit/val_eupp_files.pkl"
        era5_file_train_path = "./TrainValTestSplit/train_era5_files.pkl"
        era5_file_val_path = "./TrainValTestSplit/val_era5_files.pkl"
  
        # Load the split files based on dataset type
        if dataset_type == "train":
            with open(eupp_file_train_path, 'rb') as f:
                self.eupp_files = pickle.load(f)
            with open(era5_file_train_path, 'rb') as f:
                self.era5_files = pickle.load(f)
        elif dataset_type == "test":
            with open(eupp_file_val_path, 'rb') as f:
                self.eupp_files = pickle.load(f)
            with open(era5_file_val_path, 'rb') as f:
                self.era5_files = pickle.load(f)
                
    def __len__(self):
        return len(self.eupp_files)

    def __getitem__(self, idx):
        if isinstance(idx, int):
            ds_eupp = xr.open_dataset(self.eupp_files[idx]).drop_vars("time", errors="ignore")
            ds_eupp = ds_eupp.fillna(9999.0)
            ds_era5 = xr.open_dataset(self.era5_files[idx]).fillna(9999.0).isel(step=slice(1, None))  # Exclude the first step, its absent in the reforecast files 
            ds_era5 = ds_era5.rename({'w100_obs': 'w100'})
            orography_data = xr.open_dataset("/home/jupyter-aaron/Postprocessing/TfMBM_wind/baselines/data/oro.nc") 
            #orography_data=orography_data.sel(latitude=slice(min_lat, max_lat), longitude=slice(min_lon, max_lon))
            ensemble=ds_eupp["number"].values
            step=ds_eupp["step"].values
            oro_expand=orography_data["oro"].expand_dims(number=ensemble,step=step)
            orography_expanded=oro_expand.reindex(number=ensemble, step=step, latitude=orography_data.latitude, longitude=orography_data.longitude)
            min_lat, max_lat = 53.5, 45.75
            min_lon, max_lon = 2.4, 10.6
            ds_eupp = ds_eupp.sel(latitude=slice(min_lat, max_lat), longitude=slice(min_lon, max_lon))
            ds_era5 = ds_era5.sel(latitude=slice(min_lat, max_lat), longitude=slice(min_lon, max_lon))
            orography_data = orography_data.sel(latitude=slice(min_lat, max_lat), longitude=slice(min_lon, max_lon))
             # Load orography data
            # Expand orography data to match the dimensions of other variables
            # orography_expanded = orography_data["oro"].expand_dims(dim={"number": np.arange(self.num_ensemble)})
            self.orography_values = torch.as_tensor(np.copy(orography_expanded.to_numpy()))
            len_lat=len(ds_eupp['latitude'].values)
            len_lon=len(ds_eupp['longitude'].values)
            len_TD=len(ds_eupp['step'].values)
            ds_eupp = ds_eupp.stack(space=["latitude","longitude"])
            ds_era5 = ds_era5.stack(space=["latitude","longitude"])
            inputs = torch.zeros((len(self.variables), self.num_ensemble, len_TD,len_lat, len_lon)) #(3,51,32,33) 
            for k in range(len(self.variables)): #len is 1
                variable = self.variables[k] 
                if variable == 'oro': # Check if variable is orography
                    values = self.orography_values # Use pre-loaded orography values
                else:
                    values = torch.reshape(torch.as_tensor(ds_eupp[variable].to_numpy()[:self.num_ensemble,:]), (self.num_ensemble,len_TD,len_lat,len_lon)) #(11,32,33)
                if self.normalized:
                    minval, maxval = self.value_range[variable]
                    values = (values - minval) / (maxval - minval)
                inputs[k, :] = values
                if (variable == self.target_var):
                    values_tar = torch.reshape(torch.as_tensor(ds_eupp[variable].to_numpy()[:self.num_ensemble,:]), (1,self.num_ensemble,len_TD,len_lat*len_lon))
                    targets = torch.reshape(torch.as_tensor(ds_era5[variable].to_numpy()),(1,len_TD,len_lat*len_lon))
                    scale_std, scale_mean = torch.std_mean(values_tar, dim=1, unbiased=False)
                    
            inputs = inputs.movedim(0,1) #(11,1,32,33) 
            inputs = inputs.movedim(1,-1)


            if self.return_time:
                return  inputs, targets , scale_mean, scale_std
            else:
                return inputs, targets , scale_mean, scale_std


def loader_prepare(args):
    trainloader = DataLoader(EUPPFullEnsembleDataset(data_path=args.data_path,
                                                      nsample=32*33,
                                                      target_var=args.target_var,
                                                      dataset_type='train', num_ensemble=args.ens_num),
                                 args.batch_size, shuffle=True, num_workers=8, pin_memory=True, prefetch_factor=6, persistent_workers=True)

    testloader = DataLoader(EUPPFullEnsembleDataset(data_path=args.data_path,
                                                     nsample=32*33,
                                                     target_var=args.target_var,
                                                     dataset_type='test', num_ensemble=args.ens_num, return_time=True),
                                args.batch_size, shuffle=False, num_workers=4, pin_memory=True, prefetch_factor=4, persistent_workers=False)
    return trainloader, testloader




