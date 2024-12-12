#This code is needed to split randomly in training and validation files
        
import glob 
import random 
import pickle 
import os 
import torch

eupp_files = glob.glob("/home/jupyter-aaron/Postprocessing/TfMBM_100ws/baselines/data/reforecasts_w100/output.sfc.*.nc")
era5_files = glob.glob("/home/jupyter-aaron/Postprocessing/TfMBM_100ws/baselines/data/era5_w100/era.sfc.*.nc")
print(len(eupp_files))

file_pairs = {}
     
for eupp_file in eupp_files:
    parts = eupp_file.split('.')  
    i_number = int(parts[-3]) 
    date_part = parts[-2] 
    file_pairs.setdefault((i_number, date_part), []).append(eupp_file)

for era5_file in era5_files:
    parts = era5_file.split('.')  
    i_number = int(parts[-3])  
    date_part = parts[-2]  
    file_pairs.setdefault((i_number, date_part), []).append(era5_file)

file_keys = list(file_pairs.keys())
random.shuffle(file_keys)


total_files = len(file_keys)
train_size = int(0.85 * total_files)
val_size = int(0.1 * total_files)


train_file_keys = file_keys[:train_size]
val_file_keys = file_keys[train_size:train_size + val_size]
test_file_keys = file_keys[train_size + val_size:]


train_eupp_files = []
train_era5_files = []
val_eupp_files = []
val_era5_files = []
test_eupp_files = []
test_era5_files = []

for key in train_file_keys:
    for file_path in file_pairs[key]:
        if 'output' in file_path:  # Check if eupp file
            train_eupp_files.append(file_path)
        else:  # Otherwise, it's an era5 file
            train_era5_files.append(file_path)

for key in val_file_keys:
    for file_path in file_pairs[key]:
        if 'output' in file_path:  # Check if eupp file
            val_eupp_files.append(file_path)
        else:  # Otherwise, it's an era5 file
            val_era5_files.append(file_path)
            
for key in test_file_keys:
    for file_path in file_pairs[key]:
        if 'output' in file_path:  # Check if eupp file
            test_eupp_files.append(file_path)
        else:  # Otherwise, it's an era5 file
            test_era5_files.append(file_path)

eupp_file_train_path = "train_eupp_files.pkl"
era5_file_train_path = "train_era5_files.pkl"
eupp_file_val_path = "val_eupp_files.pkl"
era5_file_val_path = "val_era5_files.pkl"
eupp_file_test_path = "test_eupp_files.pkl"
era5_file_test_path = "test_era5_files.pkl"

with open(eupp_file_train_path, 'wb') as f:
    pickle.dump(train_eupp_files, f)
with open(era5_file_train_path, 'wb') as f:
    pickle.dump(train_era5_files, f)
with open(eupp_file_val_path, 'wb') as f:
    pickle.dump(val_eupp_files, f)
with open(era5_file_val_path, 'wb') as f:
    pickle.dump(val_era5_files, f)
with open(eupp_file_test_path, 'wb') as f:
    pickle.dump(test_eupp_files, f)
with open(era5_file_test_path, 'wb') as f:
    pickle.dump(test_era5_files, f)
            
