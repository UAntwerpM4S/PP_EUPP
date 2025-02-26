import glob 
import pickle 

eupp_files = glob.glob("/home/jupyter-aaron/Postprocessing/PP_EUPP/data/EUPP/output.sfc.*.nc")
era5_files = glob.glob("/home/jupyter-aaron/Postprocessing/PP_EUPP/data/ERA5/era.sfc.*.nc")

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


file_keys_sorted = sorted(file_pairs.keys(), key=lambda x: (x[0], int(x[1])))

train_file_keys = [key for key in file_keys_sorted if key[0] <= 17]
val_file_keys = [key for key in file_keys_sorted if key[0] == 18]
test_file_keys = [key for key in file_keys_sorted if key[0] == 19]



# Initialize file lists
train_eupp_files, train_era5_files = [], []
val_eupp_files, val_era5_files = [], []
test_eupp_files, test_era5_files = [], []

# Assign files to the respective lists
def assign_files(keys, eupp_list, era5_list):
    for key in keys:
        for file_path in file_pairs[key]:
            if 'output' in file_path:
                eupp_list.append(file_path)
            else:
                era5_list.append(file_path)

assign_files(train_file_keys, train_eupp_files, train_era5_files)
assign_files(val_file_keys, val_eupp_files, val_era5_files)
assign_files(test_file_keys, test_eupp_files, test_era5_files)


def save_pickle(file_list, filename):
    with open(filename, 'wb') as f:
        pickle.dump(file_list, f)

save_pickle(train_eupp_files, "train_eupp_files.pkl")
save_pickle(train_era5_files, "train_era5_files.pkl")
save_pickle(val_eupp_files, "val_eupp_files.pkl")
save_pickle(val_era5_files, "val_era5_files.pkl")
save_pickle(test_eupp_files, "test_eupp_files.pkl")
save_pickle(test_era5_files, "test_era5_files.pkl")

