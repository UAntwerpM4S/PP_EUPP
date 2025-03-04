import sys, os
import numpy as np
sys.path.append('/home/jupyter-aaron/Postprocessing/pythie')
import xarray as xr
import pickle
import gc
import time 
from core.data import Data
import postprocessors.MBM as MBM

os.environ["OMP_NUM_THREADS"] = "16"  # Reduce OpenMP threads (was 32)
os.environ["MKL_NUM_THREADS"] = "16"  # Reduce MKL threads
os.environ["OPENBLAS_NUM_THREADS"] = "16"  # Reduce OpenBLAS threads
os.environ["NUMEXPR_NUM_THREADS"] = "16"  # Reduce NumExpr threads


target = 't2m'
params = ['t2m', 'z', 't', 'u10', 'v10', 'tcc', 'sd', 'mx2t6', 'mn2t6', 'w10', 'p10fg6']

directory = "../TrainValTestSplit"
base_dir = "../data"
output_dir = './resultsClassicalMBM/t2m'
time_log_path = os.path.join(output_dir, 'training_times_per_lead_time.txt')


def load_forecast_data(file_list, base_dir, params, lead, chunk_size=5):
    """Loads forecast data in smaller chunks to reduce memory usage."""
    print(f"Loading forecast data for lead time {lead}...")

    num_files = len(file_list)
    chunked_arrays = []

    for start in range(0, num_files, chunk_size):
        end = min(start + chunk_size, num_files)
        temp_array = np.empty((len(params), end - start, 11, 1, 1, 32, 33), dtype=np.float32)

        for i, file in enumerate(file_list[start:end]):
            file_path = os.path.join(base_dir, file)
            with xr.open_dataset(file_path) as dataset:
                dataset = dataset.isel(step=lead)
                for j, var in enumerate(params):
                    if var in dataset:
                        temp_array[j, i, :, 0, 0, :, :] = dataset[var].values 
                    else:
                        print(f"Warning: {var} not found in {file}")

        chunked_arrays.append(temp_array)
        gc.collect()  # Free memory after each chunk

    final_array = np.concatenate(chunked_arrays, axis=1)
    print("Forecast data loaded.")
    return final_array

def load_observation_data(file_list, base_dir, lead, chunk_size=5):
    """Loads observation data in smaller chunks to reduce memory usage."""
    print(f"Loading observation data for lead time {lead}...")

    num_files = len(file_list)
    chunked_arrays = []

    for start in range(0, num_files, chunk_size):
        end = min(start + chunk_size, num_files)
        temp_array = np.empty((1, end - start, 1, 1, 1, 32, 33), dtype=np.float32)

        for i, file in enumerate(file_list[start:end]):
            file_path = os.path.join(base_dir, file)
            with xr.open_dataset(file_path) as dataset:
                dataset = dataset.isel(step=lead)
                temp_array[0, i, :, 0, 0, :, :] = dataset[target].values 

        chunked_arrays.append(temp_array)
        gc.collect()

    final_array = np.concatenate(chunked_arrays, axis=1)
    print("Observation data loaded.")
    return final_array



print("Loading testing forecast and observation data files...")
with open(os.path.join(directory, "test_eupp_files.pkl"), 'rb') as f:
    rfcs_test_file = pickle.load(f)
with open(os.path.join(directory, "test_era5_files.pkl"), 'rb') as f:
    obs_test_file = pickle.load(f)


with open(time_log_path, 'w') as time_log_file:
    for lead_time in range(0, 20):
        try:
            print(f"\nProcessing lead time {lead_time}...")

            start_time = time.time()

            # Load training data in chunks to avoid memory spikes
            print("Loading training forecast data files...")
            with open(os.path.join(directory, "train_eupp_files.pkl"), 'rb') as f:
                rfcs_train_file = pickle.load(f)
            rfcs_array_train = load_forecast_data(rfcs_train_file, base_dir, params, lead_time)
            del rfcs_train_file  # Free memory
            gc.collect()

            print("Loading training observation data files...")
            with open(os.path.join(directory, "train_era5_files.pkl"), 'rb') as f:
                obs_train_file = pickle.load(f)
            obs_array_train = load_observation_data(obs_train_file, base_dir, lead_time)
            del obs_train_file  # Free memory
            gc.collect()

            # Train MBM
            print("Initializing training data structures for MBM...")
            rfcs_whole_Data = Data(rfcs_array_train)
            obs_whole_Data = Data(obs_array_train)
            print("Training MBM postprocessor...")

            if target == 't2m':
                essacc = MBM.EnsembleSpreadScalingNgrCRPSCorrection()
            else:
                essacc = MBM.EnsembleAbsCRPSTruncCorrection()
            essacc.train(obs_whole_Data, rfcs_whole_Data, ntrial=1)
            print("MBM training complete.")

            # Free memory
            del rfcs_array_train, obs_array_train, rfcs_whole_Data, obs_whole_Data
            gc.collect()

            # Load test data in chunks
            rfcs_array_test = load_forecast_data(rfcs_test_file, base_dir, params, lead_time)
            obs_array_test = load_observation_data(obs_test_file, base_dir, lead_time)
            gc.collect()

            # Apply MBM postprocessor to the test data
            print("Applying MBM postprocessor to testing data...")
            test_whole = Data(rfcs_array_test)
            testMBM_whole = essacc(test_whole)[:, :, :, :, :, :]
            print(f"MBM testing complete for lead time {lead_time}.")

            # Save results
            output_path = os.path.join(output_dir, f'MBM_{target}_{lead_time}.npy')
            np.save(output_path, testMBM_whole)
            print(f"Results saved to {output_path}")

            # Log execution time
            elapsed_time = time.time() - start_time
            time_log_file.write(f"Lead time {lead_time}: {elapsed_time:.2f} seconds\n")
            time_log_file.flush()
            print(f"Lead time {lead_time} took {elapsed_time:.2f} seconds")

        except Exception as e:
            print(f"Error at lead time {lead_time}: {e}")
            time_log_file.write(f"Lead time {lead_time}: ERROR - {str(e)}\n")
            time_log_file.flush()

print("All lead times processed.")

