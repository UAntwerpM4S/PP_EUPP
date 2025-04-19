import os, sys
import torch
import numpy as np
import multiprocessing
import matplotlib.pyplot as plt
import xarray as xr
import mkl
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
import cartopy.crs as ccrs
import cartopy.feature as cfeature
from xskillscore import crps_ensemble
from Transformer import StackedTransformer
from utils.metrics import CrpsGaussianLoss, CrpsGaussianTruncatedLoss
from utils.evaluation import fair_crps_ensemble, almost_fair_crps_ensemble, minmax_normalize,compute_rank_histogram

# Explicitly limit the number of cores 
multiprocessing.cpu_count = lambda: 8
os.environ["OMP_NUM_THREADS"] = "8"
os.environ["MKL_NUM_THREADS"] = "8"
os.environ["OPENBLAS_NUM_THREADS"] = "8"
os.environ["NUMEXPR_NUM_THREADS"] = "8"

# Optional: Limit mkl threads
mkl.set_num_threads(8)

# === SETTINGS ===
target = 'w100'
base_dir = '/home/jupyter-aaron/Postprocessing/PP_EUPP'
weights_dir = f"{base_dir}/results/Transformerweights/{target}"
output_root = f"{base_dir}/results/Plots_{target}"
os.makedirs(output_root, exist_ok=True)


# === LOAD DATA ONCE ===
oro= xr.open_dataset("/home/jupyter-aaron/Postprocessing/TfMBM_wind/baselines/data/oro.nc") 
oro_2D=oro['oro']
test_rfcs = xr.open_dataset(f"{base_dir}/data/TEST/test_reforecast.nc")
obs = xr.open_dataset(f"{base_dir}/data/TEST/test_observation.nc").squeeze()
# === Select predictors and preprocess based on target ===
if target == 't2m':
    loc = 0
    pred=12
    x = 11
    obs = obs[target]
    test_rfcs = test_rfcs.drop_vars(["u100", "w100","u", "v", "w700", "v100"])
elif target == 'w10':
    loc = 6
    pred=9
    x = 1
    obs = obs[target]
    test_rfcs = test_rfcs.drop_vars(['sd', 'mx2t6', 'mn2t6','u100','w100','v100',"u","v","w700"])
elif target == 'w100':
    loc = 8
    pred= 15
    x = 1
    obs = obs.rename({"w100_obs": "w100"})
    obs = obs[target]
    test_rfcs = test_rfcs.drop_vars(["sd", "mx2t6", "mn2t6"])
else:
    raise ValueError(f"Unsupported target: {target}")

    
oro = xr.open_dataset(f"{base_dir}/../TfMBM_wind/baselines/data/oro.nc")

min_max_values = {
    "t2m": (235, 304), "z": (48200, 58000), "t": (240, 299),
    "u10": (-13., 11.), "v10": (-30, 35), "tcc": (0., 1.0), "sd": (0, 8),
    "mx2t6": (230, 320), "mn2t6": (225, 315), "v": (-50, 55), "w100": (0, 50),
    "w10": (0, 30), "u100": (-35, 45), "u": (-45, 60), "v100": (-40, 45),
    "w700": (0, 60), "p10fg6": (0, 60), "oro": (-400, 2800)
}

dims = {'time': test_rfcs['time'], 'number': test_rfcs['number'], 'step': test_rfcs['step']}
xds_oro = oro.expand_dims(time=dims['time'], number=dims['number'], step=dims['step']).broadcast_like(
    xr.Dataset(coords=dims))

fcs_10 = xr.merge([test_rfcs, xds_oro])
dummy = xr.merge([test_rfcs, xds_oro])

dummy_norm = minmax_normalize(dummy, min_max_values)
fcs_norm = dummy_norm

fcs_array = fcs_10.to_array(dim='variable')
fcs_norm_array = fcs_norm.to_array(dim='variable')
rearranged = fcs_norm_array.transpose('time', 'number', 'step', 'latitude', 'longitude', 'variable')
rearranged_notnorm = fcs_array.transpose('time', 'number', 'step', 'latitude', 'longitude', 'variable')

fcs_whole_tensor = torch.tensor(rearranged.values)
fcs_whole_notnorm_tensor = torch.tensor(rearranged_notnorm.values)

#MBM 
MBM_one=np.empty((209,11,20,32,33),dtype=np.float32)
for i in range(20):
    mbm=np.load(f"/home/jupyter-aaron/Postprocessing/PP_EUPP/ClassicalMBM/resultsClassicalMBM/{target}/MBM_{target}_{i}_pred{x}.npy")
    MBM_one[:,:,i,:,:]=mbm.squeeze()
    
MBM_one = np.transpose(MBM_one, (1,0,2,3,4))

coords_forecast = {
        'member': test_rfcs['number'].rename({'number': 'member'}),
        'time': obs['time'], 'step': obs['step'],
        'latitude': obs['latitude'], 'longitude': obs['longitude']
    }
dims_forecast = ('member', 'time', 'step', 'latitude', 'longitude')

MBM_one = xr.DataArray(
    MBM_one,
    coords=coords_forecast,
    dims=dims_forecast
)

# Prepare raw forecasts (Raw)
raw_fcs = test_rfcs[target].rename({'number': 'member'})
raw_fcs = raw_fcs.transpose('member', 'time', 'step', 'latitude', 'longitude')
Raw = raw_fcs

# --- CRPS ---
MBM_crps = []
Raw_crps = []
for lt in range(obs.shape[1]):
    obs_lt = obs.values[:, lt, :, :]

    mbm_lt = MBM_one[:, :, lt, :, :].values
    raw_lt = Raw[:, :, lt, :, :].values

    MBM_crps.append(np.mean(almost_fair_crps_ensemble(obs_lt, mbm_lt, axis=0)))
    Raw_crps.append(np.mean(almost_fair_crps_ensemble(obs_lt, raw_lt, axis=0)))

# --- SPREAD ---
MBM_std = np.std(MBM_one.values, axis=0)
Raw_std = np.std(Raw.values, axis=0)
MBM_spread = np.sqrt(np.mean(MBM_std ** 2, axis=(0, 2, 3)))
Raw_spread = np.sqrt(np.mean(Raw_std ** 2, axis=(0, 2, 3)))

# --- MEAN ---
MBM_mu = np.mean(MBM_one.values, axis=0)
Raw_mu = np.mean(Raw.values, axis=0)

# === PRECOMPUTE SER for Raw and MBM ===
OG_SER = []
MBM_SER = []

for i in range(len(obs.step)):
    obs_i = obs.values[:, i, :, :].ravel()
    mbm_i = MBM_mu[:, i, :, :].ravel()
    og_i = Raw_mu[:, i, :, :].ravel()

    rmse_mbm = np.sqrt(np.mean((obs_i - mbm_i) ** 2))
    rmse_og = np.sqrt(np.mean((obs_i - og_i) ** 2))

    MBM_SER.append(MBM_spread[i] / rmse_mbm)
    OG_SER.append(Raw_spread[i] / rmse_og)

# === LOOP OVER WEIGHTS ===
for file in os.listdir(weights_dir):
    if 'nodecay' not in file or not file.endswith(".pth"):
        continue

    weight_path = os.path.join(weights_dir, file)
    model_name = file.replace(".pth", "")
    model_out_dir = os.path.join(output_root, model_name)
    os.makedirs(model_out_dir, exist_ok=True)

    print(f"\n⏳ Evaluating {model_name}")

    # Load model
    model = StackedTransformer(num_blocks=4, n_data_shape=(20, 32, 33, pred), n_heads=8, mlp_mult=4, projection_channels=64)
    checkpoint = torch.load(weight_path, weights_only=False)
    new_state_dict = {k.replace("module.", ""): v for k, v in checkpoint['model'].items()}
    model.load_state_dict(new_state_dict)
    model.eval()

    # Predict
    postpro = torch.empty((209, 11, 20, 32, 33, 1))
    with torch.no_grad():
        for i in range(0, 209, 2):
            end = min(i + 2, 209)
            postpro[i:end] = model(fcs_whole_tensor[i:end])

    Tformer_all = postpro[..., 0]
    scale_std, scale_mean = torch.std_mean(fcs_whole_notnorm_tensor[..., loc], dim=1)
    scaled = Tformer_all * scale_std.unsqueeze(1) + scale_mean.unsqueeze(1)
    if target != 't2m':
        scaled = torch.clamp(scaled, min=1e-6).permute(1, 0, 2, 3, 4).numpy()
    else:
        scaled = scaled.permute(1, 0, 2, 3, 4).numpy()
    
    coords_forecast = {
        'member': test_rfcs['number'].rename({'number': 'member'}),
        'time': obs['time'], 'step': obs['step'],
        'latitude': obs['latitude'], 'longitude': obs['longitude']
        }
    dims_forecast = ('member', 'time', 'step', 'latitude', 'longitude')
    Tformer_all_xr = xr.DataArray(scaled, coords=coords_forecast, dims=dims_forecast)
    # === SAVE POSTPROCESSED TRANSFORMER ENSEMBLES ===
    save_path = os.path.join(model_out_dir, "postprocessed_ensemble.npy")
    np.save(save_path, Tformer_all_xr.values)
    print(f"🧊 Saved transformer ensemble to {save_path}")
     
    # # === CRPS ===

    crps_vals = []
    print("computing CRPS")
    for lt in range(Tformer_all_xr.shape[2]):
        ens_lt = Tformer_all_xr[:, :, lt, :, :].values
        obs_lt = obs[:, lt, :, :].values
        f_crps = almost_fair_crps_ensemble(obs_lt, ens_lt, axis=0)
        crps_vals.append(np.mean(f_crps))

    Transformer_crps = crps_vals
    leadtimes = obs.step.values
    plt.figure(figsize=(10, 5))
    plt.plot(leadtimes, Raw_crps, label=f"Raw (μ={np.mean(Raw_crps):.3f})", color="red", marker='o')
    plt.plot(leadtimes, MBM_crps, label=f"MBM (μ={np.mean(MBM_crps):.3f})", color="black", marker='*')
    plt.plot(leadtimes, Transformer_crps, label=f"Transformer (μ={np.mean(Transformer_crps):.3f})", color="blue", marker='d')
    plt.axhline(np.mean(Raw_crps), color="red", linestyle=":")
    plt.axhline(np.mean(MBM_crps), color="black", linestyle=":")
    plt.axhline(np.mean(Transformer_crps), color="blue", linestyle=":")
    plt.title(f"CRPS {target}- {model_name}")
    plt.xlabel("Lead time (hours)")
    plt.ylabel("CRPS [m/s]")
    plt.grid(True)
    plt.legend()
    plt.tight_layout()
    plt.savefig(os.path.join(model_out_dir, "crps.png"))
    plt.close()

    # # === ENSEMBLE SPREAD ===
    print("Computing spread")
    std = np.std(Tformer_all_xr.values, axis=0)
    spread = np.sqrt(np.mean(std ** 2, axis=(0, 2, 3)))
    TF_spread = spread
    plt.figure(figsize=(10, 5))
    plt.plot(leadtimes / 24, Raw_spread, label=f"Raw (μ={np.mean(Raw_spread):.3f})", color="red", marker='o')
    plt.plot(leadtimes / 24, MBM_spread, label=f"MBM (μ={np.mean(MBM_spread):.3f})", color="black", marker='*')
    plt.plot(leadtimes / 24, TF_spread, label=f"Transformer (μ={np.mean(TF_spread):.3f})", color="blue", marker='d')

    plt.axhline(np.mean(Raw_spread), color="red", linestyle=":")
    plt.axhline(np.mean(MBM_spread), color="black", linestyle=":")
    plt.axhline(np.mean(TF_spread), color="blue", linestyle=":")

    plt.title(f"Spread {target} - {model_name}")
    plt.xlabel("Lead time (days)")
    plt.ylabel("Spread [m/s]")
    plt.grid(True)
    plt.legend()
    plt.tight_layout()
    plt.savefig(os.path.join(model_out_dir, "spread_comparison.png"))
    plt.close()

    # === SER ===
    print("Computing SER")
    mu = np.mean(Tformer_all_xr.values, axis=0)
    ser_vals = []
    for i in range(len(leadtimes)):
        obs_i = obs.values[:, i, :, :].ravel()
        tf_i = mu[:, i, :, :].ravel()
        rmse = np.sqrt(np.mean((obs_i - tf_i) ** 2))
        ser_vals.append(spread[i] / rmse)

    TF_SER = ser_vals

    plt.figure(figsize=(10, 5))

    plt.plot(leadtimes / 24, OG_SER, label=f"Raw (μ={np.mean(OG_SER):.3f})", color="red", marker='o')
    plt.plot(leadtimes / 24, MBM_SER, label=f"MBM (μ={np.mean(MBM_SER):.3f})", color="black", marker='*')
    plt.plot(leadtimes / 24, TF_SER, label=f"Transformer (μ={np.mean(TF_SER):.3f})", color="blue", marker='d')

    plt.axhline(1.0, color="green", linestyle="--", label="Ideal SER = 1")
    plt.axhline(np.mean(OG_SER), color="red", linestyle=":")
    plt.axhline(np.mean(MBM_SER), color="black", linestyle=":")
    plt.axhline(np.mean(TF_SER), color="blue", linestyle=":")

    plt.title(f"SER {target} - {model_name}")
    plt.xlabel("Lead time (days)")
    plt.ylabel("SER")
    plt.grid(True)
    plt.legend()
    plt.tight_layout()
    plt.savefig(os.path.join(model_out_dir, "ser_comparison.png"))
    plt.close()

    #=== SER MAP for first 4 lead times ===
    print("Plotting SER map for first 20 lead times")

    first_n = 4
    ser_map_stack = []

    for i in range(first_n):
        obs_i = obs[:, i, :, :].values
        pred_i = mu[:, i, :, :]
        rmse_map = np.sqrt(np.mean((pred_i - obs_i) ** 2, axis=0))
        spread_map = np.mean(std[:,i, :, :],axis=0)
        ser = spread_map / rmse_map
        ser_map_stack.append(ser)
    
    ser_map_avg = np.mean(ser_map_stack, axis=0) 
    

    # === Convert to xarray DataArray for geospatial plotting ===
    lat = obs.latitude.values
    lon = obs.longitude.values
    ser_xr = xr.DataArray(
        ser_map_avg,
        coords={"latitude": lat, "longitude": lon},
        dims=("latitude", "longitude"),
        name="SER"
        )

    plt.figure(figsize=(10, 6))
    ax = plt.axes(projection=ccrs.PlateCarree())
    ax.coastlines()

    vmin = ser_map_avg.min()
    vmax = ser_map_avg.max()
    vcenter = 1.0

    # Ensure valid ordering for TwoSlopeNorm
    if not (vmin < vcenter < vmax):
        print(f"Invalid norm range: vmin={vmin}, vcenter={vcenter}, vmax={vmax}")
        # fallback to a simpler normalization
        norm = None
    else:
        norm = mcolors.TwoSlopeNorm(vmin=vmin, vcenter=vcenter, vmax=vmax)

    # Plot filled contours of SER
    contour = plt.contourf(
        oro_2D['longitude'], oro_2D['latitude'], ser_map_avg,
        cmap='RdBu_r',norm=norm, transform=ccrs.PlateCarree()
    )

    # Add borders, gridlines, and labels
    ax.add_feature(cfeature.BORDERS, edgecolor='black')

    # Add colorbar
    cbar = plt.colorbar(contour, orientation='vertical', pad=0.02)
    cbar.ax.tick_params(labelsize=14)
    cbar.set_label('SER (Spread / RMSE)', fontsize=16)

    # Add gridlines with labels
    gl = ax.gridlines(draw_labels=True, linestyle='--')
    gl.top_labels = False
    gl.right_labels = False
    gl.xlabel_style = {'size': 14}
    gl.ylabel_style = {'size': 14}

    plt.title(f'SER Map {target} (Average Over 4 Lead Times)', fontsize=18)
    plt.savefig(os.path.join(model_out_dir, '4_ser_map_cartopy.png'))
    plt.close()



    # === RANK HISTOGRAM ===
    rank_hist = compute_rank_histogram(Tformer_all_xr.values, obs.values, bins=12)
    plt.figure(figsize=(10, 5))
    plt.bar(range(12), rank_hist, align='center', color='blue')
    plt.axhline(y=1/12, linestyle=':', color='black')
    plt.title(f"Rank Histogram {target} - {model_name}")
    plt.xlabel("Rank")
    plt.ylabel("Frequency")
    plt.tight_layout()
    plt.savefig(os.path.join(model_out_dir, "rank_histogram.png"))
    plt.close()

    print(f"Finished evaluation for {model_name}")

print("All model evaluations complete.")
