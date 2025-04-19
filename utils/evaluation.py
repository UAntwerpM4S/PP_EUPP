import numpy as np 

def minmax_normalize(ds, min_max_values):
    for var in ds.data_vars:
        if var in min_max_values:
            min_val, max_val = min_max_values[var]
            ds[var] = (ds[var] - min_val) / (max_val - min_val)
    return ds

def fair_crps_ensemble(observations, forecasts, axis=0):
    forecasts = np.asarray(forecasts)
    observations = np.asarray(observations)

    if axis != 1:
        forecasts = np.moveaxis(forecasts, axis, 1)

    m = forecasts.shape[1]
    observations_expanded = np.expand_dims(observations, axis=1)

    
    dxy = np.sum(np.abs(forecasts - observations_expanded), axis=1)

    forecast_i = np.expand_dims(forecasts, axis=2)
    forecast_j = np.expand_dims(forecasts, axis=1)
    dxx = np.sum(np.abs(forecast_i - forecast_j), axis=( 1, 2))

    return dxy / m - dxx / (m * (m - 1) * 2)

def compute_rank_histogram(ensemble_forecast, observations, bins=12):
    ensemble_size = ensemble_forecast.shape[0]
    assert bins == ensemble_size + 1
    ranks = np.zeros(bins, dtype=int)
    for t in range(ensemble_forecast.shape[1]):
        for i in range(ensemble_forecast.shape[2]):
            for x in range(ensemble_forecast.shape[3]):
                for y in range(ensemble_forecast.shape[4]):
                    ensemble_vals = ensemble_forecast[:, t, i, x, y]
                    obs_val = observations[t, i, x, y]
                    rank = np.sum(ensemble_vals < obs_val)
                    ranks[rank] += 1
    return ranks / np.sum(ranks)

