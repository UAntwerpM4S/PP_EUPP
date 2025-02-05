# Self-attentive transformer for postprocessing ensemble weather forecasts
This is the code corresponding to the paper https://arxiv.org/abs/2412.13957. The transformer described here is based on the original transformer by Finn (2021) and some parts of the code by Ashkboos et al (2022).  The transformer presented here is designed for the fast and accurate postprocessing of ensemble weather forecasts, while allowing for a flexible inclusion of predictors and lead times. The self-attention mechanism allows for information exchange across predictor, spatial, temporal and ensemble dimension. The transformer is trained on the EUPPBenchmark dataset of Demaeyer et al. (2023) and
as a competitive benchmark, we used the classical MBM method of Van Schaeybroeck & Vannitsem (2015). 

### Structure




#### References
 * https://arxiv.org/abs/2106.13924
 * https://proceedings.neurips.cc/paper_files/paper/2022/hash/89e44582fd28ddfea1ea4dcb0ebbf4b0-Abstract-Datasets_and_Benchmarks.html.
 * https://essd.copernicus.org/articles/15/2635/2023/essd-15-2635-2023.html
 * https://rmets.onlinelibrary.wiley.com/doi/full/10.1002/qj.2397
