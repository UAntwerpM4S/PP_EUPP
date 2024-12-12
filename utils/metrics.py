import numpy as np
from scipy import special, integrate, stats
from torch import nn
from torch.autograd import Function
import torch
from typing import Union, Tuple, Callable


_normal_dist = torch.distributions.Normal(0., 1.)
_frac_sqrt_pi = 1 / np.sqrt(np.pi)


class CrpsGaussianLoss(nn.Module):
    """
      This is a CRPS loss function assuming a gaussian distribution. We use
      the following link:
      https://github.com/tobifinn/ensemble_transformer/blob/9b31f193048a31efd6aacb759e8a8b4a28734e6c/ens_transformer/measures.py

      """

    def __init__(self,
                 mode = 'mean',
                 eps: Union[int, float] = 1E-15):
        super(CrpsGaussianLoss, self).__init__()

        assert mode in ['mean', 'raw'], 'CRPS mode should be mean or raw'

        self.mode = mode
        self.eps = eps

    def forward(self,
                pred_mean: torch.Tensor,
                pred_stddev: torch.Tensor,
                target: torch.Tensor):
       #mean shape is torch.Size([32, 32, 1056]) target shape is torch.Size([32, 1, 1056]) 
        normed_diff = (pred_mean - target + self.eps) / (pred_stddev + self.eps)
        try:
            cdf = _normal_dist.cdf(normed_diff)
            pdf = _normal_dist.log_prob(normed_diff).exp()
        except ValueError:
            print(normed_diff)
            raise ValueError
        crps = pred_stddev * (normed_diff * (2 * cdf - 1) + 2 * pdf - _frac_sqrt_pi) #torch.Size([32, 32, 1056])
        mae=torch.mean(abs(pred_mean-target))
        if self.mode == 'mean':
            return torch.mean(crps) 
        return crps 

class CrpsGaussianTruncatedLoss(nn.Module):
    """
    This is a CRPS loss function assuming a Gaussian distribution, truncated at zero. 
    I follow Gneiting et al. (2006).
    """
    
    def __init__(self, mode='mean', eps: Union[int, float] = 1E-15):
        super(CrpsGaussianTruncatedLoss, self).__init__()

        assert mode in ['mean', 'raw'], 'CRPS mode should be mean or raw'

        self.mode = mode
        self.eps = eps
        
    def forward(self, pred_mean: torch.Tensor, pred_stddev: torch.Tensor, target: torch.Tensor):
        # mean shape is torch.Size([32, 32, 1056]) target shape is torch.Size([32, 1, 1056]) 
        normed_diff = (target - pred_mean + self.eps) / (pred_stddev + self.eps)
        quot = (pred_mean + self.eps) / (pred_stddev + self.eps)
        try:
            cdf = _normal_dist.cdf(quot)
            cdf_n = _normal_dist.cdf(normed_diff)
            cdf_s = _normal_dist.cdf(np.sqrt(2) * quot)
            pdf_n = _normal_dist.log_prob(normed_diff).exp()
        except ValueError:
            print(normed_diff)
            raise ValueError

        crps = pred_stddev * cdf**(-2) * (
            normed_diff * cdf * (2 * cdf_n + cdf - 2) + 
            2 * pdf_n * cdf - 
            (1 / np.sqrt(np.pi)) * cdf_s
        )
        return torch.mean(crps) 

