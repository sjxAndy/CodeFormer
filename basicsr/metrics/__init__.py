from copy import deepcopy

from basicsr.utils.registry import METRIC_REGISTRY
from .psnr_ssim import calculate_psnr, calculate_ssim
from .psnr_ssim_new import calculate_psnr_new, calculate_ssim_new

__all__ = ['calculate_psnr', 'calculate_ssim', 'calculate_psnr_new', 'calculate_ssim_new']


def calculate_metric(data, opt):
    """Calculate metric from data and options.

    Args:
        opt (dict): Configuration. It must constain:
            type (str): Model type.
    """
    opt = deepcopy(opt)
    metric_type = opt.pop('type')
    metric = METRIC_REGISTRY.get(metric_type)(**data, **opt)
    return metric
