import torch
import torch.nn.functional as F
   
def psnr(lr_image, hr_image, max_val=1.0):
    mse = F.mse_loss(lr_image, hr_image)
    psnr_value = 20 * torch.log10(max_val / torch.sqrt(mse))
    return psnr_value.item()

def ssim(lr_image, hr_image, data_range=1.0, window_size=11, reduction='mean'):
    ssim_value = F.ssim(lr_image, hr_image, data_range=data_range, win_size=window_size, reduction=reduction)
    return ssim_value.item()