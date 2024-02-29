import numpy as np
from PIL import Image
import torch
import torch.nn.functional as F
   
def psnr(lr_image, hr_image, max_val=1.0):
    mse = F.mse_loss(lr_image, hr_image)
    psnr_value = 20 * torch.log10(max_val / torch.sqrt(mse))
    return psnr_value.item()

def ssim(lr_image, hr_image, data_range=1.0, window_size=11, reduction='mean'):
    ssim_value = F.ssim(lr_image, hr_image, data_range=data_range, win_size=window_size, reduction=reduction)
    return ssim_value.item()

def normalize(tensor, mean, std):
    """
    Normalize a tensor image with mean and standard deviation.
    Args:
        tensor (Tensor): Tensor image of size (C, H, W)
        mean (sequence): Sequence of means for each channel
        std (sequence): Sequence of standard deviations for each channel

    Returns:
        Tensor: Normalized Tensor image.
    """

    for t, m, s in zip(tensor, mean, std):
        t.sub_(m).div_(s)
    return tensor

def to_tensor(pil_img):
    """
    Convert PIL Image to torch.tensor
    
    Args:
        pil_img (ndarray or Image): Image to be converted

    Returns:
        Tensor: Converted Image
    """

    # If ndarray, simply return using torch libraries
    if isinstance(pil_img, np.ndarray):
        return torch.from_numpy(pil_img.transpose((2, 0, 1)))

    if isinstance(pil_img, Image.Image):
        img_array = np.array(pil_img)
        img_tensor = torch.from_numpy(img_array)
        
        if pil_img.mode in ('I', 'I;16', 'F', '1'):
            return img_tensor.float()
        elif pil_img.mode == 'YCbCr':
            return img_tensor.transpose(0, 2).transpose(1, 2).float()
        else:
            return img_tensor.permute(2, 0, 1).float() / 255.0

    raise ValueError("Unsupported input type. Must be PIL Image or NumPy array.")
