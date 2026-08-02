import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image


def psnr(lr_image, hr_image, max_val=1.0):
    mse = F.mse_loss(lr_image, hr_image)
    psnr_value = 20 * torch.log10(max_val / torch.sqrt(mse))
    return psnr_value.item()


def ssim(lr_image, hr_image, data_range=1.0, window_size=11, reduction="mean"):
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

    for t, m, s in zip(tensor, mean, std, strict=False):
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

        if pil_img.mode in ("I", "I;16", "F", "1"):
            return img_tensor.float()
        elif pil_img.mode == "YCbCr":
            return img_tensor.transpose(0, 2).transpose(1, 2).float()
        else:
            return img_tensor.permute(2, 0, 1).float() / 255.0

    raise ValueError("Unsupported input type. Must be PIL Image or NumPy array.")


def tile_image(img_tensor: torch.Tensor, patch_size: int) -> tuple[list[torch.Tensor], tuple[int, ...]]:
    """
    Breaks a large tensor into smaller patch_size x patch_size tiles.
    Handles padding if dimensions are not perfectly divisible by patch_size.

    Args:
        img_tensor (torch.Tensor): Input tensor of shape (B, C, H, W).
        patch_size (int): Size of the patches to break the image into.

    Returns:
        tuple[list[torch.Tensor], tuple[int, ...]]: List of tiles of shape (B, C, patch_size, patch_size)
        and the original shape of the image tensor.
    """
    original_shape = img_tensor.shape
    B, C, H, W = original_shape

    pad_h = (patch_size - H % patch_size) % patch_size
    pad_w = (patch_size - W % patch_size) % patch_size

    if pad_h > 0 or pad_w > 0:
        img_tensor = F.pad(img_tensor, (0, pad_w, 0, pad_h), mode='reflect')

    _, _, padded_h, padded_w = img_tensor.shape

    tiles = []
    for i in range(0, padded_h, patch_size):
        for j in range(0, padded_w, patch_size):
            tile = img_tensor[:, :, i:i+patch_size, j:j+patch_size]
            tiles.append(tile)

    return tiles, original_shape


def stitch_tiles(tiles: list[torch.Tensor], original_dims: tuple[int, ...], patch_size: int, scale_factor: int) -> torch.Tensor:
    """
    Assembles a list of upscaled tiles back into a single tensor.
    Removes any padding that was added during the tiling process.

    Args:
        tiles (list[torch.Tensor]): List of upscaled tiles of shape (B, C, patch_size*scale_factor, patch_size*scale_factor).
        original_dims (tuple[int, ...]): Original shape of the tensor before tiling (B, C, H, W).
        patch_size (int): The patch size used during tiling (before scaling).
        scale_factor (int): The scaling factor applied to the tiles.

    Returns:
        torch.Tensor: The stitched and upscaled tensor of shape (B, C, H*scale_factor, W*scale_factor).
    """
    B, C_orig, orig_h, orig_w = original_dims

    pad_h = (patch_size - orig_h % patch_size) % patch_size
    pad_w = (patch_size - orig_w % patch_size) % patch_size

    padded_h = orig_h + pad_h
    padded_w = orig_w + pad_w

    scaled_padded_h = padded_h * scale_factor
    scaled_padded_w = padded_w * scale_factor

    scaled_patch_size = patch_size * scale_factor

    out_B, out_C, _, _ = tiles[0].shape
    stitched = torch.zeros((out_B, out_C, scaled_padded_h, scaled_padded_w), dtype=tiles[0].dtype, device=tiles[0].device)

    idx = 0
    for i in range(0, scaled_padded_h, scaled_patch_size):
        for j in range(0, scaled_padded_w, scaled_patch_size):
            stitched[:, :, i:i+scaled_patch_size, j:j+scaled_patch_size] = tiles[idx]
            idx += 1

    scaled_orig_h = orig_h * scale_factor
    scaled_orig_w = orig_w * scale_factor

    stitched = stitched[:, :, :scaled_orig_h, :scaled_orig_w]
    return stitched
