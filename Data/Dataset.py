import torch
from torch.utils.data import Dataset, DataLoader
from PIL import Image
from datasets import load_dataset

class Data(Dataset):
    def __init__(self, subset, split='train'):
        self.dataset = load_dataset('eugenesiow/Div2k', subset, split=split)

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, idx):
        lr_path = self.dataset[idx]['lr']
        hr_path = self.dataset[idx]['hr']

        # Load images
        lr_image = Image.open(lr_path).convert('RGB')
        hr_image = Image.open(hr_path).convert('RGB')

        transform = torch.nn.functional.to_tensor
        lr_tensor = transform(lr_image)
        hr_tensor = transform(hr_image)

        return {'lr': lr_tensor, 'hr': hr_tensor}
