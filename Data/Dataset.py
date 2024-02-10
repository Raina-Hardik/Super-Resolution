from torch.utils.data import Dataset
from torchvision.transforms import ToTensor, Compose
from PIL.Image import open
from torch import randint
import torchvision.transforms.functional as F
from datasets import load_dataset


class Data(Dataset):
    def __init__(self, subset="bicubic_x4", split='train'):
        self.dataset = load_dataset('eugenesiow/Div2k', subset, split=split)
        self.compose = Compose([ToTensor()])

    def transform(self, lr_image, hr_image):
        lr_image, hr_image = self.compose(lr_image), self.compose(hr_image)

        if randint(2, (1,)).item():
            lr_image, hr_image = F.hflip(lr_image), F.hflip(hr_image)
        
        rn = randint(4, (1,)).item()
        return F.rotate(lr_image, rn * 90), F.rotate(hr_image, rn * 90)

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, idx):
        lr_path = self.dataset[idx]['lr']
        hr_path = self.dataset[idx]['hr']

        lr_image = open(lr_path).convert('RGB')
        hr_image = open(hr_path).convert('RGB')

        lr_image, hr_image = self.transform(lr_image, hr_image)

        return {'lr':lr_image, 'hr': hr_image}

