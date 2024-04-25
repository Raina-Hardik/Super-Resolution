import matplotlib.pyplot as plt
from torch.utils.data import DataLoader
from data.dataset import Data

import torchvision.transforms.functional as F

# Instantiate your dataset
dataset = Data()

# Create a DataLoader to iterate through the dataset
dataloader = DataLoader(dataset, batch_size=1, shuffle=True)

# Display three pairs of LR and HR images
for i, batch in enumerate(dataloader):
    lr_image, hr_image = batch['lr'][0], batch['hr'][0]  # Take the first item from the batch
    lr_image = F.to_pil_image(lr_image)
    hr_image = F.to_pil_image(hr_image)

    # Display the LR image
    plt.subplot(3, 2, 2 * i + 1)
    plt.imshow(lr_image)
    plt.title('LR Image')
    plt.axis('off')

    # Display the HR image
    plt.subplot(3, 2, 2 * i + 2)
    plt.imshow(hr_image)
    plt.title('HR Image')
    plt.axis('off')

    if i == 2:  # Display only 3 pairs
        break

plt.show()
