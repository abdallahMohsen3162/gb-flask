import os
import numpy as np
import matplotlib.pyplot as plt
import torch
from torch.utils.data import Dataset
from PIL import Image
from albumentations.pytorch import ToTensorV2
import albumentations as A

# Configuration
LOAD_MODEL = True
SAVE_MODEL = True
CHECKPOINT_GEN = "/kaggle/working/gen.pth"
CHECKPOINT_DISC = "/kaggle/working/disc.pth"
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
LEARNING_RATE = 2e-4
NUM_EPOCHS = 100
BATCH_SIZE = 16
LAMBDA_GP = 10
NUM_WORKERS = 8
HIGH_RES = 128
LOW_RES = HIGH_RES // 4
IMG_CHANNELS = 3

# Transformations
highres_transform = A.Compose([
    A.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5]),
    ToTensorV2(),
])

NOISE_LEVEL = 0.1

lowres_transform = A.Compose([
    A.Resize(width=LOW_RES, height=LOW_RES, interpolation=Image.BICUBIC),
    A.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5]),
    A.GaussNoise(var_limit=(0, NOISE_LEVEL**2)),
    ToTensorV2(),
])

both_transforms = A.Compose([
    A.Resize(width=HIGH_RES, height=HIGH_RES, interpolation=Image.LANCZOS),
    A.HorizontalFlip(p=0.5),
    A.RandomRotate90(p=0.5),
])

test_transform = A.Compose([
    A.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5]),
    ToTensorV2(),
])

# Custom Dataset
class MyImageFolder(Dataset):
    def __init__(self, root_dirs):
        super(MyImageFolder, self).__init__()
        self.root_dirs = root_dirs
        self.image_files = self.collect_image_files()

    def collect_image_files(self):
        image_files = []
        for root_dir in self.root_dirs:
            try:
                image_files.extend([
                    os.path.join(root_dir, f) 
                    for f in os.listdir(root_dir) 
                    if os.path.isfile(os.path.join(root_dir, f))
                ])
            except OSError as e:
                print(f"Error reading directory {root_dir}: {e}")
        return image_files

    def __len__(self):
        return len(self.image_files)

    def __getitem__(self, index):
        img_path = self.image_files[index]
        image = Image.open(img_path).convert("RGB")
        image = np.array(image)

        image = both_transforms(image=image)["image"]
        high_res = highres_transform(image=image)["image"]
        low_res = lowres_transform(image=image)["image"]

        return low_res, high_res

def custom_collate_fn(batch):
    return torch.stack([item[0] for item in batch]), torch.stack([item[1] for item in batch])

# Function to save images
def plot_and_save(low_res, high_res, save_dir):
    num_images = len(low_res)
    fig, axes = plt.subplots(num_images, 2, figsize=(10, 10))
    fig.patch.set_facecolor('white')

    if not os.path.exists(save_dir):
        os.makedirs(save_dir)

    for i in range(num_images):
        lr_img = np.transpose(low_res[i].numpy(), (1, 2, 0))
        hr_img = np.transpose(high_res[i].numpy(), (1, 2, 0))

        lr_img = (lr_img + 0.5).clip(0, 1)
        hr_img = (hr_img + 0.5).clip(0, 1)

        axes[i, 0].imshow(lr_img)
        axes[i, 0].set_title('Low Resolution')
        axes[i, 0].axis('off')
        
        axes[i, 1].imshow(hr_img)
        axes[i, 1].set_title('High Resolution')
        axes[i, 1].axis('off')

        lr_save_path = os.path.join(save_dir, f"low_res_{i}.png")
        plt.imsave(lr_save_path, lr_img)

    plt.tight_layout()
    plt.show()

# Main function
def main():
    root_dirs = [r'E:\GP\flask\DIV2K_valid_HR']
    dataset = MyImageFolder(root_dirs)
    print(f"Total number of images to train: {len(dataset)}")

    save_dir = 'saved_images'

    low_res_images = []
    high_res_images = []

    for i in range(min(100, len(dataset))):
        low_res, high_res = dataset[i]
        low_res_images.append(low_res)
        high_res_images.append(high_res)
        
        if (i + 1) % BATCH_SIZE == 0 or i == 99:
            plot_and_save(torch.stack(low_res_images), torch.stack(high_res_images), save_dir)
            low_res_images = []
            high_res_images = []

if __name__ == "__main__":
    main()
