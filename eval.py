import torch
from torchvision.models import vgg19
import torch.nn.functional as F
import os
from torchvision import transforms
import time
import cv2
import numpy as np
from moviepy.editor import VideoFileClip
import torch
from PIL import Image
from torchvision.models import vgg19
import torch
import matplotlib.pyplot as plt

import cv2
from IPython.display import clear_output
from torchvision.transforms import ToTensor
import albumentations as A
from archticture import ESRGAN
from albumentations.pytorch import ToTensorV2

torch.backends.cudnn.benchmark = True
device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

esrgan = ESRGAN()
esrgan.load_models()
generator=esrgan.generator
LOW_RES_LARGE_IMAGES = 512
LOW_RES_SMALL_IMAGES = 64

test_transform_large = A.Compose(
    [
        A.Resize(width=LOW_RES_LARGE_IMAGES, height=LOW_RES_LARGE_IMAGES,interpolation=Image.LANCZOS),
        A.Normalize(mean=[0,0,0], std=[1,1,1]),
        ToTensorV2(),
    ]
)



# Define the mean and std used during normalization
MEAN = [0.5, 0.5, 0.5]
STD = [0.5, 0.5, 0.5]

def denormalize(tensor, mean, std):
    """
    Denormalizes a tensor by applying the inverse of the normalization.
    """
    for t, m, s in zip(tensor, mean, std):
        t.mul_(s).add_(m)
    return tensor

def plot_images(low_res, sr_res, save_path=None):
    num_images = 1  # This can be adjusted if multiple images need to be plotted
    fig, axes = plt.subplots(num_images, 2, figsize=(10, 5))
    
    # Handle the case where num_images is 1
    if num_images == 1:
        axes = [axes]

    for i in range(num_images):
        # Debug statements to print tensor shapes
        print(f"low_res shape: {low_res.shape}")
        print(f"sr_res shape: {sr_res.shape}")

        # Ensure tensors have the expected shape (C, H, W)
        if low_res.dim() == 4:
            low_res = low_res.squeeze(0)
        if sr_res.dim() == 4:
            sr_res = sr_res.squeeze(0)

        # Denormalize
        low_res = denormalize(low_res, MEAN, STD)
        sr_res = denormalize(sr_res, MEAN, STD)

        if low_res.shape[0] == 3:
            lr_img = np.transpose(low_res.cpu().numpy(), (1, 2, 0))
        else:
            raise ValueError(f"Unexpected low_res shape: {low_res.shape}")

        if sr_res.shape[0] == 3:
            hr_img = np.transpose(sr_res.cpu().numpy(), (1, 2, 0))
        else:
            raise ValueError(f"Unexpected sr_res shape: {sr_res.shape}")

        # Clip values to [0, 1] range before plotting
        lr_img = np.clip(lr_img, 0.0, 1.0)
        hr_img = np.clip(hr_img, 0.0, 1.0)

        axes[i][0].imshow(lr_img)
        axes[i][0].set_title('Low Resolution')
        axes[i][0].axis('off')

        axes[i][1].imshow(hr_img)
        axes[i][1].set_title('High Resolution')
        axes[i][1].axis('off')

    plt.tight_layout()
    
    if save_path is not None:
        hr_img = np.clip(hr_img, 0.0, 1.0)  # Clip values to [0, 1] range
        plt.imsave(save_path, hr_img)
    
    plt.show()

def Generate_image(input_image_path, output_image_path, LOW_RES):
    input_image = Image.open(input_image_path).convert('RGB')
    
    print("LOW_RES =", LOW_RES)
    LOW_RES = int(LOW_RES)
    if LOW_RES == -1:
        test_transform = A.Compose([
            A.Resize(width=input_image.size[0], height=input_image.size[1], interpolation=Image.LANCZOS),
            A.Normalize(mean=MEAN, std=STD),
            ToTensorV2(),
        ])
    else:
        
        test_transform = A.Compose([
            A.Resize(width=LOW_RES, height=LOW_RES, interpolation=Image.LANCZOS),
            A.Normalize(mean=MEAN, std=STD),
            ToTensorV2(),
        ])

    image = np.array(input_image)
    image = test_transform(image=image)["image"].unsqueeze(0).to(device)

    with torch.no_grad():
        generated_tensor = generator(image)

    generated_image = generated_tensor.squeeze(0)
    
    plot_images(image, generated_image, output_image_path)

    
# def Generate_image(input_image_path, output_image_path):
#     input_image = Image.open(input_image_path).convert('RGB')
#     shape = input_image.size
#     print(shape)
#     input_image = input_image.resize((240, 240))
#     preprocess = transforms.Compose([
#         # transforms.Resize((64, 64)),  # Example resizing
#         transforms.ToTensor(),  # Convert to tensor
#         transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))  # Normalize pixel values
#     ])
#     input_tensor = preprocess(input_image).unsqueeze(0).to(device)  # Add batch dimension

#     # Generate new image
#     with torch.no_grad():
#         generated_tensor = generator(input_tensor)

#     # Postprocess and save the output
#     generated_image = generated_tensor.squeeze(0).permute(1, 2, 0).cpu().numpy()
#     generated_image = (generated_image + 1) / 2.0 * 255.0  # Denormalize
#     generated_image = generated_image.astype(np.uint8)
#     output_image = Image.fromarray(generated_image)
#     output_image = output_image.resize(shape)
#     output_image.save(output_image_path)

# Generate_image("ss.png", "outpoutp.jpg")


def process_frame(frame, LOW_RES, generator, device='cuda'):
    input_image = Image.fromarray(frame).convert('RGB')
    LOW_RES = int(LOW_RES)
    
    if LOW_RES == -1:
        test_transform = A.Compose([
            A.Resize(width=input_image.size[0], height=input_image.size[1], interpolation=Image.LANCZOS),
            A.Normalize(mean=MEAN, std=STD),
            ToTensorV2(),
        ])
    else:
        test_transform = A.Compose([
            A.Resize(width=LOW_RES, height=LOW_RES, interpolation=Image.LANCZOS),
            A.Normalize(mean=MEAN, std=STD),
            ToTensorV2(),
        ])

    image = np.array(input_image)
    image = test_transform(image=image)["image"].unsqueeze(0).to(device)

    with torch.no_grad():
        generated_tensor = generator(image)

    generated_image = generated_tensor.squeeze(0).cpu()
    generated_image = denormalize(generated_image, MEAN, STD).numpy().transpose(1, 2, 0)
    generated_image = np.clip(generated_image, 0, 1) * 255
    generated_image = generated_image.astype(np.uint8)
    
    return generated_image

def process_video(input_video_path, output_video_path, LOW_RES):
    clip = VideoFileClip(input_video_path)
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    
    def process_frame_function(frame):
        return process_frame(frame, LOW_RES, generator, device)
    
    processed_clip = clip.fl_image(process_frame_function)
    processed_clip.write_videofile(output_video_path, codec='libx264')

# Example usage
input_video_path = 'output_video.mp4'  # Path to the input video
output_video_path = 'esrgan64.mp4'  # Path to save the processed video
LOW_RES = 128  # Set your desired low resolution
process_video(input_video_path, output_video_path, LOW_RES)



