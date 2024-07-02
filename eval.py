import torch
from torchvision.models import vgg19
import torch.nn.functional as F
import os
from torchvision import transforms
import time
import cv2
import numpy as np
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




def plot_images(low_res, sr_res, save_path=None):
    num_images = 1
    fig, axes = plt.subplots(num_images, 2, figsize=(10, 10))
    
    # Handle the case where num_images is 1
    if num_images == 1:
        axes = [axes]

    for i in range(num_images):
        lr_img = np.transpose(np.array(low_res), (0,1,2))
        hr_img = np.transpose(np.array(sr_res), (1, 2, 0))
        
        axes[i][0].imshow(lr_img)
        axes[i][0].set_title('Low Resolution')
        axes[i][0].axis('off')
        
        axes[i][1].imshow(hr_img)
        axes[i][1].set_title('High Resolution')
        axes[i][1].axis('off')
    
    plt.tight_layout()
    
    if save_path is not None:
        # Ensure hr_img is in the range [0, 1]
        hr_img = np.clip(hr_img, 0.0, 1.0)  # Clip values to [0, 1] range
        plt.imsave(save_path, hr_img)
    
    # plt.show()
    
def Generate_image(input_image_path, output_image_path, LOW_RES):
    input_image = Image.open(input_image_path).convert('RGB')
    print("low = ",LOW_RES)
    # shape = input_image.size
    # input_image = input_image.resize((shape[0] // 2, shape[1] // 2))
        
    LOW_RES = int(LOW_RES)
    W = input_image.size[0]
    H = input_image.size[1]
    if LOW_RES != -1:
        W = LOW_RES
        H = LOW_RES

    test_transform = A.Compose(
            [
                A.Resize(width=W, height=H, interpolation=Image.LANCZOS),
                A.Normalize(mean=[0, 0, 0], std=[1, 1, 1]),
                ToTensorV2(),
            ]
        )
    image = np.array(input_image)
    image = test_transform(image=image)["image"].unsqueeze(0).to(device)

    with torch.no_grad():
        generated_tensor = generator(image)

    
    generated_image = generated_tensor.squeeze(0)
    
    plot_images(input_image, generated_image, output_image_path)


    
    
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


def preprocess_frame(frame, LOW_RES):
    image = Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)).convert('RGB')
    shape = image.size
    image = image.resize((240, 240))
    image = np.array(image)
    test_transform = A.Compose(
        [
            A.Resize(width=LOW_RES, height=LOW_RES,interpolation=Image.LANCZOS),
            A.Normalize(mean=[0,0,0], std=[1,1,1]),
            ToTensorV2(),
        ]
    )
    image = test_transform(image=image)["image"].unsqueeze(0).to(device)
    return image, shape

def postprocess_frame(tensor, shape):
    
    generated_image = tensor.squeeze(0).permute(1, 2, 0).cpu().numpy()
    generated_image = (generated_image * 255.0).clip(0, 255).astype(np.uint8)
    output_image = Image.fromarray(generated_image)
    output_image = output_image.resize(shape)
    return cv2.cvtColor(np.array(output_image), cv2.COLOR_RGB2BGR)

def process_frame(frame, LOW_RES):
    input_tensor, shape = preprocess_frame(frame, LOW_RES)
    
    with torch.no_grad():
        generated_tensor = generator(input_tensor)

    processed_frame = postprocess_frame(generated_tensor, shape)
    return processed_frame

def process_video(input_video_path, output_video_path, frame_skip, LOW_RES):
    
    cap = cv2.VideoCapture(input_video_path)
    if not cap.isOpened():
        print("Error opening video file")
        return
    
    frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = cap.get(cv2.CAP_PROP_FPS)
    frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(output_video_path, fourcc, fps, (frame_width, frame_height))
    
    frame_num = 0
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break
        
        if frame_num % frame_skip == 0:
            processed_frame = process_frame(frame, LOW_RES)
        else:
            processed_frame = frame  # Skip processing for this frame
        
        out.write(processed_frame)
        frame_num += 1
        clear_output(wait=True)
        print(f"Processed frame {frame_num}/{frame_count}")
    
    cap.release()
    out.release()
    print("Processing complete")

# Example usage


# Example usage
# input_video_path = 'video.avi'
# output_video_path = '$$$$$$$$$$.mp4'
# frame_skip = 5  # Process every second frame
# process_video(input_video_path, output_video_path, frame_skip)







