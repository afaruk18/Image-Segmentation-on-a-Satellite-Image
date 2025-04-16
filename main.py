import logging
import ever as er
from train import seed_torch
from ever.core.checkpoint import remove_module_prefix
from ever.core.config import import_config
from ever.core.builder import make_model
import argparse
import os
from configs.base.loveda import train, test, data, optimizer, learning_rate
import torch
from PIL import Image
import torchvision.transforms as transforms
from tqdm import tqdm
import cv2
import numpy as np
import albumentations as A
from albumentations.pytorch import ToTensorV2
from albumentations import Compose, OneOf, Normalize
from albumentations import HorizontalFlip, VerticalFlip, RandomRotate90, RandomCrop

def basic_smoothing(class_indices, kernel_size=3):
    #remove small noise
    return cv2.medianBlur(class_indices, kernel_size)


def morphological_operations(class_indices, classes_to_process=None):
    """Apply morphological operations per class to clean up segmentation"""
    result = class_indices.copy()
    if classes_to_process is None:
        classes_to_process = range(7)  # Assuming 7 classes as in your color map
    for class_id in classes_to_process:
        # Create binary mask for this class
        binary_mask = (result == class_id).astype(np.uint8)
        # Remove small isolated regions (noise)
        kernel = np.ones((3, 3), np.uint8)
        binary_mask = cv2.morphologyEx(binary_mask, cv2.MORPH_OPEN, kernel)
        # Fill small holes
        binary_mask = cv2.morphologyEx(binary_mask, cv2.MORPH_CLOSE, kernel)
        # Update the result for this class
        result[binary_mask == 1] = class_id

    return result

def class_specific_processing(class_indices, prob_accum):
    result = class_indices.copy()

    # 1. Road connectivity enhancement (class 2)
    road_mask = (result == 2).astype(np.uint8)
    # Connect nearby road segments that might be disconnected
    kernel = np.ones((5, 5), np.uint8)
    road_mask = cv2.morphologyEx(road_mask, cv2.MORPH_CLOSE, kernel)

    # 2. Water body smoothing (class 3)
    water_mask = (result == 3).astype(np.uint8)
    # Water bodies tend to be smooth and continuous
    kernel = np.ones((7, 7), np.uint8)
    water_mask = cv2.morphologyEx(water_mask, cv2.MORPH_CLOSE, kernel)

    # 3. Forest consolidation (class 5)
    forest_mask = (result == 5).astype(np.uint8)
    forest_mask = cv2.morphologyEx(forest_mask, cv2.MORPH_CLOSE, np.ones((5, 5), np.uint8))

    result[water_mask == 1] = 3  # Water first
    result[road_mask == 1] = 2  # Roads second
    result[forest_mask == 1] = 5  # Forest third

    return result

def ensemble_post_processing(class_indices, prob_accum):
    result = basic_smoothing(class_indices)
    important_classes = [2, 3, 5]  # roads, water, forest
    result = morphological_operations(result, classes_to_process=important_classes)
    result = class_specific_processing(result, prob_accum)
    return result

# Color mapping as specified
color_map = {
    0: (0, 0, 0),  # background -> black
    1: (128, 0, 0),  # building -> dark red
    2: (128, 128, 128),  # road -> gray
    3: (0, 0, 255),  # water -> blue
    4: (255, 255, 0),  # barren -> yellow
    5: (0, 128, 0),  # forest -> green
    6: (255, 243, 128)  # agriculture -> light yellow
}
config = dict(  # model config file taken from hrnet32.py
    model=dict(
        type='HRNetFusion',
        params=dict(
            backbone=dict(
                hrnet_type='hrnetv2_w32',
                pretrained=True,
                norm_eval=False,
                frozen_stages=-1,
                with_cp=False,
                with_gc=False,
            ),
            neck=dict(
                in_channels=480,
            ),
            classes=7,
            head=dict(
                in_channels=480,
                upsample_scale=4.0,
            ),
            loss=dict(
                ignore_index=-1,
                ce=dict(),
            )
        )
    ),
    data=data,
    optimizer=optimizer,
    learning_rate=learning_rate,
    train=train,
    test=test
)


def main():
    parser = argparse.ArgumentParser(description='Generate segmentation mask for satellite image.')
    parser.add_argument('--image-path', type=str, required=True, help='Path to the input satellite image.')
    parser.add_argument('--ckpt-path', type=str, default='./log/hrnetw32.pth',
                        help='Path to the pre-trained model weights')
    args = parser.parse_args()

    # Load image
    Image.MAX_IMAGE_PIXELS = None
    image = Image.open(args.image_path).convert('RGB')
    image_np = np.array(image)
    height, width = image_np.shape[:2]

    # Load model
    statedict = torch.load(args.ckpt_path, map_location=torch.device("cuda" if torch.cuda.is_available() else "cpu"))
    model_state_dict = remove_module_prefix(statedict)

    print('Load model!')
    model = make_model(config['model'])
    model.load_state_dict(model_state_dict)
    model.cuda()
    model.eval()

    # Preprocessing parameters (adjusted for LoveDA)
    mean = np.array([123.675, 116.28, 103.53]) / 255
    std = np.array([58.395, 57.12, 57.375]) / 255
    preprocess = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(mean.tolist(), std.tolist())
    ])

    # Initialize logits and count arrays
    prob_accum = np.zeros((height, width, 7), dtype=np.float32)
    count = np.zeros((height, width), dtype=np.uint8)

    tile_size = 512
    stride = 128

    # Process in batches to reduce memory pressure
    with torch.no_grad():
        for y in tqdm(range(0, height, stride)):
            for x in range(0, width, stride):
                # Calculate tile coordinates
                y_end = min(y + tile_size, height)
                x_end = min(x + tile_size, width)
                tile_height = y_end - y
                tile_width = x_end - x

                # Extract and pad tile
                tile = image_np[y:y_end, x:x_end]
                padded_tile = np.pad(tile,
                                     ((0, tile_size - tile_height),
                                      (0, tile_size - tile_width),
                                      (0, 0)),
                                     mode='reflect')

                # Process on GPU
                input_tensor = preprocess(padded_tile).unsqueeze(0).cuda()
                logits = model(input_tensor).squeeze(0)

                # Convert to probabilities while still on GPU
                probs = torch.softmax(logits, dim=0)

                # Crop to original tile size and move to CPU
                probs = probs[:, :tile_height, :tile_width].cpu().numpy()
                probs = np.transpose(probs, (1, 2, 0))

                # Accumulate probabilities
                prob_accum[y:y_end, x:x_end] += probs
                count[y:y_end, x:x_end] += 1

    # Final averaging and prediction
    prob_accum /= np.maximum(count[..., np.newaxis], 1)  # Avoid division by zero
    class_indices = np.argmax(prob_accum, axis=-1).astype(np.uint8)
    class_indices = ensemble_post_processing(class_indices, prob_accum)

    # Create and save mask
    mask_rgb = np.zeros((height, width, 3), dtype=np.uint8)
    for class_idx, color in color_map.items():
        mask_rgb[class_indices == class_idx] = color

    mask_image = Image.fromarray(mask_rgb)
    output_path = os.path.splitext(args.image_path)[0] + '_mask_.tif'
    mask_image.save(output_path)
    print(f"Segmentation mask saved to {output_path}")

if __name__ == '__main__':
    main()