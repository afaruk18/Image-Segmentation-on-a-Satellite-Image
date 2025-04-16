import argparse
import os
import sys
import numpy as np
import torch
from PIL import Image
import logging
from skimage.io import imsave
import albumentations as A


def parse_args():
    parser = argparse.ArgumentParser(description='Satellite Image Segmentation')
    parser.add_argument('--image-path', type=str, required=True, help='Path to the input satellite image')
    parser.add_argument('--config-path', type=str, default='baseline.UNetFormer',
                        help='Config path in the format module.class')
    parser.add_argument('--ckpt-path', type=str, default='./LoveDA/Semantic_Segmentation/checkpoints/UNetFormer.pth',
                        help='Path to the pre-trained model weights')
    return parser.parse_args()


def setup_environment():
    """
    Set up the environment by adding necessary paths and importing required modules
    """
    # Add the repository root to the Python path
    sys.path.append('./LoveDA/Semantic_Segmentation')

    # Set up logging
    logger = logging.getLogger(__name__)
    logging.basicConfig(level=logging.INFO)

    # Import necessary modules from LoveDA repository
    try:
        import ever as er
        er.registry.register_all()
        from ever.core.builder import make_model
        from ever.core.checkpoint import remove_module_prefix
        from ever.core.config import import_config
    except ImportError as e:
        logger.error(f"Error importing required modules: {e}")
        logger.error("Make sure the LoveDA repository is correctly cloned and accessible.")
        sys.exit(1)

    return logger, er


def preprocess_image(image_path):
    """
    Preprocess the input image to match the format expected by the model.

    Args:
        image_path (str): Path to the input image

    Returns:
        tuple: (image_tensor, original_size)
    """
    try:
        # Load the image
        image = Image.open(image_path).convert('RGB')
    except Exception as e:
        print(f"Error loading image: {e}")
        sys.exit(1)

    # Save original dimensions for later resizing
    original_size = (image.width, image.height)

    # Convert to numpy array
    image_np = np.array(image)

    # Use the same normalization as in the config
    transform = A.Compose([
        A.Normalize(
            mean=(123.675, 116.28, 103.53),
            std=(58.395, 57.12, 57.375),
            max_pixel_value=255.0
        )
    ])

    # Apply transformation
    transformed = transform(image=image_np)
    image_np = transformed["image"]

    # Convert to tensor and add batch dimension
    image_tensor = torch.from_numpy(image_np.transpose(2, 0, 1)).float().unsqueeze(0)

    return image_tensor, original_size


def load_model(config_path, ckpt_path, er):
    """
    Load the pre-trained model using the ever framework.

    Args:
        config_path (str): Path to the model config
        ckpt_path (str): Path to the pre-trained weights
        er: The ever module

    Returns:
        torch.nn.Module: The loaded model
    """
    try:
        # Import config
        cfg = er.core.config.import_config(config_path)

        # Load state dict
        statedict = torch.load(ckpt_path, map_location=lambda storage, loc: storage)
        model_state_dict = er.core.checkpoint.remove_module_prefix(statedict)

        # Create model
        model = er.core.builder.make_model(cfg['model'])
        model.load_state_dict(model_state_dict)

        # Move to GPU if available
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        model = model.to(device)
        model.eval()

        return model, device

    except Exception as e:
        print(f"Error loading model: {e}")
        sys.exit(1)


def generate_segmentation_mask(image_tensor, model, device, original_size):
    """
    Generate a segmentation mask from the input image.

    Args:
        image_tensor (torch.Tensor): The preprocessed input image
        model (torch.nn.Module): The model to use for inference
        device (torch.device): The device to run inference on
        original_size (tuple): The original image dimensions (width, height)

    Returns:
        numpy.ndarray: The segmentation mask as an RGB image
    """
    try:
        # Move image tensor to the same device as model
        image_tensor = image_tensor.to(device)

        # Run inference
        with torch.no_grad():
            pred = model(image_tensor)
            pred = pred.argmax(dim=1).cpu().numpy()[0]  # Get first batch item

        # Resize predictions back to original image size
        pred_resized = Image.fromarray(pred.astype(np.uint8)).resize(original_size, Image.NEAREST)
        pred_resized = np.array(pred_resized)

        # Define color map as specified
        color_map = {
            0: (0, 0, 0),  # background -> black
            1: (128, 0, 0),  # building -> dark red
            2: (128, 128, 128),  # road -> gray
            3: (0, 0, 255),  # water -> blue
            4: (255, 255, 0),  # barren -> yellow
            5: (0, 128, 0),  # forest -> green
            6: (255, 243, 128)  # agriculture -> light yellow
        }

        # Create RGB mask
        height, width = pred_resized.shape
        mask = np.zeros((height, width, 3), dtype=np.uint8)

        # Apply color mapping
        for class_id, color in color_map.items():
            mask[pred_resized == class_id] = color

        return mask

    except Exception as e:
        print(f"Error generating segmentation mask: {e}")
        sys.exit(1)


def main():
    """
    Main function to run the segmentation process.
    """
    # Parse arguments
    args = parse_args()

    # Set up environment
    logger, er = setup_environment()

    # Check if the input image exists
    if not os.path.exists(args.image_path):
        logger.error(f"Input image not found: {args.image_path}")
        sys.exit(1)

    # Check if the weights path exists
    if not os.path.exists(args.ckpt_path):
        logger.error(f"Weights file not found: {args.ckpt_path}")
        sys.exit(1)

    logger.info(f"Processing image: {args.image_path}")

    # Preprocess the input image
    image_tensor, original_size = preprocess_image(args.image_path)

    # Load the model
    logger.info(f"Loading model from {args.ckpt_path}")
    model, device = load_model(args.config_path, args.ckpt_path, er)
    logger.info(f"Using device: {device}")

    # Generate segmentation mask
    logger.info("Generating segmentation mask...")
    mask = generate_segmentation_mask(image_tensor, model, device, original_size)

    # Save the mask
    input_filename = os.path.basename(args.image_path)
    input_dirname = os.path.dirname(args.image_path)
    output_filename = f"{os.path.splitext(input_filename)[0]}_segmentation.png"
    output_path = os.path.join(input_dirname, output_filename)

    # Save as PNG
    Image.fromarray(mask).save(output_path)
    logger.info(f"Segmentation mask saved to {output_path}")


if __name__ == "__main__":
    main()