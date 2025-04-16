# Satellite Image Segmentation

This project implements satellite image segmentation using HRNetV2 with pretrained weights from the LoveDA dataset. The segmentation mask classifies each pixel into one of seven categories: background, building, road, water, barren, forest, and agriculture.

## Features

- Processing of high-resolution satellite imagery
- Efficient tiling strategy to handle large images with minimal GPU memory
- Advanced post-processing techniques to improve segmentation quality
- Support for various image formats

## Installation

1. Clone this repository:
```bash
git clone https://github.com/yourusername/satellite-segmentation.git
cd satellite-segmentation
```

2. Install dependencies:
```bash
pip install -r requirements.txt
```

3. Download the pretrained model weights from the [LoveDA repository](https://github.com/Junjue-Wang/LoveDA/tree/master/Semantic_Segmentation) or use the weights provided in the `./log` directory.

## Usage

Run the segmentation script with the following command:

```bash
python main.py --image-path /path/to/your/satellite/image.tif --ckpt-path ./log/hrnetw32.pth
```

Arguments:
- `--image-path` (required): Path to the input satellite image
- `--ckpt-path` (optional): Path to the model checkpoint file (default: `./log/hrnetw32.pth`)

The output segmentation mask will be saved in the same directory as the input image with the suffix `_outputmask_.tif`.

## Color Mapping

The segmentation mask uses the following color mapping:

| Class ID | Class Name  | RGB Color         |
|----------|-------------|-------------------|
| 0        | Background  | (0, 0, 0)         |
| 1        | Building    | (128, 0, 0)       |
| 2        | Road        | (128, 128, 128)   |
| 3        | Water       | (0, 0, 255)       |
| 4        | Barren      | (255, 255, 0)     |
| 5        | Forest      | (0, 128, 0)       |
| 6        | Agriculture | (255, 243, 128)   |

## Optimization Strategies

To maximize segmentation quality, the following strategies were implemented:

1. **Sliding Window with Overlap**: The image is processed using a tiling strategy with overlap between adjacent tiles (stride of 64 pixels for a 512×512 tile), which helps reduce boundary artifacts.

2. **Multi-stage Post-processing**:
   - Basic smoothing with median filter to remove noise
   - Class-specific morphological operations to enhance segmentation coherence
   - Special treatment for important classes (roads, water, forest) with targeted morphological operations

3. **Probability Accumulation**: Overlapping predictions are accumulated and averaged, providing more robust predictions, especially at tile boundaries.

4. **Class-Specific Processing**:
   - Road connectivity enhancement using morphological closing
   - Water body smoothing with larger kernels to ensure continuous water regions
   - Forest consolidation to reduce fragmentation

## Considerations and Limitations

- The model is trained on the LoveDA dataset, which may have different characteristics than your target imagery. Consider fine-tuning the model if accuracy is insufficient.
- Processing very large images (>10,000×10,000 pixels) may require significant memory and processing time.
- The post-processing parameters (kernel sizes, etc.) may need adjustment for different image resolutions and landscapes.

## License

This project is based on the LoveDA repository and follows its licensing terms.

## Acknowledgments

- The HRNetV2 model and pretrained weights come from the [LoveDA repository](https://github.com/Junjue-Wang/LoveDA/tree/master/Semantic_Segmentation)
- The Ever framework is used for model implementation
