# Satellite Image Segmentation

This project implements semantic segmentation for satellite imagery using pretrained weights from given source. The segmentation mask classifies each pixel into one of seven land cover categories with specific color coding.

## Installation

1. Clone the LoveDA repository:
```bash
git clone https://github.com/Junjue-Wang/LoveDA.git
cd LoveDA/Semantic_Segmentation
```

2. Install the required dependencies:
```bash
pip install -r requirements.txt
```

3. Download the pretrained model weights and place them in the `log` directory:
```bash
mkdir -p log
# Download the weights manually and place in log directory as hrnetw32.pth
```

## Usage

Run the segmentation script with:

```bash
python main.py --image-path /path/to/satellite/image.tif 
```

Arguments:
- `--image-path`: Path to the input satellite image (required)
- `--ckpt-path`: Path to the model checkpoint file (default is used: './log/hrnetw32.pth')

## Output

The script generates a segmentation mask with the same dimensions as the input image, saved in the same directory with the suffix `_outputmask_.tif`. The mask uses the following color mapping:

| Class ID | Class Name  | RGB Color         |
|----------|-------------|-------------------|
| 0        | Background  | (0, 0, 0)         |
| 1        | Building    | (128, 0, 0)       |
| 2        | Road        | (128, 128, 128)   |
| 3        | Water       | (0, 0, 255)       |
| 4        | Barren      | (255, 255, 0)     |
| 5        | Forest      | (0, 128, 0)       |
| 6        | Agriculture | (255, 243, 128)   |

## Implementation Details

### Segmentation Approach


1. **Efficient Processing for Large Image:
   - Firstly, tried to resize image and give as input. Output quality was low due to information loss with resizing. Decided to change method and below one is implemented. 
   - Divides large satellite images into overlapping tiles (512×512 pixels with 128-pixel stride)
   - Processes each tile independently and then combines the results
   - Accumulates probabilities in overlapping regions for smoother predictions

2. **Other Post-processing Methods that I tried:
   - Basic smoothing using median filtering
   - Class-specific morphological operations
   - Road connectivity enhancement, Water body smoothing, Forest consolidation.


### Considerations and Assumptions

1. **Memory Usage**: 
   - The script processes images in tiles to reduce memory consumption.
   - Stride can be lower than 128-pixel with more GPU memory for better output. However, there is tradeoff that it takes more time since there are more overlaps with lower stride.

2. **GPU Acceleration**:
   - The script uses CUDA if available, falling back to CPU if necessary.
   - Processing is significantly faster with a CUDA-compatible GPU.

3. **Model Limitations**:
   - Results is generally good. However, as I mentioned before, I have a problem that background and barren classes  are mixing with each other since we are using pretrained model.
   - I am sharing 2 outputs with you. First one is with default class maping. Second one is that I manually modified class maping by changing background and barren color map codes with     each other.  

	
## File Structure

```
.
├── main.py                # Main segmentation script (Self implemented)
├── eval.py                # Evaluation script from LoveDA
├── predict.py             # Prediction script from LoveDA
├── render.py              # Rendering script from LoveDA
├── train.py               # Training script from LoveDA
├── requirements.txt       # Dependencies
├── README.md              # This file
├── configs/               # Configuration files
│   ├── base/
│   │   └── loveda.py      # LoveDA dataset configuration	
│   ├── baseline/
│   ├── hrnet32.py         # HRNet-32 configuration
├── log/                   # Model checkpoints
│   └── hrnetw32.pth       # Pretrained weights
├── module/                # Module implementations
└── scripts/               # Helper scripts

```

## Acknowledgments

- The HRNetV2 model and pretrained weights are from the [LoveDA repository](https://github.com/Junjue-Wang/LoveDA)
- Implementation uses the Ever framework for model integration
