# Computer Vision - Basics

## What is Computer Vision?

Computer Vision (CV) is a field of artificial intelligence that enables computers to interpret and process visual information from images and videos, similar to how humans see and understand the visual world.

## Key Applications

- **Image Classification**: Categorizing images into predefined classes
- **Object Detection**: Locating and identifying objects in images
- **Image Segmentation**: Dividing images into meaningful regions
- **Facial Recognition**: Identifying or verifying faces
- **Motion Detection**: Tracking movement in videos
- **Medical Imaging**: Analyzing medical scans and images
- **Autonomous Vehicles**: Scene understanding for navigation
- **Quality Control**: Automated inspection in manufacturing

## Image Representation

### Digital Images

Images are represented as 2D arrays of pixels. Each pixel contains intensity values:

- **Grayscale images**: Single channel, values from 0-255
- **Color images**: Three channels (RGB), each 0-255

$$
\text{Image} = \begin{bmatrix}
p_{11} & p_{12} & \cdots & p_{1n} \\
p_{21} & p_{22} & \cdots & p_{2n} \\
\vdots & \vdots & \ddots & \vdots \\
p_{m1} & p_{m2} & \cdots & p_{mn}
\end{bmatrix}
$$

Where $p_{ij}$ is the pixel intensity at position $(i,j)$.

### Color Spaces

| Color Space | Channels | Use Case |
|-------------|----------|----------|
| RGB | Red, Green, Blue | Display, most common |
| Grayscale | Intensity | Processing, efficient |
| HSV | Hue, Saturation, Value | Color-based detection |
| YCbCr | Luma, Chroma | Video compression |

## Fundamental Concepts

### Filters and Convolution

A filter (kernel) is a small matrix that slides over an image to extract features:

$$
\text{Output}(i,j) = \sum_{u,v} \text{Image}(i+u, j+v) \cdot \text{Filter}(u,v)
$$

Common filters:
- **Sobel**: Edge detection
- **Gaussian**: Blurring, smoothing
- **Laplacian**: Second derivative edge detection

### Feature Detection

**Features** are distinctive patterns that can be detected and matched:

- **Edges**: Boundaries between regions with different intensities
- **Corners**: Points where edges change direction
- **Blobs**: Regions of similar intensity

Popular feature detectors:
- Harris Corner Detection
- SIFT (Scale-Invariant Feature Transform)
- SURF (Speeded Up Robust Features)
- ORB (Oriented FAST and Rotated BRIEF)

### Image Transformations

| Transformation | Description | Use Case |
|---------------|-------------|----------|
| Rotation | Rotate image by angle θ | Data augmentation |
| Scaling | Resize image | Model input |
| Translation | Shift image | Invariance testing |
| Affine | Linear transformation | Perspective correction |
| Perspective | Non-linear transformation | 3D view changes |

## Deep Learning in Vision

### Convolutional Neural Networks (CNNs)

CNNs are specialized neural networks for image processing with:
- **Convolutional layers**: Extract local features
- **Pooling layers**: Reduce spatial dimensions
- **Fully connected layers**: Make final predictions

See [CNN](CNN.md) for detailed information.

## Image Processing Pipeline

```
Raw Image
    ↓
Preprocessing (normalization, resizing)
    ↓
Feature Extraction
    ↓
Feature Processing
    ↓
Classification/Detection/Segmentation
    ↓
Post-processing
    ↓
Output/Decision
```

### Preprocessing Steps

1. **Normalization**: Scale pixel values to [0, 1] or [-1, 1]
2. **Resizing**: Standardize image dimensions
3. **Color Space Conversion**: Convert to appropriate color space
4. **Augmentation**: Apply transformations for training data

## Common Vision Tasks

### Image Classification
Assign an image to one or more predefined categories.

Example: Is this image a cat or a dog?

### Object Detection
Locate and classify objects within an image.

Example: Find all people in an image and draw bounding boxes.

### Segmentation
Classify each pixel into a category.

**Types:**
- **Semantic Segmentation**: Each pixel classified to one class
- **Instance Segmentation**: Each object instance separately identified

### Pose Estimation
Detect and locate human body keypoints.

Example: Detect head, shoulders, elbows, wrists, etc.

## Challenges in Computer Vision

- **Variation**: Scale, rotation, illumination changes
- **Occlusion**: Objects partially hidden
- **Background Clutter**: Difficulty distinguishing objects from background
- **Viewpoint Variation**: Different perspectives
- **Computational Cost**: Processing high-resolution images
- **Limited Data**: Insufficient labeled training examples

## Metrics for Vision Tasks

### Classification
- Accuracy
- Precision
- Recall
- F1-score
- Confusion Matrix

### Object Detection
- IoU (Intersection over Union)
- mAP (mean Average Precision)
- AP (Average Precision)

### Segmentation
- IoU (per class and mean)
- Dice Coefficient
- Pixel Accuracy

## Popular Datasets

- **MNIST**: Handwritten digits (70K images)
- **CIFAR-10/100**: Small images, 10/100 classes
- **ImageNet**: 14M images, 1000 classes
- **COCO**: Objects in context, detection/segmentation
- **Cityscapes**: Urban street scenes, segmentation

## Tools and Libraries

| Library | Purpose |
|---------|---------|
| OpenCV | Image processing, feature detection |
| PIL/Pillow | Image manipulation |
| scikit-image | Image processing algorithms |
| TensorFlow/PyTorch | Deep learning models |
| YOLO | Object detection |
| Detectron2 | Instance segmentation |

## Further Reading

- [Convolution Operations](Convolution.md)
- [Convolutional Neural Networks](CNN.md)
- [Data Processing](DataProcessing.md)
