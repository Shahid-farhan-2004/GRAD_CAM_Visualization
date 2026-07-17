# Grad-CAM Visualization using ResNet18 (PyTorch)

## 📌 Project Overview

This project demonstrates **Grad-CAM (Gradient-weighted Class Activation Mapping)** using a pretrained **ResNet18** model in PyTorch. Grad-CAM is an explainable AI (XAI) technique that helps visualize which parts of an image influenced the model's prediction.

Instead of treating a neural network as a "black box," Grad-CAM produces a heatmap highlighting the most important image regions responsible for the predicted class.

---

## Features

- Uses pretrained **ResNet18** from torchvision.
- Loads and preprocesses an RGB image.
- Registers forward and backward hooks.
- Computes gradients and feature maps.
- Generates a Grad-CAM heatmap.
- Overlays the heatmap on the original image.
- Displays the visualization using Matplotlib.

---

## Technologies Used

- Python
- PyTorch
- Torchvision
- OpenCV
- NumPy
- PIL (Pillow)
- Matplotlib

---

## Project Structure

```
project/
│
├── data/
│   └── doggy.png
│
├── grad_cam.py
├── README.md
└── requirements.txt
```

---

## Installation

Clone the repository

```bash
git clone <repository-url>
cd project
```

Install dependencies

```bash
pip install torch torchvision matplotlib numpy pillow opencv-python
```

---

## How It Works

### 1. Load Pretrained Model

A pretrained **ResNet18** model trained on ImageNet is loaded.

```python
model = models.resnet18(weights=ResNet18_Weights.DEFAULT)
model.eval()
```

The model is switched to evaluation mode to disable dropout and batch normalization updates.

---

### 2. Image Preprocessing

The input image is:

- Resized to **224×224**
- Converted to Tensor
- Normalized using ImageNet statistics

```python
transform = transforms.Compose([
    transforms.Resize(224),
    transforms.ToTensor(),
    transforms.Normalize(...)
])
```

The tensor shape becomes

```
(1, 3, 224, 224)
```

where

- 1 = Batch size
- 3 = RGB channels
- 224 × 224 = Image size

---

### 3. Register Hooks

Forward and backward hooks are attached to the last convolutional layer.

```python
target_layer = model.layer4[1].conv2
```

Forward hook stores

- Feature maps (activations)

Backward hook stores

- Gradients

These are required to compute Grad-CAM.

---

### 4. Forward Pass

The image is passed through the network.

```python
outputs = model(input_tensor)
```

The model outputs scores for 1000 ImageNet classes.

The predicted class is obtained by

```python
class_idx = outputs.argmax().item()
```

---

### 5. Backward Pass

Gradients are computed for only the predicted class.

```python
outputs[0, class_idx].backward()
```

This tells the model:

> "Show me which image regions contributed most to this prediction."

---

### 6. Compute Channel Weights

Average gradients are calculated for every feature map.

```python
weights = grads.mean(dim=[2,3], keepdim=True)
```

Each channel receives one importance score.

Channels with larger gradients contribute more to the final prediction.

---

### 7. Generate Grad-CAM

The feature maps are weighted and summed.

```python
cam = (weights * acts).sum(dim=1)
```

Then

- Remove negative values using ReLU
- Normalize to the range [0,1]

This produces the Grad-CAM heatmap.

---

### 8. Create Heatmap

The heatmap is

- Resized to the original image size
- Converted to color using OpenCV

```python
cv2.applyColorMap(...)
```

The **JET colormap** is used:

- Blue = Low importance
- Green = Medium importance
- Red = High importance

---

### 9. Overlay Heatmap

The heatmap is blended with the original image.

```python
overlay = cv2.addWeighted(
    original_image,
    0.6,
    heatmap,
    0.4,
    0
)
```

This creates an interpretable visualization showing where the network focused.

---

## Grad-CAM Workflow

```
Input Image
      │
      ▼
Pretrained ResNet18
      │
      ▼
Last Convolution Layer
      │
      ├──────────────► Feature Maps
      │
      ▼
Backward Pass
      │
      ▼
Gradients
      │
      ▼
Average Gradients
(Channel Weights)
      │
      ▼
Weighted Feature Maps
      │
      ▼
Sum Across Channels
      │
      ▼
ReLU
      │
      ▼
Normalize
      │
      ▼
Resize
      │
      ▼
Apply Color Map
      │
      ▼
Overlay on Original Image
      │
      ▼
Final Grad-CAM Heatmap
```

---

## Example Output

The output is a Grad-CAM visualization where:

- 🔴 Red regions indicate the most influential areas.
- 🟡 Yellow indicates moderate importance.
- 🔵 Blue indicates little or no influence.

For a dog image, the model typically focuses on:

- Face
- Eyes
- Nose
- Ears
- Body outline

rather than the background.

---

## Applications

- Explainable AI (XAI)
- Medical image analysis
- Object detection visualization
- Model debugging
- Image classification interpretation
- Deep learning education
- Research in computer vision

---

## Advantages

- Easy to implement
- Model-independent for CNNs
- Improves model interpretability
- Helps identify incorrect model attention
- Useful for debugging deep learning models

---

## Limitations

- Works primarily with convolutional neural networks.
- Produces coarse localization maps rather than precise segmentation.
- Depends on selecting an appropriate convolutional layer.
- Highlights important regions but does not explain the full reasoning process.

---

## Conclusion

This project demonstrates how Grad-CAM can be used to visualize the decision-making process of a pretrained ResNet18 model. By combining feature maps with gradient information, Grad-CAM highlights the regions that contribute most to the predicted class, making convolutional neural networks more transparent and interpretable.
