import torch
import torch.nn as nn
from torchvision import datasets,transforms,models
from PIL import Image
import cv2
import matplotlib.pyplot as plt
import numpy as np
from torchvision.models import ResNet18_Weights

model=models.resnet18(weights=ResNet18_Weights.DEFAULT)
model.eval()

transform=transforms.Compose([
    transforms.Resize(224),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], 
                         std=[0.229, 0.224, 0.225])
])

img_path="data/doggy.png"
img=Image.open(img_path).convert("RGB")
input_tensor=transform(img).unsqueeze(0)

gradients=[]
activations=[]

def backward_hook(module,grad_input,grad_output):
    gradients.append(grad_output[0])
def forward_hook(module,input,output):
    activations.append(output)

target_layer=model.layer4[1].conv2
target_layer.register_forward_hook(forward_hook)
target_layer.register_full_backward_hook(backward_hook)

outputs=model(input_tensor)
class_idx=outputs.argmax().item()
model.zero_grad()
outputs[0,class_idx].backward()

grads=gradients[0]
acts=activations[0].detach()

weights=grads.mean(dim=[2,3], keepdim=True)
cam=(weights*acts).sum(dim=1).squeeze()

cam=cam.cpu().numpy()
cam=np.maximum(cam,0)
cam=cam/cam.max()

heatmap=cv2.resize(cam,(224,224))
heatmap=np.uint8(255*heatmap)
heatmap=cv2.applyColorMap(heatmap,cv2.COLORMAP_JET)

original_image=cv2.cvtColor(np.array(img.resize((224,224))),cv2.COLOR_RGB2BGR)
overlay=cv2.addWeighted(original_image,0.6,heatmap,0.4,0)

plt.figure(figsize=(8,4))
plt.imshow(cv2.cvtColor(overlay,cv2.COLOR_BGR2RGB))
plt.title("grad-cam heatmap")
plt.axis("off")
plt.show()