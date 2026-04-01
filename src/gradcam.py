import torch
import torch.nn.functional as F
import cv2
import numpy as np


class GradCAM:
    def __init__(self, model, target_layer):
        self.model = model
        self.model.eval()

        self.gradients = None
        self.activations = None

        # Hook the target layer
        def forward_hook(module, input, output):
            self.activations = output.detach()

        def backward_hook(module, grad_in, grad_out):
            self.gradients = grad_out[0].detach()

        target_layer.register_forward_hook(forward_hook)
        target_layer.register_backward_hook(backward_hook)

    def generate(self, input_tensor, class_idx=None):
        # Forward pass
        output = self.model(input_tensor)

        if class_idx is None:
            class_idx = output.argmax(dim=1).item()

        # Backward pass for selected class
        self.model.zero_grad()
        output[:, class_idx].backward(retain_graph=True)

        gradients = self.gradients
        activations = self.activations

        # Global average pooling of gradients
        weights = gradients.mean(dim=(2, 3), keepdim=True)

        # Weighted sum of activations
        cam = (weights * activations).sum(dim=1).squeeze()

        # Normalize CAM
        cam = cam.cpu().numpy()
        cam = np.maximum(cam, 0)
        cam = cam / cam.max()

        return cam, class_idx


def overlay_cam(img, cam):
    # Resize CAM to match input image resolution
    cam_resized = cv2.resize(cam, (img.shape[1], img.shape[0]))

    heatmap = cv2.applyColorMap(np.uint8(255 * cam_resized), cv2.COLORMAP_JET)
    heatmap = cv2.cvtColor(heatmap, cv2.COLOR_BGR2RGB)

    # Blend heatmap with original
    overlay = 0.4 * heatmap + 0.6 * img
    return np.uint8(overlay)