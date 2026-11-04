"""Grad-CAM: mapa de calor de las zonas de la radiografía que más pesan en la predicción."""
import numpy as np
import torch


class GradCAM:
    def __init__(self, model, layer):
        self.model, self.acts, self.grads = model, None, None
        layer.register_forward_hook(lambda m, i, o: setattr(self, "acts", o.detach()))
        layer.register_full_backward_hook(lambda m, gi, go: setattr(self, "grads", go[0].detach()))

    def __call__(self, x, cls=1):
        self.model.eval()
        x = x.clone().requires_grad_(True)
        out = self.model(x)
        self.model.zero_grad()
        out[0, cls].backward()
        w = self.grads.mean(dim=(2, 3), keepdim=True)
        cam = torch.relu((w * self.acts).sum(1))[0]
        cam = cam / (cam.max() + 1e-8)
        prob = torch.softmax(out, 1)[0, cls].item()
        return cam.cpu().numpy(), prob


def overlay(gray, cam, alpha=0.45):
    """gray: HxW en [0,1]; cam: hxw en [0,1] -> imagen RGB HxWx3 con el mapa de calor encima."""
    import matplotlib.cm as cm
    from PIL import Image
    h, w = gray.shape
    cam = np.array(Image.fromarray((cam * 255).astype(np.uint8)).resize((w, h), Image.BILINEAR)) / 255.0
    heat = cm.jet(cam)[..., :3]
    base = np.stack([gray] * 3, -1)
    return np.clip((1 - alpha) * base + alpha * heat, 0, 1)
