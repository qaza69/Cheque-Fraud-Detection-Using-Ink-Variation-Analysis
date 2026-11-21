"""Grad-CAM explanations for the ink CNNs.

A Grad-CAM map shows which ink pixels of a 64x64 optical-density patch pushed a
network towards its output.  The maps are built with forward/backward hooks on
the last convolution layer; the hooks are always removed afterwards, so calling
the helpers repeatedly never stacks hooks on the model.
"""
import cv2
import numpy as np
import torch
import torch.nn.functional as F


class GradCAM:
    """Context manager that records one conv layer's activations and gradients."""

    def __init__(self, layer):
        self._layer = layer
        self._seen = {}
        self._handles = []

    # -- hook management -------------------------------------------------
    def __enter__(self):
        self._handles = [
            self._layer.register_forward_hook(
                lambda module, args, out: self._seen.update(act=out.detach())),
            self._layer.register_full_backward_hook(
                lambda module, g_in, g_out: self._seen.update(grad=g_out[0].detach())),
        ]
        return self

    def __exit__(self, *exc):
        self.remove()
        return False

    def remove(self):
        """Detach both hooks (safe to call more than once)."""
        for handle in self._handles:
            handle.remove()
        self._handles = []

    # -- result ----------------------------------------------------------
    def heatmap(self, size):
        """Normalised (0..1) importance map resized to `size` = (height, width)."""
        act, grad = self._seen.get("act"), self._seen.get("grad")
        if act is None or grad is None:
            return np.zeros(size, np.float32)
        channel_weight = grad.mean(dim=(2, 3), keepdim=True)
        cam = F.relu((channel_weight * act).sum(dim=1, keepdim=True))
        cam = F.interpolate(cam, size=size, mode="bilinear", align_corners=False)[0, 0]
        cam = cam.cpu().numpy()
        span = float(cam.max() - cam.min())
        return (cam - cam.min()) / span if span > 1e-8 else np.zeros_like(cam)

    @staticmethod
    def overlay(image, heatmap, alpha=0.5):
        """Blend a 0..1 heat-map over an RGB image (uint8 or float 0..1)."""
        if image.dtype != np.uint8:
            image = (np.clip(image, 0, 1) * 255).astype(np.uint8)
        height, width = image.shape[:2]
        resized = cv2.resize(heatmap.astype(np.float32), (width, height))
        colour = cv2.applyColorMap((resized * 255).astype(np.uint8), cv2.COLORMAP_JET)
        colour = cv2.cvtColor(colour, cv2.COLOR_BGR2RGB)
        return cv2.addWeighted(image, 1 - alpha, colour, alpha, 0)


def _explain(encoder, patch, score_fn):
    """Back-propagate `score_fn(input)` and return the ink-masked Grad-CAM map."""
    device = next(encoder.parameters()).device
    x = torch.as_tensor(patch[None], dtype=torch.float32, device=device).requires_grad_(True)
    with GradCAM(encoder.last_conv) as cam, torch.enable_grad():
        score_fn(x).backward()
        heat = cam.heatmap(patch.shape[1:])
    return heat * (patch.sum(axis=0) > 0)           # only ink pixels can be "important"


def gradcam_pen_classifier(net, patch):
    """Evidence for the pen that the CNN classifier predicts for `patch`."""
    net.eval()

    def predicted_logit(x):
        logits = net(x)
        return logits[0, logits.argmax(dim=1).item()]

    return _explain(net.encoder, patch, predicted_logit)


def gradcam_siamese(net, patch, reference_patch):
    """Evidence that `patch` carries a different ink than `reference_patch`."""
    net.eval()
    device = next(net.parameters()).device
    reference = torch.as_tensor(reference_patch[None], dtype=torch.float32, device=device)
    with torch.no_grad():
        reference_embedding = net.embed(reference)

    return _explain(net.encoder, patch,
                    lambda x: net.pair_logit(net.embed(x), reference_embedding)[0])


def patch_to_rgb(patch):
    """Draw an optical-density patch (3, H, W) as ink on white paper."""
    density = patch.transpose(1, 2, 0) * 2.0         # undo the OD_SCALE used for the networks
    bgr = np.clip(np.exp(-density) * 255, 0, 255).astype(np.uint8)
    return cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
