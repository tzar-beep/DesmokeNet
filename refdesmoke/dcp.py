"""Classical Dark Channel Prior desmoking (He et al., CVPR 2009) with guided-filter refinement.

Needs no training, so it is the demo's day-one method and a baseline in every table.
Inputs and outputs are float RGB arrays in [0, 1].
"""
import cv2
import numpy as np


def dark_channel(img, size=15):
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (size, size))
    return cv2.erode(img.min(axis=2), kernel)


def atmospheric_light(img, dark, top=0.001):
    n = max(1, int(dark.size * top))
    idx = np.argpartition(dark.ravel(), -n)[-n:]
    return img.reshape(-1, 3)[idx].mean(axis=0)


def guided_filter(guide, src, r=40, eps=1e-3):
    box = lambda x: cv2.boxFilter(x, -1, (r, r))
    mean_i, mean_p = box(guide), box(src)
    cov_ip = box(guide * src) - mean_i * mean_p
    var_i = box(guide * guide) - mean_i * mean_i
    a = cov_ip / (var_i + eps)
    b = mean_p - a * mean_i
    return box(a) * guide + box(b)


def dcp_desmoke(img, omega=0.95, t_min=0.1, size=15):
    """Returns (restored image, refined transmission map in [0, 1])."""
    img = img.astype(np.float32)
    dark = dark_channel(img, size)
    A = np.maximum(atmospheric_light(img, dark), 1e-3)
    t = 1 - omega * dark_channel(img / A, size)
    gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
    t = np.clip(guided_filter(gray, t.astype(np.float32)), 0, 1)
    out = (img - A) / np.maximum(t, t_min)[..., None] + A
    return np.clip(out, 0, 1), t
