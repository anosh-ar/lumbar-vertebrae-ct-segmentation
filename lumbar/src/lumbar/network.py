"""Part B: the segmentation network.

Choice and reason
  We use the residual-encoder U-Net ("ResEnc") from Isensee et al., "nnU-Net
  Revisited" (MICCAI 2024). Their fair benchmark found that a well-configured
  CNN U-Net with residual blocks in the encoder beats newer Transformer and
  Mamba designs when trained from scratch. The network code comes from their
  package ``dynamic_network_architectures``; the weights are trained by us from
  scratch (no pretrained spine model, as the assignment requires).

  We take the layout of their smallest preset ("M"): 6 resolution stages,
  32 -> 320 feature channels, residual blocks 1-3-4-6-6-6 in the encoder.
  We do NOT use the nnU-Net framework itself: it would replace our own
  preprocessing (Part A) and choose most settings automatically, and its default
  mirroring along every axis would swap L1 and L5.

Input and output
  One channel: CT in our canonical space (RAS, isotropic voxels of the size stored
  in the checkpoint: 2 mm or 1.5 mm, HU window scaled to 0..1). Eight output channels: background, L1-L5, T12, sacrum (lumbar/labels.py).
  T12 and sacrum are "anchor" classes: they help the network count the levels;
  only L1-L5 are reported.
"""

from __future__ import annotations

import torch
from torch import nn
from dynamic_network_architectures.architectures.unet import ResidualEncoderUNet

from .labels import NUM_TRAIN_CLASSES

# The patch is fixed in centimetres, not voxels: 19.2 x 19.2 x 32 cm (x, y, z).
# Chosen from the 2 mm training crops (median 86 x 82 x 152 voxels, 95 % within
# 98 x 97 x 208): one patch sees almost the whole region T12..sacrum, which the
# network needs to count levels. In voxels every side must be divisible by 32,
# because the network halves the image 5 times.
PATCH_SIZE_MM = (192.0, 192.0, 320.0)
DOWNSAMPLING_FACTOR = 32


def patch_size_for_spacing(spacing_mm: float) -> tuple[int, int, int]:
    """Patch in voxels for a voxel size: each side rounded to the nearest multiple of 32.
    2 mm -> (96, 96, 160) = 19.2 x 19.2 x 32 cm; 1.5 mm -> (128, 128, 224) = 19.2 x 19.2 x 33.6 cm."""
    f = DOWNSAMPLING_FACTOR
    return tuple(max(f, int(round(mm / spacing_mm / f)) * f) for mm in PATCH_SIZE_MM)

NETWORK_SETTINGS = {
    "input_channels": 1,
    "n_stages": 6,
    "features_per_stage": [32, 64, 128, 256, 320, 320],
    "kernel_sizes": 3,
    "strides": [1, 2, 2, 2, 2, 2],
    "n_blocks_per_stage": [1, 3, 4, 6, 6, 6],
    "n_conv_per_stage_decoder": [1, 1, 1, 1, 1],
    "num_classes": NUM_TRAIN_CLASSES,
}


def build_lumbar_segmentation_network(settings: dict = NETWORK_SETTINGS) -> nn.Module:
    """The ResEnc U-Net with the nnU-Net default building blocks: 3D convolutions,
    instance normalisation, leaky ReLU, no deep supervision (one output only)."""
    return ResidualEncoderUNet(
        conv_op=nn.Conv3d, conv_bias=True,
        norm_op=nn.InstanceNorm3d, norm_op_kwargs={"eps": 1e-5, "affine": True},
        nonlin=nn.LeakyReLU, nonlin_kwargs={"inplace": True},
        deep_supervision=False, **settings)


def load_trained_network(weights_path, device: str | torch.device = "cpu") -> tuple[nn.Module, dict]:
    """Network with trained weights in evaluation mode, plus the checkpoint's
    stored settings (patch size, spacing, HU window, ...)."""
    ckpt = torch.load(weights_path, map_location=device, weights_only=False)
    net = build_lumbar_segmentation_network(ckpt["network_settings"])
    net.load_state_dict(ckpt["model_state"])
    return net.to(device).eval(), ckpt
