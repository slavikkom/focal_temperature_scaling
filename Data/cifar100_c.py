"""
Create test iterators for CIFAR-100-C.

CIFAR-100-C stores each corruption as a 50,000-image .npy file:
five severity levels x 10,000 CIFAR-100 test images. Severity 1 is
indices [0:10000], severity 5 is [40000:50000].
"""

import os

import numpy as np
import torch
from PIL import Image
from torch.utils.data import ConcatDataset, Dataset
from torchvision import transforms

import Data.cifar100 as cifar100


CIFAR100_C_CORRUPTIONS = [
    "brightness",
    "contrast",
    "defocus_blur",
    "elastic_transform",
    "fog",
    "frost",
    "gaussian_blur",
    "gaussian_noise",
    "glass_blur",
    "impulse_noise",
    "jpeg_compression",
    "motion_blur",
    "pixelate",
    "saturate",
    "shot_noise",
    "snow",
    "spatter",
    "speckle_noise",
    "zoom_blur",
]


def _resolve_cifar100_c_root(data_dir):
    candidates = [
        data_dir,
        os.path.join(data_dir, "CIFAR-100-C"),
        os.path.join(data_dir, "CIFAR100-C"),
        os.path.join(".", "Data", "datasets", "CIFAR-100-C"),
        os.path.join(".", "Data", "datasets", "CIFAR100-C"),
    ]
    for candidate in candidates:
        if candidate and os.path.exists(os.path.join(candidate, "labels.npy")):
            return candidate
    raise FileNotFoundError(
        "Could not find CIFAR-100-C. Expected labels.npy in one of: "
        + ", ".join(candidates)
    )


def _parse_severity(severity):
    if severity is None:
        return None
    if isinstance(severity, str):
        severity = severity.lower()
        if severity in ("all", "none"):
            return None
    severity = int(severity)
    if severity < 1 or severity > 5:
        raise ValueError("CIFAR-100-C severity must be one of 1, 2, 3, 4, 5, or 'all'.")
    return severity


def _severity_slice(severity):
    severity = _parse_severity(severity)
    if severity is None:
        return slice(None)
    start = (severity - 1) * 10000
    return slice(start, start + 10000)


class CIFAR100C(Dataset):
    """
    CIFAR-100-C dataset for one corruption and one severity level.

    Args:
        root: Directory containing CIFAR-100-C .npy files.
        corruption: One corruption name, e.g. 'gaussian_noise'.
        severity: 1..5, or 'all' to use all severities for this corruption.
        transform: Transform applied to each PIL image.
    """

    def __init__(self, root, corruption="gaussian_noise", severity=1, transform=None):
        if corruption not in CIFAR100_C_CORRUPTIONS:
            raise ValueError(
                "Unknown CIFAR-100-C corruption '{}'. Valid corruptions are: {}".format(
                    corruption, ", ".join(CIFAR100_C_CORRUPTIONS)
                )
            )

        self.root = _resolve_cifar100_c_root(root)
        self.corruption = corruption
        self.severity = severity
        self.transform = transform

        data_path = os.path.join(self.root, corruption + ".npy")
        labels_path = os.path.join(self.root, "labels.npy")
        if not os.path.exists(data_path):
            raise FileNotFoundError("Missing CIFAR-100-C corruption file: {}".format(data_path))

        data_slice = _severity_slice(severity)
        self.data = np.load(data_path, mmap_mode="r")[data_slice]
        self.targets = np.load(labels_path, mmap_mode="r")[data_slice]

    def __getitem__(self, index):
        img = Image.fromarray(self.data[index])
        target = int(self.targets[index])

        if self.transform is not None:
            img = self.transform(img)

        return img, target

    def __len__(self):
        return len(self.targets)


def get_train_valid_loader(batch_size,
                           augment,
                           random_seed,
                           data_dir='./data',
                           valid_size=0.1,
                           shuffle=True,
                           num_workers=4,
                           pin_memory=False,
                           get_val_temp=0,
                           smoke_test=False):
    """
    Return clean CIFAR-100 train/validation loaders.

    CIFAR-100-C is test-only, so calibration still uses the clean CIFAR-100
    validation split used for the original CIFAR-100 models.
    """
    return cifar100.get_train_valid_loader(
        batch_size=batch_size,
        augment=augment,
        random_seed=random_seed,
        data_dir=data_dir,
        valid_size=valid_size,
        shuffle=shuffle,
        num_workers=num_workers,
        pin_memory=pin_memory,
        get_val_temp=get_val_temp,
        smoke_test=smoke_test,
    )


def get_test_loader(batch_size,
                    data_dir='./Data/datasets',
                    corruption='gaussian_noise',
                    severity=1,
                    shuffle=False,
                    num_workers=4,
                    pin_memory=False,
                    smoke_test=False):
    """
    Return a CIFAR-100-C test loader.

    Args:
        corruption: A corruption name from CIFAR100_C_CORRUPTIONS, or 'all'.
        severity: 1..5, or 'all'.
    """
    normalize = transforms.Normalize(
        mean=[0.485, 0.456, 0.406],
        std=[0.229, 0.224, 0.225],
    )
    transform = transforms.Compose([
        transforms.ToTensor(),
        normalize,
    ])

    root = _resolve_cifar100_c_root(data_dir)
    if corruption == "all":
        datasets = [
            CIFAR100C(root=root, corruption=name, severity=severity, transform=transform)
            for name in CIFAR100_C_CORRUPTIONS
        ]
        dataset = ConcatDataset(datasets)
    else:
        dataset = CIFAR100C(
            root=root,
            corruption=corruption,
            severity=severity,
            transform=transform,
        )

    if smoke_test:
        indices = np.arange(min(100, len(dataset)))
        dataset = torch.utils.data.Subset(dataset, indices)

    data_loader = torch.utils.data.DataLoader(
        dataset, batch_size=batch_size, shuffle=shuffle,
        num_workers=num_workers, pin_memory=pin_memory,
    )

    return data_loader
