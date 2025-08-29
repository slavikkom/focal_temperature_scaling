
import medmnist
import torch
import numpy as np
from torchvision.transforms import Compose, ToTensor, Normalize

from PIL import Image

class PathMNISTWrapper(medmnist.PathMNIST):
    def __getitem__(self, index):
        """
        return: (without transform/target_transofrm)
            img: PIL.Image
            target: np.array of `L` (L=1 for single-label)
        """
        img, target = self.imgs[index], self.labels[index].astype(int)
        img = Image.fromarray(img)

        if self.as_rgb:
            img = img.convert("RGB")

        if self.transform is not None:
            img = self.transform(img)

        if self.target_transform is not None:
            target = self.target_transform(target)

        return img, target.squeeze()


def get_data_loader(root,
                    batch_size,
                    split='train',
                    shuffle=True,
                    num_workers=4,
                    pin_memory=False,
                    smoke_test=False):
    """
    Utility function for loading and returning train and valid
    multi-process iterators over the PathMNIST dataset. 
    
    Params
    ------
    - root: The root directory for TinyImagenet dataset
    - batch_size: how many samples per batch to load.
    - split: Can be train/val/test. For train we apply the data augmentation techniques.
    - shuffle: whether to shuffle the train/validation indices.
    - num_workers: number of subprocesses to use when loading the dataset.
    - pin_memory: whether to copy tensors into CUDA pinned memory. Set it to
      True if using GPU.
    Returns
    -------
    - train_loader: training set iterator.
    - valid_loader: validation set iterator.
    """
    # Define transforms
    transform = Compose([
        ToTensor(),
        Normalize(mean=[0.5], std=[0.5])  # Normalize for grayscale images
    ])

    # load the dataset
    data_dir = root

    # Load PathMNIST dataset
    dataset = PathMNISTWrapper(root=data_dir, 
                                split=split, 
                                transform=transform, 
                                download=True)

    if smoke_test:
        indices = np.arange(100)  # Use only the first 100 samples
        dataset = torch.utils.data.Subset(dataset, indices)

    # Create DataLoader
    data_loader = torch.utils.data.DataLoader(
        dataset, batch_size=batch_size, 
        num_workers=num_workers, pin_memory=pin_memory, shuffle=shuffle
    )

    return data_loader