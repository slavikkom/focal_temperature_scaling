import medmnist
import torch
import numpy as np
from torchvision.transforms import Compose, ToTensor, Normalize
from PIL import Image

class MedMNISTWrapper:
    def __init__(self, base_dataset):
        self.base_dataset = base_dataset

    def __len__(self):
        return len(self.base_dataset)

    def __getitem__(self, index):
        img, target = self.base_dataset.imgs[index], self.base_dataset.labels[index].astype(int)
        img = Image.fromarray(img)
        if getattr(self.base_dataset, "as_rgb", False):
            img = img.convert("RGB")
        if self.base_dataset.transform is not None:
            img = self.base_dataset.transform(img)
        if self.base_dataset.target_transform is not None:
            target = self.base_dataset.target_transform(target)
        return img, target.squeeze()


def get_medmnist_data_loader(dataset_name, root, batch_size, split='train', shuffle=True, num_workers=4, pin_memory=False, smoke_test=False, as_rgb=False):
    medmnist_class = getattr(medmnist, medmnist.INFO[dataset_name]['python_class'])
    transform = Compose([ToTensor(), Normalize(mean=[0.5], std=[0.5])])
    base_dataset = medmnist_class(root=root, split=split, transform=transform, download=True, as_rgb=as_rgb)
    dataset = MedMNISTWrapper(base_dataset)
    if smoke_test:
        indices = np.arange(100)
        dataset = torch.utils.data.Subset(dataset, indices)
    data_loader = torch.utils.data.DataLoader(dataset, batch_size=batch_size, num_workers=num_workers, pin_memory=pin_memory, shuffle=shuffle)
    return data_loader