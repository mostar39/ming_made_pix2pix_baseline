import torch
import torch.nn as nn
from PIL import Image
import numpy as np
from torch.utils.data import Dataset
from torch.utils.data import DataLoader
import torchvision.transforms as transforms
from torchvision.utils import save_image

class ImageDataset(Dataset):
    def __init__(self, before,after, transforms_=None, mode="train"): # before, after ==> root
        self.transform = transforms_
        self.before = before
        self.after = after


    def __getitem__(self, index):
        before_image = Image.open(self.before[index % len(self.before)])
        after_image = Image.open(self.after[index % len(self.after)])

        before_ya = self.transform(before_image)
        after_ya = self.transform(after_image)


        return {"before":before_ya, "after":after_ya}

    def __len__(self):
        return len(self.before)