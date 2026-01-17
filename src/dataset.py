#importing libraries
import os
from typing import List, Tuple
import numpy as np
import torch
from torch.utils.data import Dataset
from PIL import Image


#creating class
class KvasirSegDataset(Dataset):
    def __init__(self, root_dir: str, files: List[str], transform=None):
        self.root_dir = root_dir
        self.img_dir = os.path.join(root_dir, "images")
        self.msk_dir = os.path.join(root_dir, "masks")
        self.files = files
        self.transform = transform

    
    def __len__(self):
        return len(self.files)
    
    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor]:
        fname = self.files[idx]
        img_path = os.path.join(self.img_dir, fname)
        msk_path = os.path.join(self.msk_dir, fname)
        
        image = np.array(Image.open(img_path).convert("RGB"))
        mask = np.array(Image.open(msk_path).convert("L"))

        #binarize mask: {0,1}
        mask = (mask > 0).astype(np.float32)

        if self.transform is not None:
            aug = self.transform(image=image , mask=mask)
            image = aug["image"]
            mask = aug["mask"]
            

        if isinstance(mask, torch.Tensor):
            mask = mask.unsqueeze(0).float()
        else:
            mask = torch.from_numpy(mask).unsqueeze(0).float()
        
        return image.float(), mask


