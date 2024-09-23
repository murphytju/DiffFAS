import torch
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from PIL import Image
import os
import random
import torchvision
import json

class OCIMDataset(Dataset):
    def __init__(self, json_file, root_dir, transform=None):
        self.image_pairs = self.read_json_file(json_file)
        self.root_dir = root_dir
        self.transform = transform

    def read_json_file(self, json_file):
        with open(json_file, 'r') as f:
            data = json.load(f)
        
        image_pairs = []
        for pair in data:
            content_path = pair[0].lstrip('/')
            gt_path = pair[1].lstrip('/')
            image_pairs.append((content_path, gt_path))
        return image_pairs

    def __len__(self):
        return len(self.image_pairs)

    def __getitem__(self, idx):
        img_paths = self.image_pairs[idx]
        content_path, gt_path = img_paths

        content_img = Image.open(os.path.join(self.root_dir, content_path)).convert('RGB')
        gt_img = Image.open(os.path.join(self.root_dir, gt_path)).convert('RGB')

        style_folder = os.path.dirname(gt_path)
        style_dir = self.root_dir
        style_file = random.choice(os.listdir(os.path.join(style_dir, style_folder)))
        style_path = os.path.join(style_folder, style_file)
        style_img = Image.open(os.path.join(self.root_dir, style_path)).convert('RGB')

        if self.transform:
            content_img = self.transform(content_img)
            style_img = self.transform(style_img)
            gt_img = self.transform(gt_img)

        return {'content': content_img, 'style_spoof': style_img, 'GT': gt_img}
 


class WMCADataset(Dataset):
    def __init__(self, json_file, root_dir, transform=None):
        self.image_pairs = self.read_json_file(json_file)
        self.root_dir = root_dir
        self.transform = transform

    def read_json_file(self, json_file):
        with open(json_file, 'r') as f:
            data = json.load(f)

        image_pairs = []
        for pair in data:
            content_path = pair[0].lstrip('/')
            gt_path = pair[1].lstrip('/')
            image_pairs.append((content_path, gt_path))
        return image_pairs

    def __len__(self):
        return len(self.image_pairs)

    def __getitem__(self, idx):
        img_paths = self.image_pairs[idx]
        content_path, gt_path = img_paths
        content_img = Image.open(os.path.join(self.root_dir, content_path)).convert('RGB')
        gt_img = Image.open(os.path.join(self.root_dir, gt_path)).convert('RGB')

        
        style_folder = os.path.dirname(gt_path)
        style_dir = self.root_dir
        style_folder_name = os.path.basename(style_folder)

        style_file = random.choice(os.listdir(os.path.join(style_dir, style_folder_name)))
        style_path = os.path.join(style_dir, style_folder_name, style_file)
        style_img = Image.open(style_path).convert('RGB')

        if self.transform:
            content_img = self.transform(content_img)
            style_img = self.transform(style_img)
            gt_img = self.transform(gt_img)

        return {'content': content_img, 'style_spoof': style_img, 'GT': gt_img}



class PADISIDataset(Dataset):
    def __init__(self, json_file, root_dir, transform=None):
        self.image_pairs = self.read_json_file(json_file,root_dir)
        self.root_dir = root_dir
        self.transform = transform

    def read_json_file(self, json_file,root_dir):
        with open(json_file, 'r') as f:
            data = json.load(f)
        image_pairs = [(os.path.join(root_dir, pair[0].lstrip('/')), os.path.join(root_dir, pair[1].lstrip('/'))) for pair in data]  
        return image_pairs

    def __len__(self):
        return len(self.image_pairs)

    def __getitem__(self, idx):
        img_paths = self.image_pairs[idx]
        content_path, gt_path = img_paths
        content_img = Image.open(content_path).convert('RGB')
        gt_img = Image.open(gt_path).convert('RGB')

        style_folder_name = self.get_style_folder_name(gt_path)
        style_file = random.choice(os.listdir(os.path.join(self.root_dir, style_folder_name)))
        style_path = os.path.join(self.root_dir, style_folder_name, style_file)
        style_img = Image.open(style_path).convert('RGB')

        if self.transform:
            content_img = self.transform(content_img)
            style_img = self.transform(style_img)
            gt_img = self.transform(gt_img)

        return {'content': content_img, 'style_spoof': style_img, 'GT': gt_img}

    def get_style_folder_name(self, gt_path):
        basename = os.path.basename(os.path.dirname(gt_path))
        
        if "glass" in basename:
            return "glass"
        elif "print" in basename:
            return "print"
        elif "paper_glass" in basename:
            return "paper_glass"
        elif "3d_face" in basename:
            return "3d_face"
        elif "eyeball_1" in basename:
            return "eyeball_1"
        elif "eyeball_2" in basename:
            return "eyeball_2"
        else:
            mask_match = [f"mask_{i}" for i in range(1, 10)]
            for mask in mask_match:
                if mask in basename:
                    return mask
            if "makeup" in basename:
                return "makeup"
        return "unknown"

