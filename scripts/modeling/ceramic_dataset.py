import torch
from torch.utils.data import Dataset, random_split
from pathlib import Path
from PIL import Image
import json


class CeramicArtifactDataset(Dataset):
    """Dataset for ceramic artifacts with descriptions"""
    
    def __init__(self, image_dir, descriptions_file, image_size=512):
        self.image_dir = Path(image_dir)
        self.image_size = image_size
        self.descriptions = {}
        
        if Path(descriptions_file).exists():
            with open(descriptions_file, 'r', encoding='utf-8') as f:
                for line in f:
                    if line.strip():
                        try:
                            data = json.loads(line)
                            self.descriptions[data['filename']] = data['description']
                        except json.JSONDecodeError:
                            continue
        
        self.image_paths = sorted(self.image_dir.glob("*.png"))
        print(f"Loaded {len(self.image_paths)} images")
        print(f"Loaded {len(self.descriptions)} descriptions")
    
    def __len__(self):
        return len(self.image_paths)
    
    def __getitem__(self, idx):
        image_path = self.image_paths[idx]
        
        try:
            image = Image.open(image_path).convert("RGB")
            image = image.resize((self.image_size, self.image_size), Image.Resampling.LANCZOS)
            
            image_array = torch.from_numpy(
                __import__('numpy').array(image)
            ).permute(2, 0, 1).float()
            image_array = image_array / 127.5 - 1
            
            description = self.descriptions.get(
                image_path.name,
                "ceramic artifact with decorative patterns"
            )
            
            return {
                "image": image_array,
                "text": description,
                "filename": image_path.name
            }
        except Exception as e:
            print(f"Error loading {image_path}: {e}")
            return {
                "image": torch.randn(3, self.image_size, self.image_size),
                "text": "ceramic artifact",
                "filename": image_path.name
            }


def create_train_test_splits(image_dir, descriptions_file, image_size=512, train_ratio=0.8, seed=42):
    """Create train and test dataset splits"""
    
    full_dataset = CeramicArtifactDataset(image_dir, descriptions_file, image_size)
    
    train_size = int(train_ratio * len(full_dataset))
    test_size = len(full_dataset) - train_size
    
    generator = torch.Generator().manual_seed(seed)
    train_dataset, test_dataset = random_split(
        full_dataset,
        [train_size, test_size],
        generator=generator
    )
    
    print(f"\nTrain samples: {len(train_dataset)}")
    print(f"Test samples: {len(test_dataset)}")
    
    return train_dataset, test_dataset
