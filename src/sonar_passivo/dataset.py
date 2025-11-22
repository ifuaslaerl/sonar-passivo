import os
import torch
import numpy as np
import scipy.io
from torch.utils.data import Dataset
from typing import List, Tuple, Optional
from random import shuffle, seed

class SonarDataset(Dataset):
    """
    Dataset class for loading Sonar .mat files.
    """

    def __init__(self, root_path: str, transform=None):
        self.root_path = root_path
        self.transform = transform
        self.samples: List[Tuple[str, int]] = [] #(file_path, row_index)
        self.labels: List[int] = []
        
        # Check if path exists
        if not os.path.exists(root_path):
            raise FileNotFoundError(f"Dataset path not found: {root_path}")

        self.classes = sorted([d for d in os.listdir(root_path) if os.path.isdir(os.path.join(root_path, d))])
        self._load_data()

    def _load_data(self):
        for class_idx, class_name in enumerate(self.classes):
            class_dir = os.path.join(self.root_path, class_name)
            
            for filename in os.listdir(class_dir):
                if not filename.endswith('.mat'):
                    continue
                    
                file_path = os.path.join(class_dir, filename)
                try:
                    mat_data = scipy.io.loadmat(file_path)
                    # Assuming 'ent_norm' is the key, as per original code
                    if "ent_norm" not in mat_data:
                        continue
                        
                    matrix = mat_data["ent_norm"]
                    # Store reference to file and index to avoid loading everything into RAM at once
                    # if the dataset is huge. If small, we could cache it.
                    for i in range(len(matrix)):
                        self.samples.append((file_path, i))
                        self.labels.append(class_idx)
                except Exception as e:
                    print(f"Error loading {file_path}: {e}")

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, index: int) -> Tuple[torch.Tensor, torch.Tensor]:
        file_path, line_idx = self.samples[index]
        label = self.labels[index]

        # Load on demand
        mat_data = scipy.io.loadmat(file_path)
        signal = mat_data["ent_norm"][line_idx]

        # Convert to tensor and ensure correct shape (Channels, Length)
        # Original code used unsqueeze(0) for Conv1d channel dim
        data_tensor = torch.tensor(signal, dtype=torch.float32).unsqueeze(0)
        label_tensor = torch.tensor(label, dtype=torch.long)

        if self.transform:
            data_tensor = self.transform(data_tensor)

        return data_tensor, label_tensor

    def get_subset(self, percentage: float, random_seed: Optional[int] = None) -> List[Tuple]:
        """Returns a random subset of the data."""
        if random_seed is not None:
            seed(random_seed)
        
        combined = list(zip(self.samples, self.labels))
        shuffle(combined)
        
        cutoff = int(len(combined) * percentage)
        return combined[:cutoff]

    def merge_with(self, other_dataset: 'SonarDataset', ratio: float):
        """
        Merges this dataset with another one in place.
        ratio: Percentage of THIS dataset to keep.
        """
        # Logic adapted from original mix_datasets.py
        # Note: This simple implementation merges lists. 
        # For production, consider ConcatDataset.
        
        subset_self = self.get_subset(ratio)
        subset_other = other_dataset.get_subset(1.0 - ratio)
        
        # Reconstruct internal lists
        combined = subset_self + subset_other
        self.samples = [x[0] for x in combined]
        self.labels = [x[1] for x in combined]
        print(f"Merged datasets. New size: {len(self.samples)}")

    def save_to_disk(self, output_dir: str):
        """Saves the current state of the dataset to disk organized by class."""
        os.makedirs(output_dir, exist_ok=True)
        
        # Group by class
        data_by_class = {i: [] for i in range(len(self.classes))}
        
        print("Saving dataset to disk...")
        for idx in range(len(self)):
            # Load actual data
            fpath, row = self.samples[idx]
            label = self.labels[idx]
            
            mat = scipy.io.loadmat(fpath)["ent_norm"][row]
            data_by_class[label].append(mat)

        for label_idx, data_list in data_by_class.items():
            if not data_list:
                continue
                
            class_name = self.classes[label_idx]
            class_dir = os.path.join(output_dir, class_name)
            os.makedirs(class_dir, exist_ok=True)
            
            combined_data = np.vstack(data_list)
            out_file = os.path.join(class_dir, f"{class_name}.mat")
            
            scipy.io.savemat(out_file, {"ent_norm": combined_data})
            print(f"Saved {out_file}")
