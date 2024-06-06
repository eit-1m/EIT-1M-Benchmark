from torch.utils.data import Dataset
import numpy as np
import os

class NPYDataset(Dataset):
    def __init__(self, root_dirs, transform, modality='image'):
        self.samples = []
        self.labels = []

        label_to_index = {}
        current_label_id = 0

        # Populate samples and labels
        for root_dir in root_dirs:
            for label in sorted(os.listdir(root_dir)):
                class_path = os.path.join(root_dir, label)
                if os.path.isdir(class_path):
                    if label not in label_to_index:
                        label_to_index[label] = current_label_id
                        current_label_id += 1
                    for filename in os.listdir(class_path):
                        file_path = os.path.join(class_path, filename)
                        if 'image' in modality and file_path.endswith('101.npy'):
                            self.samples.append(file_path)
                            self.labels.append(label_to_index[label])
                        if 'text' in modality and file_path.endswith('13.npy'):
                            self.samples.append(file_path)
                            self.labels.append(label_to_index[label])

        self.transform = transform
        
    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        data = np.load(self.samples[idx])
        data = self.transform(eeg=data)['eeg']
        label = self.labels[idx]
        return data, label
