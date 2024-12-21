from torch.nn.utils.rnn import pad_sequence
import os
from PIL import Image
from torch.utils.data import Dataset
import pandas as pd
import torch
import re
import json
from typing import List
from typing import List, Dict
from tokenizers import Tokenizer, models, pre_tokenizers, normalizers, processors
from tokenizers.pre_tokenizers import PreTokenizer




class ImageLatexDatasetCSV(Dataset):
    def __init__(self, csv_file, image_dir, processor, transform=None, tokenizer=None, max_seq_length=150):
        self.data = pd.read_csv(csv_file)  # Load the CSV file
        self.image_dir = image_dir        # Directory containing the images
        self.transform = transform        # Image preprocessing transforms
        self.tokenizer = tokenizer        # Tokenizer for LaTeX formulas
        self.max_seq_length = max_seq_length
        self.processor = processor

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        # Get image name and LaTeX formula
        image_name = self.data.iloc[idx]['image']
        formula = self.data.iloc[idx]['formula']

        # Load and preprocess the image
        img_path = os.path.join(self.image_dir, image_name)
        img = Image.open(img_path).convert("RGB")
        if self.transform:
            img = self.transform(img)

        # Tokenize the LaTeX formula
        pixel_values = self.processor(img, return_tensors = "pt").pixel_values
        
        
        labels = self.tokenizer.encode(formula).ids
        labels = [label if label != self.tokenizer.token_to_id("[PAD]") else -100 for label in labels]
        encoding = {"pixel_values" : pixel_values.squeeze(), "labels": torch.tensor(labels)}
        return encoding
    
    

def collate_fn(batch):
    """
    Custom collate function for batching image-latex pairs.
    Pads the LaTeX token sequences and batches the images together.
    
    Args:
        batch (list): List of tuples (image, input_ids, tokens)
    """
    # Separate the images, input_ids (tokenized formulas), and tokens (raw token strings)
    images = [item[0] for item in batch]  # Images
    input_ids = [torch.tensor(item[1]) for item in batch]  # Tokenized LaTeX formula (input_ids)
    tokens = [item[2] for item in batch]  # Raw tokens (optional, for labels or analysis)
    
    # Pad the input_ids to the maximum length in the batch
    input_ids_padded = pad_sequence(input_ids, batch_first=True, padding_value=0)  # Pad with 0 for [PAD] token
    
    # Stack the images into a single tensor
    images = torch.stack(images, dim=0)  # Shape: (batch_size, channels, height, width)

    # Return the batch
    return images, input_ids_padded, tokens
