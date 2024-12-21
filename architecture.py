from torch.nn.utils.rnn import pad_sequence
import os
from PIL import Image
from torch.utils.data import Dataset
import pandas as pd
import torch
import re
import json
from typing import List
from torch import nn
from transformers import DeiTModel, AutoConfig

class ImageToLatexModel(nn.Module):
    def __init__(self, vocab_size, max_seq_length):
        super().__init__()
        # Load DeiT model as the encoder
        config = AutoConfig.from_pretrained("facebook/deit-base-distilled-patch16-384")
        self.encoder = DeiTModel(config)
        
        # Learnable embedding layer for target tokens (to match d_model)
        self.token_embedding = nn.Embedding(vocab_size, config.hidden_size)
        
        # Define a transformer decoder for sequence generation
        self.decoder = nn.Transformer(
            d_model=config.hidden_size,  # Use the hidden size from the encoder
            nhead=8,                     # Number of attention heads
            num_encoder_layers=4,
            num_decoder_layers=4,
            dim_feedforward=2048,
            dropout=0.1
        )
        
        # Linear projection layer to map decoder outputs to token logits
        self.token_projection = nn.Linear(config.hidden_size, vocab_size)
        self.max_seq_length = max_seq_length

    def forward(self, images, input_ids=None):
        # Encode the image using DeiT encoder
        encoder_outputs = self.encoder(pixel_values=images).last_hidden_state  # (batch_size, seq_length, hidden_size)

        # Prepare target embeddings for the decoder
        if input_ids is not None:
            # Shift input_ids for teacher forcing
            tgt_embeddings = self.token_embedding(input_ids)
        else:
            # During inference, generate step-by-step (not covered in this example)
             # If no input_ids are provided, we cannot run (used during inference)
            raise ValueError("input_ids must be provided during training")
        
        # Decode sequence
        decoder_outputs = self.decoder(
            src=encoder_outputs.permute(1, 0, 2),  # Transformer expects (seq_length, batch_size, hidden_size)
            tgt=tgt_embeddings.permute(1, 0, 2)   # Same shape for tgt
        )
        
        # Project decoder outputs to logits
        logits = self.token_projection(decoder_outputs.permute(1,0,2))  # (batch_size, seq_length, vocab_size)
        return logits


class LatexTokenizer:
    def __init__(self, vocab_file):
        # Define regex patterns for tokenization
        self.token_specification = [
            ('COMMAND', r'\\[a-zA-Z]+'),  # LaTeX commands
            ('VARIABLE', r'[a-zA-Z0-9]'), # Variables (letters and numbers)
            ('OPERATOR', r'[+\-*/=]'),    # Operators
            ('BRACE', r'[{}]'),           # Curly braces
            ('PAREN', r'[()]'),           # Parentheses
            ('SUBSUP', r'[_^]'),          # Subscripts and superscripts
            ('COMMA', r','),              # Commas
            ('SEMICOLON', r';'),           # Semicolons
            ('SPACE', r'\s+'),            # Spaces (ignored in token IDs)
            ('OTHER', r'.'),              # Catch-all for any remaining symbols
        ]
        self.token_regex = '|'.join(f'(?P<{pair[0]}>{pair[1]})' for pair in self.token_specification)
        self.vocab_file = vocab_file
        # Define a vocabulary with special tokens
        if self.vocab_file and os.path.exists(self.vocab_file):
            self.vocab = self.load_vocab(self.vocab_file)
            self.inverse_vocab = {v: k for k, v in self.vocab.items()}
        
        # self.vocab = {
        #     '[PAD]': 0, '[UNK]': 1, '[CLS]': 2, '[SEP]': 3, '[MASK]': 4
        # }
        # self.inverse_vocab = {v: k for k, v in self.vocab.items()}
    
    def tokenize(self, formula: str) -> List[str]:
        """Tokenize the input LaTeX formula into a list of tokens."""
        tokens = []
        for match in re.finditer(self.token_regex, formula):
            token_type = match.lastgroup
            token_value = match.group()
            # if token_type != 'SPACE':  # Ignore spaces
            tokens.append(token_value)
        return tokens

    def encode(self, formula: str) -> List[int]:
        """Convert tokens into token IDs based on the vocabulary."""
        tokens = self.tokenize(formula)
        token_ids = []
        for token in tokens:
            if token not in self.vocab:
                # Add token to vocabulary if it's not already present
                new_id = len(self.vocab)
                self.vocab[token] = new_id
                self.inverse_vocab[new_id] = token
                
                if self.vocab_file:
                    self.save_vocab(self.vocab_file)
                    
            token_ids.append(self.vocab[token])
        return [self.vocab['[CLS]']] + token_ids + [self.vocab['[SEP]']]

    def decode(self, token_ids: List[int]) -> str:
        """Convert token IDs back to the original LaTeX formula."""
        tokens = [self.inverse_vocab.get(token_id, '[UNK]') for token_id in token_ids]
        # Remove special tokens for decoding
        tokens = [token for token in tokens if token not in {'[CLS]', '[SEP]', '[PAD]'}]
        decoded_formula = ''.join(tokens)
        
        # Fix any misplaced tokens (like square brackets) around `right`
        decoded_formula = decoded_formula.replace(r'\]right', r'\right')  # Ensure no extraneous square brackets
        
        return decoded_formula
    
    def save_vocab(self, vocab_file: str):
        """Save the vocabulary to a JSON file."""
        with open(vocab_file, 'w') as f:
            json.dump(self.vocab, f)
    
    def load_vocab(self, vocab_file: str) -> dict:
        """Load the vocabulary from a JSON file."""
        with open(vocab_file, 'r') as f:
            vocab = json.load(f)
        return vocab
    
    def vocab_size(self):
       #returns vocab size
        return len(self.vocab)


class ImageLatexDatasetCSV(Dataset):
    def __init__(self, csv_file, image_dir, transform=None, tokenizer=None, max_seq_length=128):
        self.data = pd.read_csv(csv_file)  # Load the CSV file
        self.image_dir = image_dir        # Directory containing the images
        self.transform = transform        # Image preprocessing transforms
        self.tokenizer = tokenizer        # Tokenizer for LaTeX formulas
        self.max_seq_length = max_seq_length

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
        tokens = self.tokenizer.tokenize(formula)
        input_ids = self.tokenizer.encode(formula)
        
        input_ids = input_ids[:self.max_seq_length]
        return img, input_ids, tokens
    
    

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
