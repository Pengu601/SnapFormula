import torch
import torch.nn as nn
import torch.nn.functional as F
import os
from PIL import Image
from torchvision import transforms
from torch.utils.data import Dataset, DataLoader
import pandas as pd

class ImageToLatexModel(nn.Module):
    def __init__(self, vocab_size, embed_dim=256, enc_channels=64, dec_hidden_size=512, max_seq_length=128):
        super(ImageToLatexModel, self).__init__()
        self.vocab_size = vocab_size
        self.max_seq_length = max_seq_length

        # CNN-based Encoder
        self.encoder = nn.Sequential(
            nn.Conv2d(3, enc_channels, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2),
            nn.Conv2d(enc_channels, enc_channels * 2, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2),
        )
        self.enc_fc = nn.Linear(enc_channels * 2 * (128 // 4) * (128 // 4), embed_dim)

        # Decoder: Transformer
        self.embedding = nn.Embedding(vocab_size, embed_dim)
        self.transformer_decoder = nn.Transformer(
            d_model=embed_dim,
            nhead=8,
            num_encoder_layers=6,
            num_decoder_layers=6,
            dim_feedforward=2048,
        )
        self.fc_out = nn.Linear(embed_dim, vocab_size)

    def forward(self, images, input_ids, attention_mask):
        # Encode image (extract image features)
        batch_size = images.size(0)
        enc_features = self.encoder(images)  # [batch_size, channels, height, width]
        enc_features = enc_features.view(batch_size, -1)  # Flatten to [batch_size, features]
        enc_features = self.enc_fc(enc_features)  # [batch_size, embed_dim]
        enc_features = enc_features.unsqueeze(0)  # [1, batch_size, embed_dim] for transformer

        # Prepare input for the decoder (embedding the LaTeX formula)
        embeddings = self.embedding(input_ids)  # [batch_size, seq_len, embed_dim]
        embeddings = embeddings.transpose(0, 1)  # [seq_len, batch_size, embed_dim] for transformer

        # Pass to transformer: `memory` is encoder output and `tgt` is the LaTeX formula
        dec_output = self.transformer_decoder(
            tgt=embeddings,        # [seq_len, batch_size, embed_dim]
            memory=enc_features,   # [1, batch_size, embed_dim]
            src_key_padding_mask=attention_mask  # Mask to ignore padding tokens
        )

        # Final output layer to predict LaTeX tokens
        dec_output = dec_output.transpose(0, 1)  # [batch_size, seq_len, embed_dim]
        logits = self.fc_out(dec_output)  # [batch_size, seq_len, vocab_size]
        return logits

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
        tokens = self.tokenizer(
            formula,
            padding="max_length",
            truncation=True,
            max_length=self.max_seq_length,
            return_tensors="pt"
        )
        return img, tokens["input_ids"].squeeze(), tokens["attention_mask"].squeeze()