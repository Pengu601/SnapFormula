import pandas as pd
import torch
from torch import nn
from torch.optim import Adam
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
import architecture
from transformers import AutoTokenizer
from transformers import DeiTForImageClassification
from torch.optim.lr_scheduler import StepLR
from timm.data.constants import IMAGENET_DEFAULT_MEAN, IMAGENET_DEFAULT_STD
from tqdm import tqdm
from PIL import Image
import re

# Function to test the model with a single image
def preprocess_image(image_path, transform):
    """
    Preprocess the input image using the same transforms as during training.
    """
    image = Image.open(image_path).convert("RGB")
    image = transform(image)  # Apply the same transformations
    return image.unsqueeze(0)  # Add a batch dimension


def predict_latex(model, image_path, tokenizer, device, transform, max_seq_length):
    """
    Predict the LaTeX formula for a given image.
    """
    # Step 1: Preprocess the image
    image = preprocess_image(image_path, transform).to(device)
    
    # Step 2: Encode the image
    with torch.no_grad():
        encoder_outputs = model.encoder(pixel_values=image).last_hidden_state  # Encoder output
    
    # Step 3: Initialize decoding
    current_token = torch.tensor([[tokenizer.vocab['[CLS]']]]).to(device)  # Start with the [CLS] token
    generated_tokens = []

    for _ in range(max_seq_length):
        # Generate token embeddings for the current sequence
        tgt_embeddings = model.token_embedding(current_token)  # Convert tokens to embeddings
        
        # Decode using the Transformer decoder
        decoder_outputs = model.decoder(
            src=encoder_outputs.permute(1, 0, 2),  # (seq_length, batch_size, hidden_size)
            tgt=tgt_embeddings.permute(1, 0, 2)   # (seq_length, batch_size, hidden_size)
        )
        
        # Project decoder output to logits
        logits = model.token_projection(decoder_outputs.permute(1, 0, 2))  # (batch_size, seq_length, vocab_size)
        next_token_logits = logits[:, -1, :]  # Get logits for the last predicted token
        next_token = torch.argmax(next_token_logits, dim=-1).unsqueeze(0)  # Get the predicted token ID
        
        # Append the predicted token to the sequence
        generated_tokens.append(next_token.item())
        
        # Stop decoding if the [SEP] token is predicted
        if next_token.item() == tokenizer.vocab['[SEP]']:
            break
        
        # Update the current token sequence
        current_token = torch.cat([current_token, next_token], dim=1)
    
    # Step 4: Decode the token IDs into a LaTeX formula
    decoded_formula = tokenizer.decode(generated_tokens)
    return decoded_formula

# sample_formula = "\\mathcal { A } _ { S G } = \int d ^ { 2 } x \\left( \\frac { ( \\partial _ { \mu } \\phi ) ^ { 2 } } { 1 6 \\pi } + \\lambda \\sqrt { 2 } \\operatorname { c o s } ( \\gamma \\phi ) . \\right)"  # Replace with your LaTeX formula
# tokenizer = architecture.LatexTokenizer(vocab_file = "D:\\GitHub Projects\\SnapFormula\\latex_vocab.json" )
# tokens = tokenizer.tokenize(sample_formula)
# token_ids = tokenizer.encode(sample_formula)
# decoded_formula = tokenizer.decode(token_ids)

# print("Tokens:", tokens)
# print("Token IDs:", token_ids)
# print("Decoded Formula:", decoded_formula)

# for token, token_id in tokenizer.vocab.items():
#     print(f"Token: {token}, ID: {token_id}")

# Load the saved model weights
model_save_path = 'D:\\GitHub Projects\\SnapFormula\\best_model.pth'
latex_vocab_path = "D:\\GitHub Projects\\SnapFormula\\latex_vocab.json"
test_image_path = "C:\\Users\\Nick\\Downloads\\test2 (2).png"  # Replace with your image path
tokenizer = architecture.LatexTokenizer(vocab_file = latex_vocab_path)  

cuda_ = "cuda:0"
device = torch.device(cuda_ if torch.cuda.is_available() else "cpu")

model = architecture.ImageToLatexModel(vocab_size = tokenizer.vocab_size(), max_seq_length= 128).to(device)
model.load_state_dict(torch.load(model_save_path))
model.eval()  # Set model to evaluation mode

# Define the image transformation pipeline
transform = transforms.Compose([
    transforms.Resize((224, 224)),  # Resize to the expected input size
    transforms.ToTensor(),         # Convert to PyTorch tensor
    transforms.Normalize(IMAGENET_DEFAULT_MEAN, IMAGENET_DEFAULT_STD)  # Normalize pixel values to [-1, 1]
])



# Test the model with a sample image

predicted_formula = predict_latex(model, test_image_path, tokenizer, device, transform, 128)
print(f"Predicted LaTeX Formula: {predicted_formula}")