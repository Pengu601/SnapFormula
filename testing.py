import pandas as pd
import torch
from torch import nn
from torch.optim import Adam
from torchvision import datasets, transforms, models
from torch.utils.data import DataLoader
import architecture
from transformers import AutoTokenizer
from transformers import DeiTForImageClassification
from torch.optim.lr_scheduler import StepLR
from timm.data.constants import IMAGENET_DEFAULT_MEAN, IMAGENET_DEFAULT_STD
from tqdm import tqdm
from PIL import Image
import re




transform = transforms.Compose([
    transforms.Resize((224, 224)),  # Resize to a standard size
    
    transforms.ToTensor(),         # Convert to PyTorch tensor
    transforms.Normalize(IMAGENET_DEFAULT_MEAN, IMAGENET_DEFAULT_STD)  # Normalize pixel values to [-1, 1]
])

latex_vocab_path = "D:\\GitHub Projects\\SnapFormula\\latex_vocab.json"
tokenizer = architecture.LatexTokenizer(vocab_file = latex_vocab_path)  # Replace with a LaTeX-specific tokenizer if needed

test_dataset = architecture.ImageLatexDatasetCSV(
    csv_file='train_split.csv',
    image_dir='D:\\archive\\formula_images_processed\\formula_images_processed',
    transform=transform,
    tokenizer=tokenizer
)

val_dataset = architecture.ImageLatexDatasetCSV(
    csv_file='val_split.csv',
    image_dir='D:\\archive\\formula_images_processed\\formula_images_processed',
    transform=transform,
    tokenizer=tokenizer
)

train_loader = DataLoader(test_dataset, batch_size=64, shuffle=False, pin_memory=True, collate_fn = architecture.collate_fn)
val_loader = DataLoader(val_dataset, batch_size=64, shuffle=False, pin_memory=True, collate_fn = architecture.collate_fn)



device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

model = ImageToLatexModel(vocab_size, embed_dim, hidden_dim, max_seq_len)
tokenizer = architecture.LatexTokenizer(vocab_file = "D:\\GitHub Projects\\SnapFormula\\latex_vocab.json" )
# Loss & Optimizer
criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=1e-4)

scheduler = StepLR(optimizer, step_size=3)

def train_epoch(model, dataloader, criterion, optimizer, device):
    model.train()
    epoch_loss = 0
    correct = 0
    total = 0
    # count = 1
    for images, input_ids, attention_mask  in tqdm(dataloader):
        images, input_ids,  = images.to(device), input_ids.to(device), 
        input_ids = torch.clamp(input_ids, min=0, max=tokenizer.vocab_size() - 1)
        vocab_size = tokenizer.vocab_size()
        # print(f"Input IDs: {input_ids}")
        # print(f"Min value: {input_ids.min()}, Max value: {input_ids.max()}, Vocab size: {tokenizer.vocab_size}")
        assert input_ids.min() >= 0 and input_ids.max() < vocab_size, "Invalid input IDs detected!"

        
        # Zero the parameter gradients
        optimizer.zero_grad()

        # Forward pass
        outputs = model(pixel_values=images)
        logits = outputs.logits
        if torch.isnan(logits).any(): #makes sure logits aren't nan as it means there is a defect in image
            print("NaN detected in logits!")
            break
        # outputs = outputs.contiguous()
        # Compute the loss
        loss = criterion(logits, input_ids[:, 0])  # Reshape for seq-to-seq loss
        loss.backward()

        # Update parameters
        optimizer.step()

        epoch_loss += loss.item()
        # print(epoch_loss) 
        
        # Calculate training accuracy
        preds = torch.argmax(logits, dim=-1)
        correct += (preds == input_ids[:, 0]).sum().item()
        total += input_ids.size(0)
        
        # count+= 1
    accuracy = correct / total
    return epoch_loss / len(dataloader), accuracy

def validate_epoch(model, dataloader, criterion, device):
    model.eval()
    epoch_loss = 0
    correct = 0
    total = 0
    with torch.no_grad():
        for images, input_ids, attention_mask  in tqdm(dataloader):
            images, input_ids = images.to(device), input_ids.to(device), 
            input_ids = torch.clamp(input_ids, min=0, max=tokenizer.vocab_size() - 1)
            # Forward pass
            outputs = model(pixel_values=images)
            logits = outputs.logits
            # outputs = outputs.contiguous()
            # Compute the loss
            loss = criterion(logits, input_ids[:, 0]) 
            epoch_loss += loss.item()
            
            # Calculate validation accuracy
            preds = torch.argmax(logits, dim=-1)
            correct += (preds == input_ids[:, 0]).sum().item()
            total += input_ids.size(0)
    
    accuracy = correct / total
    
    return epoch_loss / len(dataloader), accuracy


num_epochs = 2
best_val_loss = float('inf')  # Initialize to a large value
# token_to_inx = tokenizer.vocab
# idx_to_token = {v: k for k, v in token_to_inx.items()}
# for token_id, latex_token in idx_to_token.items():
#     print(f"Token ID {token_id}: {latex_token}")

for epoch in range(num_epochs):
    print(f"Epoch {epoch+1}/{num_epochs}")
    
    # Train for one epoch
    train_loss, train_acc = train_epoch(model, train_loader, criterion, optimizer, device)
    print(f"Train Loss: {train_loss:.4f}, Train Accuracy: {train_acc:.4f}")
    

    # Validate
    val_loss, val_acc = validate_epoch(model, val_loader, criterion, device)
    print(f"Validation Loss: {val_loss:.4f}, Validation Accuracy: {val_acc:.4f}")

    # Save the best model
    if val_loss < best_val_loss:
        best_val_loss = val_loss
        torch.save(model.state_dict(), model_save_path)
        print(f"Best model saved with validation loss: {val_loss:.4f}")

    # Adjust learning rate
    scheduler.step(val_loss)