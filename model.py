import pandas as pd
import torch
from torch.nn.functional import cross_entropy
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
from collections import Counter
import math
from transformers import get_linear_schedule_with_warmup
#load spreadsheet

model_save_path = 'E:\\ai stuf\\best_model.pth'
latex_vocab_path = "D:\\GitHub Projects\\SnapFormula\\latex_vocab.json"
tokenizer = architecture.LatexTokenizer(vocab_file = latex_vocab_path)  

transform = transforms.Compose([
    transforms.RandomRotation(10),  # Rotate ±10 degrees
    transforms.RandomResizedCrop(224, scale=(0.8, 1.0)),
    transforms.RandomHorizontalFlip(),
    transforms.ToTensor(),
    transforms.Normalize(IMAGENET_DEFAULT_MEAN, IMAGENET_DEFAULT_STD)
])

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

# print(tokenizer.vocab_size())
# for token, token_id in tokenizer.vocab.items():
#     print(f"Token: {token}, ID: {token_id}")

train_loader = DataLoader(test_dataset, batch_size=64, shuffle=False, pin_memory=True, collate_fn = architecture.collate_fn)
val_loader = DataLoader(val_dataset, batch_size=64, shuffle=False, pin_memory=True, collate_fn = architecture.collate_fn ) 
# Example training loop
  # Define your model

cuda_ = "cuda:0"
# print(torch.cuda.is_available())  # Should return True if GPU is detected
# print(torch.cuda.device_count())  # Number of GPUs detected
# print(torch.cuda.get_device_name(0))  # Name of the first GPU
token_counts = Counter()
for _, token_ids, _ in train_loader:
    token_ids_flat = token_ids.view(-1).tolist()
    token_counts.update(token_ids_flat)
print(token_counts.most_common(10))  # Print 10 most common tokens


device = torch.device(cuda_ if torch.cuda.is_available() else "cpu")
model = architecture.ImageToLatexModel(vocab_size = tokenizer.vocab_size(), max_seq_length= 128).to(device)
# model.load_state_dict(torch.load(model_save_path))
optimizer = Adam(model.parameters(), lr=0.0001,  weight_decay=1e-4)

token_weights = torch.ones(tokenizer.vocab_size()).to(device)

k = 2  # Smoothing factor

for token, count in token_counts.items():
    token_weights[token] = 1 / (math.log(count + k) + 1)

# Normalize weights to avoid large disparities
token_weights = token_weights / token_weights.mean()
    

criterion = nn.CrossEntropyLoss(weight=token_weights, ignore_index = 0)
# model.classifier == nn.Linear(model.classifier.in_features, tokenizer.vocab_size())

model.load_state_dict(torch.load(model_save_path))

def train_epoch(model, dataloader, criterion, optimizer, device):
    model.train()
    epoch_loss = 0
    total_tokens = 0
    correct_tokens = 0
    correct_tokens_debug = 0
    total_tokens_debug = 0
    # count = 1
    for images, input_ids, _  in tqdm(dataloader):
        images, input_ids = images.to(device), input_ids.to(device), 
        
        input_ids_shifted = input_ids[:, :-1]
        labels = input_ids[:, 1:]
        # print(f"Input IDs: {input_ids}")
        # print(f"Min value: {input_ids.min()}, Max value: {input_ids.max()}, Vocab size: {tokenizer.vocab_size}")
        # assert input_ids.min() >= 0 and input_ids.max() < vocab_size, "Invalid input IDs detected!"

        # Forward pass
        
        optimizer.zero_grad() # Zero the parameter gradients
        logits = model(images, input_ids = input_ids_shifted)
        
        
        # if torch.isnan(logits).any(): #makes sure logits aren't nan as it means there is a defect in image
        #     print("NaN detected in logits!")
        #     break
        # Reshape logits and labels for loss calculation
        logits = logits.view(-1, logits.size(-1))  # Flatten (batch_size * seq_length, vocab_size)
        labels = labels.reshape(-1)                         # Flatten (batch_size * seq_length)
        # outputs = outputs.contiguous()
        
        # Compute the loss
        loss = criterion(logits, labels)  # Reshape for seq-to-seq loss
        loss.backward() #backwards propogation
        optimizer.step()

        epoch_loss += loss.item()
        # print(epoch_loss) 
        
        # Calculate training accuracy
        # Calculate token-level accuracy
        preds = torch.argmax(logits, dim=-1)  # Shape: (batch_size * seq_length)
        
        preds_debug = preds.view(images.size(0), -1)  # Reshape predictions
        labels_debug = labels.view(images.size(0), -1)  # Reshape labels
        
        # print(preds_debug)
        # print(labels_debug)
       
        # Decode ground truth and predicted tokens
        for i in range(min(1, labels.size(0))):
                 
            gt_tokens = labels_debug[i].tolist()
            pred_tokens = preds_debug[i].tolist()
            gt_latex = tokenizer.decode(labels_debug[i].tolist())
            pred_latex = tokenizer.decode(preds_debug[i].tolist())
            
            non_pad_mask_debug = (labels_debug[i] != tokenizer.vocab['[CLS]']) & (labels_debug[i] != tokenizer.vocab['[SEP]']) & (labels_debug[i] != tokenizer.vocab['[PAD]'])  # Create a mask for valid tokens (excluding [CLS] and [SEP])
            correct_tokens_debug = ((preds_debug[i] == labels_debug[i]) & non_pad_mask_debug).sum().item()
            total_tokens_debug = non_pad_mask_debug.sum().item()
            acc = correct_tokens_debug/total_tokens_debug
            # Print comparison
            # print(f"  Ground Truth (Token): {gt_tokens}")
            # print(f"  Predicted (Token): {pred_tokens}")
            print(f"  Ground Truth (LaTeX): {gt_latex}")
            print(f"  Predicted (LaTeX): {pred_latex}")
            print(f"  Accuracy: {acc}")

       
        non_pad_mask = (labels != tokenizer.vocab['[CLS]']) & (labels != tokenizer.vocab['[SEP]']) & (labels != tokenizer.vocab['[PAD]'])  # Create a mask for valid tokens (excluding [CLS] and [SEP])
        correct_tokens += (preds == labels)[non_pad_mask].sum().item()
        total_tokens += non_pad_mask.sum().item()
        
        # count+= 1
    accuracy = correct_tokens / total_tokens if total_tokens > 0 else 0
    return epoch_loss / len(dataloader), accuracy

def validate_epoch(model, dataloader, criterion, device):
    model.eval()
    epoch_loss = 0
    total_tokens = 0
    correct_tokens = 0
    with torch.no_grad():
        for images, input_ids, _  in tqdm(dataloader):
            images, input_ids = images.to(device), input_ids.to(device), 
            
            input_ids_shifted = input_ids[:, :-1]
            labels = input_ids[:, 1:]
        
            # Forward pass
            logits = model(images, input_ids = input_ids_shifted)
            
            logits = logits.view(-1, logits.size(-1))  # Flatten (batch_size * seq_length, vocab_size)
            labels = labels.reshape(-1)                         # Flatten (batch_size * seq_length)
            # outputs = outputs.contiguous()
            
            # Compute the loss
            loss = criterion(logits, labels) 
            epoch_loss += loss.item()
            
            # Calculate validation accuracy
            preds = torch.argmax(logits, dim=-1)  # Shape: (batch_size * seq_length)
            
            preds_debug = preds.view(images.size(0), -1)  # Reshape predictions
            labels_debug = labels.view(images.size(0), -1)  # Reshape labels
            
            print(preds_debug)
            print(labels_debug)
            for i in range(min(1, labels.size(0))):  # Inspect up to 5 examples
                # Decode ground truth and predicted tokens
                formula = tokenizer.decode(input_ids[i].tolist())
                gt_tokens = labels_debug[i].tolist()
                pred_tokens = preds_debug[i].tolist()
                gt_latex = tokenizer.decode(labels_debug[i].tolist())
                pred_latex = tokenizer.decode(preds_debug[i].tolist())
                # Remove special tokens ([CLS], [SEP])
                # gt_tokens = gt_tokens.strip()
                # pred_tokens = pred_tokens.strip()

                # Print comparison
                # print(f"Example {i + 1}:")
                # print(f"Formula: {formula}")
                # print(f"  Ground Truth (Token): {gt_tokens}")
                # print(f"  Predicted (Token): {pred_tokens}")
                print(f"  Ground Truth (LaTeX): {gt_latex}")
                print(f"  Predicted (LaTeX): {pred_latex}")
                
                
            non_pad_mask = (labels != tokenizer.vocab['[CLS]']) & (labels != tokenizer.vocab['[SEP]'])   # Create a mask for valid tokens (excluding [CLS] and [SEP])
            correct_tokens += (preds == labels)[non_pad_mask].sum().item()
            total_tokens += non_pad_mask.sum().item()
    
    accuracy = correct_tokens / total_tokens if total_tokens > 0 else 0
    return epoch_loss / len(dataloader), accuracy


num_epochs = 20
best_val_acc = .45 # Initialize to a small value
num_warmup_steps = len(train_loader) * 2  # Warm up over 2 epochs
total_steps = len(train_loader) * num_epochs
scheduler = StepLR(optimizer, step_size=3, gamma=0.5)

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
    if val_acc > best_val_acc:
        best_val_loss = val_loss
        torch.save(model.state_dict(), model_save_path)
        print(f"Best model saved with validation accuracy: {val_acc:.4f}")

    # Adjust learning rate
    scheduler.step(val_loss)