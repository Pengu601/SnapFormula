import pandas as pd
import torch
from torch import nn
from torch.optim import Adam
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
import architecture
from transformers import AutoTokenizer
#load spreadsheet

dataset = pd.read_csv('im2latex_train.csv')
image_latex_pairs = [(f'C:\\Users\\Nick\\Downloads\\archive\\formula_images_processed\\formula_images_processed\\{row["image"]}', row["formula"]) for _,row in dataset.iterrows()]
model_save_path = 'best_model.pth'


transform = transforms.Compose([
    transforms.Resize((128, 128)),  # Resize to a standard size
    transforms.ToTensor(),         # Convert to PyTorch tensor
    transforms.Normalize((0.5,), (0.5,))  # Normalize pixel values to [-1, 1]
])

tokenizer = AutoTokenizer.from_pretrained("bert-base-uncased")  # Replace with a LaTeX-specific tokenizer if needed

test_dataset = architecture.ImageLatexDatasetCSV(
    csv_file='im2latex_train.csv',
    image_dir='C:\\Users\\Nick\\Downloads\\archive\\formula_images_processed\\formula_images_processed\\',
    transform=transform,
    tokenizer=tokenizer
)

val_dataset = architecture.ImageLatexDatasetCSV(
    csv_file='im2latex_validate.csv',
    image_dir='C:\\Users\\Nick\\Downloads\\archive\\formula_images_processed\\formula_images_processed\\',
    transform=transform,
    tokenizer=tokenizer
)

train_loader = DataLoader(test_dataset, batch_size=16, shuffle=False)
val_loader = DataLoader(val_dataset, batch_size=16, shuffle=False)
# Example training loop
  # Define your model


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = architecture.ImageToLatexModel(vocab_size=tokenizer.vocab_size).to(device)
optimizer = Adam(model.parameters(), lr=0.001)
criterion = nn.CrossEntropyLoss(ignore_index=tokenizer.pad_token_id)

num_epochs = 10

def train_epoch(model, dataloader, criterion, optimizer, device):
    model.train()
    epoch_loss = 0
    for images, input_ids, attention_mask in dataloader:
        images, input_ids, attention_mask = images.to(device), input_ids.to(device), attention_mask.to(device)

        # Zero the parameter gradients
        optimizer.zero_grad()

        # Forward pass
        outputs = model(images, input_ids, attention_mask)

        # Compute the loss
        loss = criterion(outputs.view(-1, outputs.size(-1)), input_ids.view(-1))  # Reshape for seq-to-seq loss
        loss.backward()

        # Update parameters
        optimizer.step()

        epoch_loss += loss.item()
    
    return epoch_loss / len(dataloader)

def validate_epoch(model, dataloader, criterion, device):
    model.eval()
    epoch_loss = 0
    with torch.no_grad():
        for images, input_ids, attention_mask in dataloader:
            images, input_ids, attention_mask = images.to(device), input_ids.to(device), attention_mask.to(device)

            # Forward pass
            outputs = model(images, input_ids, attention_mask)

            # Compute the loss
            loss = criterion(outputs.view(-1, outputs.size(-1)), input_ids.view(-1))
            epoch_loss += loss.item()
    
    return epoch_loss / len(dataloader)



num_epochs = 5
best_val_loss = float('inf')  # Initialize to a large value
for epoch in range(num_epochs):
    print(f"Epoch {epoch+1}/{num_epochs}")
    
    # Train for one epoch
    train_loss = train_epoch(model, train_loader, criterion, optimizer, device)
    print(f"Train Loss: {train_loss:.4f}")

    # Validate
    val_loss = validate_epoch(model, val_loader, criterion, device)
    print(f"Validation Loss: {val_loss:.4f}")

    # Save the best model
    if val_loss < best_val_loss:
        best_val_loss = val_loss
        torch.save(model.state_dict(), model_save_path)
        print(f"Best model saved with validation loss: {val_loss:.4f}")

    # Adjust learning rate
    # scheduler.step(val_loss)