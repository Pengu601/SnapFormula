import torch
from torch import nn
from torch.optim import Adam
from torchvision import transforms
from torch.utils.data import DataLoader
import architecture
from torch.optim.lr_scheduler import StepLR
from timm.data.constants import IMAGENET_DEFAULT_MEAN, IMAGENET_DEFAULT_STD
from tqdm import tqdm
from collections import Counter
import math
from transformers import VisionEncoderDecoderModel, TrOCRProcessor
from transformers import PreTrainedTokenizerFast

model_save_path = 'E:\\ai stuf\\best_model.pth'
latex_vocab_path = "D:\\GitHub Projects\\SnapFormula\\latex_vocab.json"
from tokenizers import Tokenizer
from tokenizers.models import WordLevel
max_length = 150
tokenizer = Tokenizer(WordLevel(unk_token="[UNK]"))
tokenizer.enable_padding(length=max_length)
tokenizer.enable_truncation(max_length=max_length)


from tokenizers.trainers import WordLevelTrainer
trainer = WordLevelTrainer(special_tokens=["[UNK]", "[CLS]", "[SEP]", "[PAD]", "[MASK]"],
                     vocab_size=600,
                     show_progress=True,
                     )

from tokenizers.pre_tokenizers import Whitespace
tokenizer.pre_tokenizer = Whitespace()

from tokenizers.processors import TemplateProcessing
tokenizer.post_processor = TemplateProcessing(
    single="[CLS] $A [SEP]",
    pair="[CLS] $A [SEP] $B:1 [SEP]:1",
    special_tokens=[
        ("[CLS]", 1),
        ("[SEP]", 2),
    ],
)


files = ["D:\\archive\\im2latex_formulas.norm.lst"]

tokenizer.train(files, trainer)
processor = TrOCRProcessor.from_pretrained("microsoft/trocr-small-printed")
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
   
    tokenizer=tokenizer,
    processor = processor
)

val_dataset = architecture.ImageLatexDatasetCSV(
    csv_file='val_split.csv',
    image_dir='D:\\archive\\formula_images_processed\\formula_images_processed',
    
    tokenizer=tokenizer,
    processor = processor
)

# print(tokenizer.vocab_size())
# for token, token_id in tokenizer.vocab.items():
#     print(f"Token: {token}, ID: {token_id}")

train_loader = DataLoader(test_dataset, batch_size=64, shuffle=False, pin_memory=True, )
val_loader = DataLoader(val_dataset, batch_size=64, shuffle=False, pin_memory=True, ) 
# Example training loop
  # Define your model

cuda_ = "cuda:0"


# token_counts = Counter()
# for token_ids, _ in train_loader:
#     token_ids_flat = token_ids.view(-1).tolist()
#     token_counts.update(token_ids_flat)
# print(token_counts.most_common(10))  # Print 10 most common tokens


device = torch.device(cuda_ if torch.cuda.is_available() else "cpu")

model = VisionEncoderDecoderModel.from_pretrained('microsoft/trocr-small-stage1').to(device)
model.config.decoder_start_token_id = tokenizer.token_to_id("[CLS]")
model.config.pad_token_id = tokenizer.token_to_id("[PAD]")
model.config.eos_token_id = tokenizer.token_to_id("[SEP]")
model.tokenizer = tokenizer
model.vocab_size = 600
model.max_length = 150
# model.load_state_dict(torch.load(model_save_path))
optimizer = Adam(model.parameters(), lr=0.0001,  weight_decay=1e-4)
# token_weights = torch.ones(tokenizer.vocab_size()).to(device)

k = 2  # Smoothing factor
# for token, count in token_counts.items():
#     token_weights[token] = 1 / (math.log(count + k) + 1)

# Normalize weights to avoid large disparities
# token_weights = token_weights / token_weights.mean()
    


# criterion = nn.CrossEntropyLoss(weight=token_weights, ignore_index = 0)
# model.classifier == nn.Linear(model.classifier.in_features, tokenizer.vocab_size())

# model.load_state_dict(torch.load(model_save_path))

def train_epoch(model, dataloader, optimizer, device):
    model.train()
    epoch_loss = 0
    total_tokens = 0
    correct_tokens = 0
    correct_tokens_debug = 0
    total_tokens_debug = 0
    # count = 1
    for _, batch  in enumerate(tqdm(dataloader)):
        for k, v in batch.items():
            batch[k] = v.to(device)
        
        
        
        outputs = model(**batch)
        loss = outputs.loss
        loss.backward() #backwards propogation
        optimizer.step()
        optimizer.zero_grad()
        epoch_loss += loss.item()
        # print(epoch_loss) 
        
        # Calculate training accuracy
        # Calculate token-level accuracy
        # preds = torch.argmax(logits, dim=-1)  # Shape: (batch_size * seq_length)
        
        # preds_debug = preds.view(images.size(0), -1)  # Reshape predictions
        # labels_debug = labels.view(images.size(0), -1)  # Reshape labels
        
        # print(preds_debug)
        # print(labels_debug)
       
        # Decode ground truth and predicted tokens
        # for i in range(min(1, labels.size(0))):
                 
        #     gt_tokens = labels_debug[i].tolist()
        #     pred_tokens = preds_debug[i].tolist()
        #     gt_latex = tokenizer.decode(labels_debug[i].tolist())
        #     pred_latex = tokenizer.decode(preds_debug[i].tolist())
            
        #     non_pad_mask_debug = (labels_debug[i] != tokenizer.vocab['[CLS]']) & (labels_debug[i] != tokenizer.vocab['[SEP]']) & (labels_debug[i] != tokenizer.vocab['[PAD]'])  # Create a mask for valid tokens (excluding [CLS] and [SEP])
        #     correct_tokens_debug = ((preds_debug[i] == labels_debug[i]) & non_pad_mask_debug).sum().item()
        #     total_tokens_debug = non_pad_mask_debug.sum().item()
        #     acc = correct_tokens_debug/total_tokens_debug
        #     # Print comparison
        #     # print(f"  Ground Truth (Token): {gt_tokens}")
        #     # print(f"  Predicted (Token): {pred_tokens}")
        #     print(f"  Ground Truth (LaTeX): {gt_latex}")
        #     print(f"  Predicted (LaTeX): {pred_latex}")
        #     print(f"  Accuracy: {acc}")

       
        # non_pad_mask = (labels != tokenizer.vocab['[CLS]']) & (labels != tokenizer.vocab['[SEP]']) & (labels != tokenizer.vocab['[PAD]'])  # Create a mask for valid tokens (excluding [CLS] and [SEP])
        # correct_tokens += (preds == labels)[non_pad_mask].sum().item()
        # total_tokens += non_pad_mask.sum().item()
        
        # count+= 1
    # accuracy = correct_tokens / total_tokens if total_tokens > 0 else 0
    return epoch_loss / len(dataloader)

def validate_epoch(model, dataloader, device):
    model.eval()
    epoch_loss = 0
    total_tokens = 0
    correct_tokens = 0
    with torch.no_grad():
        for _, batch  in enumerate(tqdm(dataloader)):
            for k, v in batch.items():
                batch[k] = v.to(device)
            
            
            
            outputs = model(**batch)
            loss = outputs.loss
            epoch_loss += loss.item()
    
    return epoch_loss / len(dataloader)


num_epochs = 20
best_val_loss = 500 # Initialize to a small value
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
    train_loss = train_epoch(model, train_loader, optimizer, device)
    print(f"Train Loss: {train_loss:.4f}")
    

    # Validate
    val_loss = validate_epoch(model, val_loader,  device)
    print(f"Validation Loss: {val_loss:.4f}")

    # Save the best model
    if val_loss < best_val_loss:
        best_val_loss = val_loss
        torch.save(model.state_dict(), model_save_path)
        print(f"Best model saved with validation loss: {val_loss:.4f}")

    # Adjust learning rate
    scheduler.step(val_loss)