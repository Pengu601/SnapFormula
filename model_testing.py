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
from tokenizers.trainers import WordLevelTrainer
from tokenizers.models import WordLevel
from transformers import VisionEncoderDecoderModel, TrOCRProcessor

# Function to test the model with a single image
def preprocess_image(image_path, transform):
    """
    Preprocess the input image using the same transforms as during training.
    """
    image = Image.open(image_path).convert("RGB")
    image = transform(image)  # Apply the same transformations
    return image.unsqueeze(0)  # Add a batch dimension


def predict_latex(model, image_path, processor, tokenizer, device, max_seq_length):
    """
    Predict the LaTeX formula for a given image.
    """
    # Step 1: Preprocess the image
    
    
    # Step 2: Encode the image
    image = Image.open(image_path).convert("RGB")
    pixel_values = processor(image, return_tensors="pt").pixel_values
    # Step 3: Initialize decoding
    output = model.generate(pixel_values)
    generated_text = processor.batch_decode(output, skip_special_tokens = True)[0]
   
    return generated_text

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



cuda_ = "cuda:0"
device = torch.device(cuda_ if torch.cuda.is_available() else "cpu")
csv_file = 'train_split.csv'
df = pd.read_csv(csv_file)

formula = df['formula']

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

tokenizer.save("tokenizer-wordlevel.json")

model = VisionEncoderDecoderModel.from_pretrained('microsoft/trocr-small-stage1')
model.config.decoder_start_token_id = tokenizer.token_to_id("[CLS]")
model.config.pad_token_id = tokenizer.token_to_id("[PAD]")
model.config.eos_token_id = tokenizer.token_to_id("[SEP]")
model.tokenizer = tokenizer
model.vocab_size = 600
model.max_length = 150



# model = architecture.ImageToLatexModel(vocab_size = tokenizer.vocab_size(), max_seq_length= 128).to(device)
# model.load_state_dict(torch.load(model_save_path))
# model.eval()  # Set model to evaluation mode

# # Define the image transformation pipeline
# transform = transforms.Compose([
#     transforms.Resize((224, 224)),  # Resize to the expected input size
#     transforms.ToTensor(),         # Convert to PyTorch tensor
#     transforms.Normalize(IMAGENET_DEFAULT_MEAN, IMAGENET_DEFAULT_STD)  # Normalize pixel values to [-1, 1]
# ])



# # Test the model with a sample image
processor = TrOCRProcessor.from_pretrained("microsoft/trocr-small-printed")

predicted_formula = predict_latex(model, test_image_path, processor, tokenizer, device, 150)
print(f"Predicted LaTeX Formula: {predicted_formula}")