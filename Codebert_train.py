import torch
import os
from transformers import RobertaTokenizer
from torch.utils.data import Dataset, DataLoader
import pandas as pd
from sklearn.model_selection import train_test_split
import numpy as np
import random 
from transformers import RobertaForMaskedLM, AdamW, get_linear_schedule_with_warmup
from tqdm import tqdm

# Define dataset class
import torch
from torch.utils.data import Dataset

device = torch.device("cuda:3" if torch.cuda.is_available() else "cpu")

#os.environ['CUDA_VISIBLE_DEVICES'] = '3'
os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "max_split_size_mb:128"

class CodeDocumentationDataset(Dataset):
    def __init__(self, method_code, documentation, tokenizer, max_length=512):
        self.tokenizer = tokenizer
        self.method_code = method_code  # Corrected from method_codes to method_code
        self.documentation = documentation  # Already correct
        self.max_length = max_length

    def __len__(self):
        return len(self.method_code)
    
    def __getitem__(self, idx):
        # Tokenize method code and documentation separately
        tokenized_code = self.tokenizer(self.method_code[idx], add_special_tokens=False, truncation=True, max_length=self.max_length//2)
        tokenized_doc = self.tokenizer(self.documentation[idx], add_special_tokens=False, truncation=True, max_length=self.max_length//2)
        
        # Manually concatenate token ids with [CLS], [SEP], and [EOS]
        input_ids = [self.tokenizer.cls_token_id] + tokenized_code['input_ids'] + [self.tokenizer.sep_token_id] + tokenized_doc['input_ids'] + [self.tokenizer.eos_token_id]
        attention_mask = [1] * len(input_ids)

        # Ensure input length matches max_length
        if len(input_ids) > self.max_length:
            input_ids = input_ids[:self.max_length]
            attention_mask = attention_mask[:self.max_length]
        else:
            padding_length = self.max_length - len(input_ids)
            input_ids += [self.tokenizer.pad_token_id] * padding_length
            attention_mask += [0] * padding_length

        input_ids = torch.tensor(input_ids)
        attention_mask = torch.tensor(attention_mask)

        # Setup labels for masked language modeling
        labels = torch.full_like(input_ids, -100)  # Initially ignore all for loss computation
        doc_start = len(tokenized_code['input_ids']) + 2  # Starting index for documentation tokens after [CLS] and [SEP]
        doc_end = doc_start + len(tokenized_doc['input_ids'])

        # Choose up to 4 tokens to mask randomly in the documentation segment
        num_masks = min(4, doc_end - doc_start)  # Mask a maximum of 4 tokens, or fewer if not enough tokens
        mask_indices = random.sample(range(doc_start, doc_end), num_masks)

        # Displaying the masking information
        #print("Mask Indices:", mask_indices)
        #print("Original IDs for Mask:", [input_ids[idx].item() for idx in mask_indices])

        for idx in mask_indices:
            labels[idx] = input_ids[idx]  # Only consider the masked token's original ID in the loss
            input_ids[idx] = self.tokenizer.mask_token_id  # Replace the token with the mask token

        # Additional debug: Print the tokens before and after masking
        #print("Input IDs before Masking:", input_ids.tolist())
        #print("Labels after Masking:", labels.tolist())

        return {"input_ids": input_ids, "labels": labels, "attention_mask": attention_mask}

# Example usage
#tokenizer = RobertaTokenizer.from_pretrained('roberta-base')
tokenizer = RobertaTokenizer.from_pretrained('microsoft/codebert-base')
#method_codes = ["def print_hello(): print('Hello, world!')"]
#docs = ["This function prints Hello World to the console."]
#dataset = CodeDocumentationDataset(method_codes, docs, tokenizer)
#data_loader = DataLoader(dataset, batch_size=1, shuffle=True)


# Load dataset
df = pd.read_json('preprocessed_dataset.json')

# Split dataset into train, validation, and test sets
train_val_df, test_df = train_test_split(df, test_size=0.1, random_state=42)
train_df, val_df = train_test_split(train_val_df, test_size=0.1, random_state=42)

# Save test_df to a JSON file
test_df.to_json('test.json', orient='records', lines=True)

# Initialize tokenizer
tokenizer = RobertaTokenizer.from_pretrained('microsoft/codebert-base')

# Create datasets
train_dataset = CodeDocumentationDataset(train_df['method_code'].tolist(), train_df['documentation'].tolist(), tokenizer)
val_dataset = CodeDocumentationDataset(val_df['method_code'].tolist(), val_df['documentation'].tolist(), tokenizer)

# Create DataLoaders
train_loader = DataLoader(train_dataset, batch_size=4, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=4, shuffle=False)

# Checking the first item in the training DataLoader
#first_train_item = next(iter(train_loader))
#print("First Train Item Input IDs:", first_train_item['input_ids'])
#print("First Train Item Labels:", first_train_item['labels'])
#print("First Train Item Labels:", first_train_item['attention_mask'])

# Load model
model = RobertaForMaskedLM.from_pretrained("microsoft/codebert-base")

# Check CUDA availability and select the appropriate device
if torch.cuda.is_available() and torch.cuda.device_count() > 3:
    try:
        device_name = torch.cuda.get_device_name(3)
        print(f"Device name: {device_name}")
        device = torch.device("cuda:3")
        model.to(device)
    except Exception as e:
        print(f"Error: {e}")
        print("Falling back to CPU.")
        device = torch.device("cpu")
        model.to(device)
else:
    print("CUDA device 3 not available, using CPU.")
    device = torch.device("cpu")
    model.to(device)

print(f'Using device: {device}')

def clear_cuda_cache():
    torch.cuda.empty_cache()
    print("Cleared CUDA cache")

# Define optimizer and scheduler
optimizer = AdamW(model.parameters(), lr=1e-5)
total_steps = len(train_loader) * 20  # Assume 10 epochs
scheduler = get_linear_schedule_with_warmup(optimizer, num_warmup_steps=0, num_training_steps=total_steps)

def compute_accuracy(logits, labels):
    # Flatten all the labels
    labels_flat = labels.view(-1)
    # Flatten all the predictions
    logits_flat = logits.view(-1, logits.size(-1))
    # Get the predictions
    _, predictions = torch.max(logits_flat, dim=1)
    # Calculate accuracy only for masked tokens (labels != -100)
    mask = labels_flat != -100
    predictions = predictions[mask]
    labels_flat = labels_flat[mask]
    return torch.sum(predictions == labels_flat) / torch.numel(labels_flat) if torch.numel(labels_flat) > 0 else torch.tensor(0.0)

# Training loop
for epoch in range(20):  # Let's train for 10 epochs
    model.train()
    train_loss = 0
    for batch in tqdm(train_loader, desc=f"Training Epoch {epoch+1}"):
        input_ids = batch['input_ids'].to(device)
        labels = batch['labels'].to(device)
        attention_mask = batch['attention_mask'].to(device)  # Ensuring attention_mask is used
        optimizer.zero_grad()
        outputs = model(input_ids=input_ids, attention_mask=attention_mask, labels=labels)
        loss = outputs.loss
        train_loss += loss.item()
        loss.backward()
        optimizer.step()
        scheduler.step()

    avg_train_loss = train_loss / len(train_loader)
    print(f"Epoch {epoch+1} - Average Training Loss: {avg_train_loss:.4f}")

    # Validation loop
    model.eval()
    val_loss = 0
    total_accuracy = 0
    for batch in tqdm(val_loader, desc=f"Validating Epoch {epoch+1}"):
        input_ids = batch['input_ids'].to(device)
        labels = batch['labels'].to(device)
        attention_mask = batch['attention_mask'].to(device)  # Using attention_mask in validation
        with torch.no_grad():
            outputs = model(input_ids=input_ids, attention_mask=attention_mask, labels=labels)
            loss = outputs.loss
            val_loss += loss.item()
            accuracy = compute_accuracy(outputs.logits, labels)
            total_accuracy += accuracy.item()

    avg_val_loss = val_loss / len(val_loader)
    avg_val_accuracy = total_accuracy / len(val_loader)
    print(f"Epoch {epoch+1} - Average Validation Loss: {avg_val_loss:.4f}, Accuracy: {avg_val_accuracy:.4f}")


# Save the fine-tuned model and tokenizer
model.save_pretrained("best3_codebert")
tokenizer.save_pretrained("best3_codebert")