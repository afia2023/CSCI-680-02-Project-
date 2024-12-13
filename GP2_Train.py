import os
import torch
from transformers import GPT2LMHeadModel, GPT2Tokenizer, Trainer, TrainingArguments, DataCollatorForLanguageModeling, AdamW
from datasets import load_from_disk
from torch.cuda.amp import GradScaler, autocast
from torch.nn import DataParallel

# Setting up environment and device
os.environ["CUDA_VISIBLE_DEVICES"] = "4,5"
os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "max_split_size_mb:128"
torch.backends.cudnn.enabled = False
torch.backends.cudnn.benchmark = False
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f'Using device: {device}')

# Load and prepare data
dataset_path = 'data1/dataset_dict'
full_datasets = load_from_disk(dataset_path)
datasets = {split: full_datasets[split].shuffle(seed=42).select(range(int(0.5 * len(full_datasets[split])))) for split in full_datasets.keys()}

# Initialize tokenizer and model
tokenizer = GPT2Tokenizer.from_pretrained("distilgpt2")
tokenizer.pad_token = tokenizer.eos_token
model = GPT2LMHeadModel.from_pretrained("distilgpt2")
model.to(device)

# Data collation
data_collator = DataCollatorForLanguageModeling(tokenizer=tokenizer, mlm=False)

# Setup training arguments
training_args = TrainingArguments(
    output_dir='./results',
    evaluation_strategy='epoch',
    save_strategy='epoch',
    logging_strategy='epoch',
    learning_rate=5e-5,
    per_device_train_batch_size=1,
    per_device_eval_batch_size=1,
    num_train_epochs=5,
    weight_decay=0.01,
    gradient_accumulation_steps=4,
    fp16=True,
    remove_unused_columns=False,
    save_total_limit=1,
    load_best_model_at_end=True,
    metric_for_best_model='loss',
    logging_dir='./logs',
    log_level='info'
)

# Trainer for evaluation
trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=datasets['train'],
    eval_dataset=datasets['validation'],
    data_collator=data_collator
)

# Initialize optimizer and scaler
optimizer = AdamW(model.parameters(), lr=5e-5)
scaler = GradScaler()

# Manual training loop
def train_model():
    model.train()
    for epoch in range(training_args.num_train_epochs):
        for step, batch in enumerate(trainer.get_train_dataloader()):
            inputs, labels = batch['input_ids'], batch['labels']
            inputs, labels = inputs.to(device), labels.to(device)
            
            with autocast():
                outputs = model(inputs, labels=labels)
                loss = outputs.loss.mean()
                loss = loss / training_args.gradient_accumulation_steps

            scaler.scale(loss).backward()

            if (step + 1) % training_args.gradient_accumulation_steps == 0:
                scaler.step(optimizer)
                scaler.update()
                optimizer.zero_grad()

            model.zero_grad()

        # Evaluate the model using Trainer to handle metrics calculation
        trainer.evaluate()

    # Save the best model
    if hasattr(model, 'module'):
        model.module.save_pretrained('./GPT2_best_model')
    else:
        model.save_pretrained('./GPT2_best_model')
    print("Model saved to './GPT2_best_model'.")

train_model()
