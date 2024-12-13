import torch
import os
import pandas as pd
import re
import numpy as np
from sklearn.model_selection import train_test_split
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM
from transformers import Trainer, TrainingArguments, DataCollatorForSeq2Seq ,TrainerCallback
from datasets import Dataset
from nltk.stem import WordNetLemmatizer
from evaluate import load
import nltk
import json
import shutil


#os.environ['CUDA_VISIBLE_DEVICES'] = '3'
os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "max_split_size_mb:128"

tokenizer = AutoTokenizer.from_pretrained("Salesforce/codet5-base-multi-sum")
model = AutoModelForSeq2SeqLM.from_pretrained("Salesforce/codet5-base-multi-sum")

# Load the preprocessed dataset
preprocessed_data = pd.read_json('preprocessed_dataset.json', lines=True)

# Add prefix to each code snippet
preprocessed_data['method_code'] = preprocessed_data['method_code'].apply(lambda x: "summarize: " + x)

# Split dataset into training, validation, and test sets
train_df, temp_df = train_test_split(preprocessed_data, test_size=0.30, random_state=42)
val_df, test_df = train_test_split(temp_df, test_size=0.20, random_state=42)

# Function to tokenize the data
def tokenize_data(dataframe):
    def tokenize_function(examples):
        model_inputs = tokenizer(examples['method_code'], max_length=128, padding="max_length", truncation=True)
        # Ensure labels are also tokenized similarly
        with tokenizer.as_target_tokenizer():
            labels = tokenizer(examples['documentation'], max_length=128, padding="max_length", truncation=True)
        model_inputs['labels'] = labels['input_ids']
        return model_inputs
    return Dataset.from_pandas(dataframe).map(tokenize_function, batched=True)

# Tokenize the datasets
train_dataset = tokenize_data(train_df)
val_dataset = tokenize_data(val_df)

# Save the test dataset for later evaluation
test_df.to_json('test.json', orient='records', lines=True)


class SaveBestCheckpointsCallback(TrainerCallback):
    """A callback that saves the best N checkpoints based on a specific metric."""
    def __init__(self, save_path, max_save=2, metric="eval_loss"):
        self.best_scores = []
        self.checkpoint_paths = []
        self.save_path = save_path
        self.max_save = max_save
        self.metric = metric

    def on_evaluate(self, args, state, control, metrics=None, **kwargs):
        current_score = metrics.get(self.metric)
        current_checkpoint_path = f"{self.save_path}/checkpoint-{state.global_step}"
        
        if len(self.best_scores) < self.max_save or current_score < max(self.best_scores):
            model.save_pretrained(current_checkpoint_path)
            self.best_scores.append(current_score)
            self.checkpoint_paths.append(current_checkpoint_path)
            
            if len(self.best_scores) > self.max_save:
                worst_idx = self.best_scores.index(max(self.best_scores))
                worst_checkpoint_path = self.checkpoint_paths[worst_idx]
                
                # Remove the worst checkpoint
                del self.best_scores[worst_idx]
                del self.checkpoint_paths[worst_idx]
                shutil.rmtree(worst_checkpoint_path)  # Delete the directory

# Training arguments for the Trainer
training_args = TrainingArguments(
    output_dir='./results',
    eval_strategy='steps',
    eval_steps=50,  # Evaluate every 50 steps
    save_strategy='steps',
    save_steps=50,  # Save every 50 steps
    learning_rate=2e-5,
    per_device_train_batch_size=2,  # Adjust batch size based on available memory
    gradient_accumulation_steps=4,  # Adjust based on your memory and batch size
    per_device_eval_batch_size=2,
    num_train_epochs=10,
    save_total_limit=2,
    load_best_model_at_end=True,
    metric_for_best_model='eval_loss',
    logging_dir='./logs',
    fp16=True
)


# Create the Trainer object
trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=train_dataset,
    eval_dataset=val_dataset,
    data_collator=DataCollatorForSeq2Seq(tokenizer, model=model),
    callbacks=[SaveBestCheckpointsCallback('./results', max_save=2)]
)

# Train the model
# Add the optimization memory management technique
torch.backends.cuda.matmul.allow_tf32 = True  # Enable TF32 on Ampere GPUs to speed up training
torch.backends.cudnn.benchmark = True  # Enable cudnn auto-tuning for performance
torch.cuda.empty_cache()


# Save the trained model to disk
model_path = './T5_best_model'
if not os.path.exists(model_path):
    os.makedirs(model_path)
model.save_pretrained(model_path)
print(f"Model saved to {model_path}")