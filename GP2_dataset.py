import pandas as pd
from transformers import GPT2Tokenizer
from datasets import Dataset
from sklearn.model_selection import train_test_split
from datasets import Dataset, load_dataset, DatasetDict
from transformers import GPT2Tokenizer
import pandas as pd
import os

def load_and_prepare_data(file_path):
    df = pd.read_json(file_path)
    tokenizer = GPT2Tokenizer.from_pretrained("gpt2")
    tokenizer.pad_token = tokenizer.eos_token  # Set pad token

    def prepare_and_tokenize(row):
        input_text = f"{row['method_code']} {tokenizer.eos_token} {row['documentation']}"
        input_ids = tokenizer(input_text, truncation=True, padding="max_length", max_length=128, return_tensors='pt')['input_ids'].squeeze().tolist()
        labels = tokenizer(row['documentation'], truncation=True, padding="max_length", max_length=128, return_tensors='pt')['input_ids'].squeeze().tolist()
        return {'input_ids': input_ids, 'labels': labels}

    # Apply the tokenization and preparation to each row in the DataFrame
    tokenized_data = df.apply(prepare_and_tokenize, axis=1, result_type='expand')

    # Convert the DataFrame to a Hugging Face dataset
    dataset = Dataset.from_pandas(pd.DataFrame(tokenized_data))
    return dataset

def split_and_save_data(dataset):
    # Split the dataset
    train_val_split = dataset.train_test_split(test_size=0.1)
    train_split = train_val_split['train'].train_test_split(test_size=0.1)

    # Create a DatasetDict
    dataset_dict = {
        'train': train_split['train'],
        'validation': train_split['test'],
        'test': train_val_split['test']
    }

    # Save the dataset
    os.makedirs('data1', exist_ok=True)
    DatasetDict(dataset_dict).save_to_disk('data1/dataset_dict')

def main():
    file_path = 'preprocessed_dataset.json'  # Ensure this is the correct path
    dataset = load_and_prepare_data(file_path)
    split_and_save_data(dataset)
    print("Datasets have been split and saved successfully.")

if __name__ == "__main__":
    main()
