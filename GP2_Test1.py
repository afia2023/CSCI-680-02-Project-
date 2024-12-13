import torch
from transformers import GPT2LMHeadModel, GPT2Tokenizer
from datasets import load_from_disk
import pandas as pd
import sacrebleu

# Setup device for GPU if available
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Load tokenizer and model from pre-trained or saved directory
model_path = './GPT2_best_model'
tokenizer = GPT2Tokenizer.from_pretrained("distilgpt2", pad_token="<|endoftext|>")
model = GPT2LMHeadModel.from_pretrained(model_path).to(device)
model.eval()  # Set model to evaluation mode

# Load the test dataset
dataset_path = 'data1/dataset_dict'
datasets = load_from_disk(dataset_path)
test_dataset = datasets['test']

def generate_text_and_compute_scores(test_dataset, model, tokenizer, device):
    results = []
    predictions = []
    references = []
    for example in test_dataset:
        input_ids = torch.tensor(example['input_ids']).unsqueeze(0).to(device)
        labels = torch.tensor(example['labels']).unsqueeze(0).to(device)
        
        # Generate outputs
        with torch.no_grad():
            outputs = model.generate(input_ids, max_length=128, num_beams=5, early_stopping=True)
        
        generated_text = tokenizer.decode(outputs[0], skip_special_tokens=True)
        input_text = tokenizer.decode(input_ids[0], skip_special_tokens=True)
        documentation = tokenizer.decode(labels[0], skip_special_tokens=True)
        
        predictions.append(generated_text)
        references.append([documentation])  # BLEU expects a list of possible references
        
        # Collect results for CSV
        results.append({
            "method_code": input_text,
            "documentation": documentation,
            "generated_text": generated_text
        })

    # Compute BLEU and Accuracy
    bleu_score = sacrebleu.corpus_bleu(predictions, references).score
    accuracy = sum([1 if pred == ref[0] else 0 for pred, ref in zip(predictions, references)]) / len(predictions) * 100
    
    return results, accuracy, bleu_score

def save_results_to_csv(results, accuracy, bleu_score):
    df = pd.DataFrame(results[:5])  # Only saving first 5 for display
    df.loc['Overall'] = ["Accuracy:", accuracy, "BLEU Score:", bleu_score, ""]
    df.to_csv("test_results.csv", index=False)
    print("Results and scores have been saved to 'GPT2_test_output.csv'.")

# Generate text, compute scores, and save results
results, accuracy, bleu_score = generate_text_and_compute_scores(test_dataset, model, tokenizer, device)
save_results_to_csv(results, accuracy, bleu_score)
