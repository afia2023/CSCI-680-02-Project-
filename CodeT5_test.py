import json
import pandas as pd
from transformers import AutoTokenizer, T5ForConditionalGeneration
import torch
import sacrebleu

# Load tokenizer and model
tokenizer = AutoTokenizer.from_pretrained("Salesforce/codet5-base-multi-sum")
model = T5ForConditionalGeneration.from_pretrained("/home/afarjana/Code_Summarization/T5/T5_best_model").to('cuda')

# Load the JSON test data
with open('CodeT5_test.json', 'r') as file:
    test_dataset = json.load(file)

# Evaluation function
def evaluate_model(model, tokenizer, dataset):
    model.eval()
    total_correct = 0
    total_samples = 0
    references = []
    hypotheses = []
    results = []

    for item in dataset:
        # Prepare inputs and labels
        inputs = tokenizer("summarize: " + item['method_code'], return_tensors="pt", max_length=512, truncation=True, padding='max_length').to('cuda')
        with torch.no_grad():
            outputs = model.generate(inputs['input_ids'], max_length=150, num_beams=5, early_stopping=True)
        
        # Decode outputs to text
        decoded_output = tokenizer.decode(outputs[0], skip_special_tokens=True)
        references.append([item['documentation'].strip()])
        hypotheses.append(decoded_output.strip())

        # Collect results for CSV
        results.append({
            "Method Code": item['method_code'],
            "Documentation": item['documentation'].strip(),
            "Generated Text": decoded_output.strip()
        })

        # Check accuracy
        total_samples += 1
        if decoded_output.strip() == item['documentation'].strip():
            total_correct += 1

    # Calculate accuracy
    accuracy = total_correct / total_samples * 100

    # Calculate BLEU score using sacrebleu
    bleu_score = sacrebleu.corpus_bleu(hypotheses, references).score

    return accuracy, bleu_score, results

# Evaluate the model
accuracy, bleu_score, results = evaluate_model(model, tokenizer, test_dataset)
print(f"Test Set Accuracy: {accuracy:.2f}%")
print(f"Test Set BLEU Score: {bleu_score:.2f}")

# Save results to a CSV file
df = pd.DataFrame(results)
df.to_csv("CodeT5_Code_Summarization_Results.csv", index=False)
print("Results saved to test_output.csv.")
