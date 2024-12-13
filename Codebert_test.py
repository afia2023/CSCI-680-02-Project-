import torch
from transformers import RobertaTokenizer, RobertaForMaskedLM
import pandas as pd
from nltk.translate.bleu_score import sentence_bleu
from sklearn.metrics import accuracy_score

# Load the tokenizer and model
tokenizer = RobertaTokenizer.from_pretrained('/home/afarjana/Code_Summarization/Codebert_version2/best3_codebert')
model = RobertaForMaskedLM.from_pretrained('/home/afarjana/Code_Summarization/Codebert_version2/best3_codebert')
model.eval()
model.to('cuda' if torch.cuda.is_available() else 'cpu')

# Load your test set
test_data = pd.read_json('/home/afarjana/Code_Summarization/Codebert_version2/Codebert_test.json')

# Function to calculate BLEU-4
def calculate_bleu(reference, candidate):
    return sentence_bleu([reference.split()], candidate.split(), weights=(0.25, 0.25, 0.25, 0.25))

# Function to process and generate text
def generate_text(method_code, documentation):
    # Tokenize and prepare the input sequence
    unmasked_length = int(len(documentation.split()) * 0.30)
    unmasked_doc = ' '.join(documentation.split()[:unmasked_length])
    masked_doc = ' '.join(['<mask>'] * (len(documentation.split()) - unmasked_length))
    
    # Combine method code and documentation with special tokens
    input_text = f"{tokenizer.cls_token}{method_code}{tokenizer.sep_token}{unmasked_doc} {masked_doc}{tokenizer.eos_token}"
    input_ids = tokenizer.encode(input_text, return_tensors='pt')

    # Ensure the length does not exceed 512 tokens
    max_length = model.config.max_position_embeddings  # typically 512 for BERT-like models
    if input_ids.size(1) > max_length:
        input_ids = input_ids[:, :max_length]  # Truncate the tokens

    input_ids = input_ids.to('cuda' if torch.cuda.is_available() else 'cpu')

    # Generate text with a maximum of 100 tokens for the documentation
    with torch.no_grad():
        outputs = model(input_ids)
        predictions = outputs.logits.argmax(dim=-1)
    
    # Decode the generated tokens into text
    generated_doc = tokenizer.decode(predictions[0], skip_special_tokens=True)
    return generated_doc

# Evaluate on the test set
results = []
for index, item in test_data.iterrows():
    generated_doc = generate_text(item['method_code'], item['documentation'])
    accuracy = accuracy_score(item['documentation'].split(), generated_doc.split())
    bleu_score = calculate_bleu(item['documentation'], generated_doc)
    results.append([item['method_code'], item['documentation'], generated_doc, accuracy, bleu_score])
    if index == 4:  # Stop after 5 entries
        break

# Convert results to DataFrame
result_df = pd.DataFrame(results, columns=['Method Code', 'Documentation', 'Generated Text', 'Accuracy', 'BLEU Score'])

# Save to CSV
csv_path = "/mnt/data/CodeBERT_Test_Results.csv"
result_df.to_csv(csv_path, index=False)

# Output average scores and path to CSV
average_accuracy = sum([res[3] for res in results]) / len(results)
average_bleu = sum([res[4] for res in results]) / len(results)
print(f"Average Accuracy: {average_accuracy}, Average BLEU-4: {average_bleu}")
print(f"Results saved to {csv_path}")
