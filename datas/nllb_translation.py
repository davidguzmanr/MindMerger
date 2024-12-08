import json
import random
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM
import torch
from datasets import load_dataset
from tqdm import tqdm

# Load the MetaMathQA dataset from Hugging Face Datasets
dataset = load_dataset("meta-math/MetaMathQA")['train']

# Define the target languages and their FLORES-200 language codes
languages = {
    'French': 'fra_Latn',
    'Amharic': 'amh_Ethi',
    'Ewe': 'ewe_Latn',
    'Hausa': 'hau_Latn',
    'Igbo': 'ibo_Latn',
    'Kinyarwanda': 'kin_Latn',
    'Lingala': 'lin_Latn',
    'Luganda': 'lug_Latn',
    'Oromo': 'orm_Latn',
    'Shona': 'sna_Latn',
    'Sotho': 'sot_Latn',
    'Swahili': 'swa_Latn',
    'Twi': 'twi_Latn',
    'Wolof': 'wol_Latn',
    'Xhosa': 'xho_Latn',
    'Yoruba': 'yor_Latn',
    'Zulu': 'zul_Latn'
}

# Initialize the tokenizer and model
model_name = "facebook/nllb-200-3.3B"
tokenizer = AutoTokenizer.from_pretrained(model_name, use_fast=False)
model = AutoModelForSeq2SeqLM.from_pretrained(model_name)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model.to(device)

# Define the batch size and number of samples per language
batch_size = 8
num_samples_per_language = 3000

# Function to translate a batch of queries
def translate_batch(queries, target_lang_code):
    tokenizer.src_lang = 'eng_Latn'
    tokenizer.tgt_lang = target_lang_code
    encoded = tokenizer(queries, return_tensors="pt", padding=True, truncation=True).to(device)
    
    with torch.no_grad():
        generated_tokens = model.generate(
            **encoded,
            forced_bos_token_id=tokenizer.lang_code_to_id[target_lang_code],
            max_length=512
        )
    translations = tokenizer.batch_decode(generated_tokens, skip_special_tokens=True)
    return translations

# List to hold all translated data across languages
combined_data_list = []

# For each target language, sample 3,000 data points and translate in batches
for lang_name, lang_code in languages.items():
    print(f"Processing language: {lang_name}")

    # Sample 3,000 random records from the dataset
    sampled_data = random.sample(list(dataset), num_samples_per_language)

    # Process the data in batches
    for i in tqdm(range(0, num_samples_per_language, batch_size)):
        batch_data = sampled_data[i:i + batch_size]
        queries_en = [item['query'] for item in batch_data]
        responses = [item['response'] for item in batch_data]
        types = [item['type'] for item in batch_data]

        # Translate the batch of queries
        translated_queries = translate_batch(queries_en, lang_code)

        # Create records for each translated query and append to the combined list
        for query_translated, query_en, response, type_ in zip(translated_queries, queries_en, responses, types):
            data_record = {
                "query": query_translated,
                "query_en": query_en,
                "response": response,
                "lang": lang_name,
                "type": type_
            }
            combined_data_list.append(data_record)

# Save the entire combined data to a single JSON file
output_file = "metamathqa.json"
with open(output_file, 'w', encoding='utf-8') as f:
    json.dump(combined_data_list, f, ensure_ascii=False, indent=4)

print(f"Saved {len(combined_data_list)} records to {output_file}")