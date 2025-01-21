import torch
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM
from tqdm import tqdm
import json
from typing import List

def translate_sentences_in_batches(
    model,
    tokenizer,
    sentences: List[str],
    source_lang: str,
    target_lang: str,
    batch_size: int = 32
) -> List[str]:
    """
    Translate sentences in batches to optimize inference speed.
    """
    tokenizer.src_lang = source_lang
    tokenizer.tgt_lang = target_lang

    translated_sentences = []
    for i in tqdm(range(0, len(sentences), batch_size)):
        batch = sentences[i:i + batch_size]

        # Tokenize the input batch and move tensors to GPU if available
        inputs = tokenizer(batch, return_tensors="pt", padding=True, truncation=True, max_length=512)
        inputs = {key: value.to(model.device) for key, value in inputs.items()}

        # Generate translations for the batch
        outputs = model.generate(
            **inputs,
            forced_bos_token_id=tokenizer.convert_tokens_to_ids(target_lang),
            max_length=512
        )

        # Decode and append the results for the batch
        translated_sentences.extend(
            tokenizer.batch_decode(outputs, skip_special_tokens=True)
        )

    return translated_sentences


# Check if a GPU is available and set the device accordingly
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Load the model and tokenizer
model_name = "facebook/nllb-200-distilled-1.3B"
tokenizer = AutoTokenizer.from_pretrained(model_name, use_fast=False)
model = AutoModelForSeq2SeqLM.from_pretrained(model_name).to(device)  # Move the model to the appropriate device

languages = {
    'Aymara': {"nllb_code": "ayr_Latn", "huggingface_code": "aym"},
    'Guarani': {"nllb_code": "grn_Latn", "huggingface_code": "gn"},
    'Quechua': {"nllb_code": "quy_Latn", "huggingface_code": "quy"}
}

data = []
id_to_label = {
    0: "entailment",
    1: "neutral",
    2: "contradiction"
}
for lang_name, lang_info in tqdm(languages.items()):
    nllb_code = lang_info["nllb_code"]
    huggingface_code = lang_info["huggingface_code"]

    dataset = load_dataset("nala-cub/americas_nli", huggingface_code)["validation"]

    premises = dataset["premise"]
    hypotheses = dataset["hypothesis"]
    labels = dataset["label"]

    # Process the translations in batches
    translated_premises = translate_sentences_in_batches(
        model,
        tokenizer,
        premises,
        source_lang=nllb_code,
        target_lang="spa_Latn",
        batch_size=4
    )

    # Process the translations in batches
    translated_hypotheses = translate_sentences_in_batches(
        model,
        tokenizer,
        hypotheses,
        source_lang=nllb_code,
        target_lang="spa_Latn",
        batch_size=4
    )

    assert len(premises) == len(translated_premises)

    for i in range(len(dataset)):
        premise = premises[i]
        hypothesis = hypotheses[i]
        
        premise_es = translated_premises[i]
        hypothesis_es = translated_hypotheses[i]
        
        label = labels[i]

        data_record = {
            # "query": query_translated,
            # "query_en": query_en,
            "premise": premise,
            "premise_es": premise_es,
            "hypothesis": hypothesis,
            "hypothesis_es": hypothesis_es,
            "response": id_to_label[label],
            "lang": lang_name,
            "type": "NLI"
        }
        data.append(data_record)

# Save the entire combined data to a single JSON file
output_file = "AmericasNLI-en.json"
with open(output_file, 'w', encoding='utf-8') as f:
    json.dump(data, f, ensure_ascii=False, indent=4)