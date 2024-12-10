import torch
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM
from tqdm import tqdm
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
model_name = "facebook/nllb-200-distilled-600M"
tokenizer = AutoTokenizer.from_pretrained(model_name, use_fast=False)
model = AutoModelForSeq2SeqLM.from_pretrained(model_name).to(device)  # Move the model to the appropriate device

langs = ["ay", "gn", "qu"]

for lang in langs:
    with open(f"americas_blilingual_pairs/en-{lang}/train.es", encoding="utf-8") as f:
        sentences = f.readlines()

    # Process the translations in batches
    translated_sentences = translate_sentences_in_batches(
        model,
        tokenizer,
        sentences,
        source_lang="spa_Latn",
        target_lang="eng_Latn",
        batch_size=128
    )

    output_file = f"americas_blilingual_pairs/en-{lang}/train.en"
    with open(output_file, "w", encoding="utf-8") as file:
        for translation in translated_sentences:
            file.write(f"{translation}\n")
