import os
from datasets import load_dataset, get_dataset_config_names
from tqdm import tqdm

# Define the target languages and their NLLB language codes
languages = {
    'French': 'fra_Latn',
    'Amharic': 'amh_Ethi',
    'Ewe': 'ewe_Latn',
    'Hausa': 'hau_Latn',
    'Igbo': 'ibo_Latn',
    'Kinyarwanda': 'kin_Latn',
    'Lingala': 'lin_Latn',
    'Luganda': 'lug_Latn',
    'Oromo': 'gaz_Latn',
    'Shona': 'sna_Latn',
    'Sotho': 'sot_Latn',
    'Swahili': 'swh_Latn',
    'Twi': 'twi_Latn',
    'Wolof': 'wol_Latn',
    'Xhosa': 'xho_Latn',
    'Yoruba': 'yor_Latn',
    'Zulu': 'zul_Latn'
}

language_mapping = {
    'fra_Latn': 'fr',  # French
    'amh_Ethi': 'am',  # Amharic
    'ewe_Latn': 'ee',  # Ewe
    'hau_Latn': 'ha',  # Hausa
    'ibo_Latn': 'ig',  # Igbo
    'kin_Latn': 'rw',  # Kinyarwanda
    'lin_Latn': 'ln',  # Lingala
    'lug_Latn': 'lg',  # Luganda
    'gaz_Latn': 'om',  # Oromo
    'sna_Latn': 'sn',  # Shona
    'sot_Latn': 'st',  # Southern Sotho
    'swh_Latn': 'sw',  # Swahili
    'twi_Latn': 'tw',  # Twi
    'wol_Latn': 'wo',  # Wolof
    'xho_Latn': 'xh',  # Xhosa
    'yor_Latn': 'yo',  # Yoruba
    'zul_Latn': 'zu'   # Zulu
}

source_lang = 'eng_Latn'
num_samples = 9000

for lang_name, target_lang_code in tqdm(languages.items()):
    config_names = get_dataset_config_names("allenai/nllb")
    
    if f"{target_lang_code}-{source_lang}" in config_names:
        pair_code = f"{target_lang_code}-{source_lang}"
    else:
        pair_code = f"{source_lang}-{target_lang_code}"

    # Load dataset in streaming mode
    dataset = load_dataset(
        "allenai/nllb",
        pair_code,
        split="train",
        streaming=True
    )

    # Initialize storage lists
    source_sentences = []
    target_sentences = []

    # Iterate through the stream and stop after collecting `num_samples`
    for i, item in enumerate(tqdm(dataset, desc=f"Extracting sentences for {lang_name}")):
        if i >= num_samples:
            break

        source_sentence = item['translation'][source_lang]
        target_sentence = item['translation'][target_lang_code]
        source_sentences.append(source_sentence)
        target_sentences.append(target_sentence)

    # Create directory for language pair
    dir_name = f"african_bilingual_pairs/en-{language_mapping[target_lang_code]}"
    os.makedirs(dir_name, exist_ok=True)

    # Save sentences to files
    source_file = os.path.join(dir_name, f"train.en")
    target_file = os.path.join(dir_name, f"train.{language_mapping[target_lang_code]}")

    with open(source_file, 'w', encoding='utf-8') as f_src, open(target_file, 'w', encoding='utf-8') as f_tgt:
        for src_line, tgt_line in zip(source_sentences, target_sentences):
            f_src.write(src_line.strip() + '\n')
            f_tgt.write(tgt_line.strip() + '\n')

    print(f"Saved {len(source_sentences)} sentences to {dir_name}")