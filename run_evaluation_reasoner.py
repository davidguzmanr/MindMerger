import os
import json
import torch
import argparse
import ast
import deepspeed
from torch.utils.data import DataLoader, SequentialSampler

from transformers import AutoTokenizer
from mindmerger_tools.utils import set_seed
from mindmerger_tools.read_datasets import *
from mindmerger_tools.deepspeed_config import get_train_ds_config
from mindmerger_tools.input_features import mt_input_features, llm_input_features
from modeling_mindreasoner import MindReasoner
from evaluation import evaluate_math, evaluate_classification

# For loading LoRA adapter
try:
    from peft import PeftConfig, PeftModel
    PEFT_AVAILABLE = True
except ImportError:
    PEFT_AVAILABLE = False

def main(args):

    # ============ 1) Basic Setup ============
    set_seed(0)
    llm_path = args.llm_path
    mt_path = args.mt_path
    task = args.task
    use_lora = True  # In MindReasoner, to reflect the final (reasoning) stage

    # Typical max length config
    max_seq_len = args.max_seq_len
    max_gen_len = args.max_gen_len
    eval_batch_size = args.eval_batch_size

    # Result paths
    result_path_base = f"./results/{args.save_name}/{task}/"
    os.makedirs(result_path_base, exist_ok=True)

    # ============ 2) Select Test Set(s) Based on Task ============
    # Similar logic to your existing run_evaluation
    if 'mgsm' in task:
        test_sets = read_mgsms()
        task = 'math'
    elif 'nli' in task:
        test_sets = read_americas_xnli()
    elif 'msvamp' in task:
        test_sets = read_msvamp()
        task = 'math'
    elif 'csqa' in task:
        test_sets = read_x_csqa()
    else:
        test_sets = read_xnli()

    # ============ 3) Tokenizers ============
    tokenizer_m2m = AutoTokenizer.from_pretrained(mt_path)
    tokenizer_llm = AutoTokenizer.from_pretrained(llm_path, use_fast=True)
    tokenizer_llm.pad_token = tokenizer_llm.eos_token
    tokenizer_llm.padding_side = "left"

    # Log some info
    print(json.dumps({
        "llm_path": llm_path,
        "mt_path": mt_path,
        "max_seq_len": max_seq_len,
        "max_gen_len": max_gen_len,
        "save_name": args.save_name,
        "task": task,
        "result_path_base": result_path_base
    }, indent=2))

    # ============ 4) Prepare DeepSpeed Config ============
    train_micro_batch_size_per_gpu = args.train_micro_batch_size_per_gpu
    train_batch_size = args.train_batch_size
    gpu_num = torch.cuda.device_count()
    gradient_accumulation = train_batch_size // (train_micro_batch_size_per_gpu * gpu_num)
    ds_config = get_train_ds_config(
        train_batch_size=train_batch_size,
        train_micro_batch_size_per_gpu=train_micro_batch_size_per_gpu,
        gradient_accumulation_steps=gradient_accumulation,
    )

    # ============ 5) Build MindReasoner (with LoRA) ============
    # We set `use_lora=True` for the final stage
    model = MindReasoner(
        mt_path=mt_path,
        llm_path=llm_path,
        max_gen_len=max_gen_len,
        llm_bos_token_id=tokenizer_llm.bos_token_id,
        llm_pad_token_id=tokenizer_llm.pad_token_id,
        use_lora=True
    )

    # ============ 6) Load Mapping Checkpoint ============
    if args.init_mapping_checkpoint is not None:
        print(f"Loading mapping from: {args.init_mapping_checkpoint}")
        checkpoint = torch.load(args.init_mapping_checkpoint, map_location='cpu')
        model_dict = checkpoint['model_state_dict']
        model.mapping.load_state_dict(model_dict, strict=False)

    # ============ 7) Load LoRA Adapter ============
    # The final reasoning stage produces a LoRA folder, e.g. `lora_adapter/`.
    # Use the PEFT library to load the adapter into model.model_llm
    if args.lora_path is not None and PEFT_AVAILABLE:
        print(f"Loading LoRA adapter from: {args.lora_path}")
        model.model_llm = PeftModel.from_pretrained(
            model.model_llm,
            args.lora_path,
            is_trainable=False  # we only need it in eval mode
        )
    elif args.lora_path is not None:
        raise ImportError("peft is not installed. Please install via `pip install peft`.")

    # Freeze everything except LoRA (though we only do eval anyway)
    model.eval()

    # ============ 8) Initialize with DeepSpeed ============
    # This for a problem with mT5 models and DeepSpeed
    for param in model.parameters():
        if not param.data.is_contiguous():
            param.data = param.data.contiguous()
            
    parameters = filter(lambda p: p.requires_grad, model.parameters())
    model, _, _, _ = deepspeed.initialize(
        config=ds_config,
        model=model,
        model_parameters=parameters,
        training_data=None
    )

    # ============ 9) Evaluate ============

    # We'll store {language -> accuracy}, then compute average
    scores_map = {}
    avg_acc = 0.0

    for lang_name, test_data in test_sets.items():
        # Wrap data in a DataLoader
        test_sampler = SequentialSampler(test_data)
        # We'll reuse your MathDataset or a new dataset class:
        from mindmerger_tools.read_datasets import MathDataset
        test_dataset = MathDataset(test_data, task)
        test_dloader = DataLoader(
            test_dataset,
            batch_size=eval_batch_size,
            sampler=test_sampler,
            shuffle=False,
            num_workers=1,
            drop_last=False
        )

        # Evaluate: math or classification
        if task == 'math':
            acc, results_list = evaluate_math(
                model, test_dloader, tokenizer_llm, tokenizer_m2m,
                max_seq_len, max_gen_len, use_prompt=True,
                langs_map=langs_map
            )
        else:
            # e.g. classification tasks
            acc, results_list = evaluate_classification(
                model, test_dloader, tokenizer_llm, tokenizer_m2m,
                max_seq_len, max_gen_len, use_prompt=True,
                langs_map=langs_map
            )

        print(f"Lang={lang_name}, Accuracy={acc:.2f}")
        scores_map[lang_name] = acc
        avg_acc += acc

        # Optionally save each language’s predictions
        out_json_path = os.path.join(result_path_base, f"{lang_name}.json")
        with open(out_json_path, "w", encoding="utf-8") as f:
            json.dump(results_list, f, ensure_ascii=False, indent=2)

    # Summarize
    avg_acc /= len(test_sets)
    print("Final Scores:", scores_map)
    print("Average Accuracy:", round(avg_acc, 2))

    # Optionally save .tsv
    score_tsv_path = os.path.join(result_path_base, "scores.tsv")
    with open(score_tsv_path, "w", encoding="utf-8") as fout:
        for k, v in scores_map.items():
            fout.write(f"{k}\t{v}\n")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    parser.add_argument("--llm_path", type=str, default="google/gemma-2-9b-it")
    parser.add_argument("--mt_path",  type=str, default="facebook/nllb-200-distilled-600M")
    parser.add_argument("--task",     type=str, default="math")

    parser.add_argument("--save_name", type=str, default="AfriMindReasoner/Gemma2-NLLB")

    # Checkpoints
    parser.add_argument("--init_mapping_checkpoint", type=str,
                        default="outputs/AfriMindReasoner/Gemma2-NLLB/math/reasoning/mapping.bin",
                        help="Path to the final mapping checkpoint from Stage 3.")
    parser.add_argument("--lora_path", type=str,
                        default="outputs/AfriMindReasoner/Gemma2-NLLB/math/reasoning/lora_adapter",
                        help="Path to the LoRA adapter folder produced during Stage 3.")

    parser.add_argument("--max_seq_len", type=int, default=512)
    parser.add_argument("--max_gen_len", type=int, default=512)
    parser.add_argument("--eval_batch_size", type=int, default=2)

    parser.add_argument("--train_batch_size", type=int, default=128)
    parser.add_argument("--train_micro_batch_size_per_gpu", type=int, default=1)

    parser.add_argument("--gpu", type=str, default="0")
    parser.add_argument("--local_rank", type=int, default=0)

    parser = deepspeed.add_config_arguments(parser)
    args = parser.parse_args()

    # GPU device
    os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu
    os.environ["TOKENIZERS_PARALLELISM"] = "false"

    # If your code uses any global variable like `langs_map`, define it here:
    langs_map_m2m = {
        'English': 'en', 'Swahili': 'sw', 'Chinese': 'zh', 'Bengali': 'bn',
        'German': 'de', 'Spanish': 'es', 'French': 'fr', 'Japanese': 'ja',
        'Russian': 'ru', 'Thai': 'th', 'Greek': 'el', 'Telugu': 'te',
        'Arabic': 'ar', 'Bulgarian': 'bg', 'Croatian': 'hr', 'Hungarian': 'hu',
        'Italian': 'it', 'Lithuanian': 'lt', 'Macedonian': 'mk', 'Polish': 'pl',
        'Portuguese': 'pt', 'Albanian': 'sq', 'Serbian': 'sr', 'Turkish': 'tr',
        'Vietnamese': 'vi', 'Hindi': 'hi', 'Flemish': 'nl', 'Urdu': 'ur',
        'Amharic': 'am', 'Ewe': 'ee', 'Hausa': 'ha', 'Igbo': 'ig',
        'Kinyarwanda': 'rw', 'Lingala': 'ln', 'Luganda': 'lg', 'Oromo': 'om',
        'Shona': 'sn', 'Sotho': 'st', 'Wolof': 'wo', 'Twi': 'tw',
        'Xhosa': 'xh', 'Yoruba': 'yo', 'Zulu': 'zu',
        # AmericasNLI
        "Aymara": "ay", "Guarani": "gn", "Quechua": "qu"
    }
    langs_map_nllb = {
        'English': 'eng_Latn', 'Swahili': 'swh_Latn', 'Chinese': 'zho_Hans', 'Bengali': 'ben_Beng',
        'German': 'deu_Latn', 'Spanish': 'spa_Latn', 'French': 'fra_Latn', 'Japanese': 'jpn_Jpan',
        'Russian': 'rus_Cyrl', 'Thai': 'tha_Thai', 'Amharic': 'amh_Ethi', 'Ewe': 'ewe_Latn',
        'Hausa': 'hau_Latn', 'Igbo': 'ibo_Latn', 'Kinyarwanda': 'kin_Latn',
        'Lingala': 'lin_Latn', 'Luganda': 'lug_Latn', 'Oromo': 'orm_Latn',
        'Shona': 'sna_Latn', 'Sotho': 'sot_Latn', 'Twi': 'twi_Latn',
        'Wolof': 'wol_Latn', 'Xhosa': 'xho_Latn', 'Yoruba': 'yor_Latn', 'Zulu': 'zul_Latn', "Telugu": 'tel_Telu',
        # AmericasNLI
        "Aymara": "ayr_Latn",
        "Guarani": "grn_Latn",
        "Quechua": "quy_Latn"
    }
    if "nllb" in args.mt_path:
        langs_map = langs_map_nllb
    else:
        langs_map = langs_map_m2m

    # Expose `langs_map` globally if required by evaluate_* functions
    # e.g.: from __main__ import langs_map
    # Or pass it around as a parameter (already done above).
    globals()['langs_map'] = langs_map

    main(args)
