# run_training_afri_naivereasoner.py

import os
import json
import argparse
import ast
import torch
import torch.fx
import deepspeed
from tqdm import tqdm
from transformers import AutoTokenizer
from torch.utils.data import DataLoader, SequentialSampler
from torch.utils.data.distributed import DistributedSampler

# local imports from your project
from mindmerger_tools.utils import set_seed, save_reasoner_model
from mindmerger_tools.read_datasets import *
from mindmerger_tools.input_features import mt_input_features, llm_input_features
from mindmerger_tools.deepspeed_config import get_train_ds_config
from evaluation import evaluate_ppl

# The key difference: we use NaiveMindReasoner, which does NOT incorporate prompt embeddings
from modeling_naivemindreasoner import NaiveMindReasoner


def main(args):
    """
    A training script that can handle:
      - Stage 1: mapping
      - Stage 2: augmentation
      - Stage 3: reasoning (with LoRA) but ignoring soft prompts
    """
    # ---------------------------
    # 1) Basic Setup
    # ---------------------------
    os.makedirs('.', exist_ok=True)
    llm_path = args.llm_path
    mt_path = args.mt_path
    stage_name = args.stage_name  # "mapping" | "augmentation" | "reasoning"
    task = args.task
    augmentation = args.augmentation
    save_name = args.save_name
    train_num = args.train_num

    # Output paths
    result_path_base = f'./results/{save_name}/{task}/{stage_name}/'
    output_model_path_base = f'./outputs/{save_name}/{task}/{stage_name}/'
    os.makedirs(output_model_path_base, exist_ok=True)
    os.makedirs(result_path_base, exist_ok=True)

    # ---------------------------
    # 2) Load Dataset Based on Stage
    # ---------------------------
    if stage_name == 'mapping':
        # Stage 1: typical bilingual data
        if 'math' in task:
            languages = ['Bengali', 'Thai', 'Swahili', 'Japanese', 'Chinese', 'German', 'French', 'Russian',
                         'Spanish']
            train_set = read_lego(train_num, languages)

        elif "nli" in task:
            languages = ["Aymara", "Guarani", "Quechua"]
            train_set = read_americas_lego(train_num, languages)

        elif 'csqa' in task:
            languages = ['Urdu', 'Hindi', 'Swahili', 'Japanese', 'Vietnamese', 'Polish', 'Chinese',
                         'Flemish', 'Russian', 'Italian', 'German', 'Portuguese', 'French', 'Spanish', 'Arabic']
            train_set = read_lego(train_num, languages)

        else:
            languages = ['Swahili', 'Urdu', 'Hindi', 'Thai', 'Arabic', 'Turkish', 'Greek',
                          'Vietnamese', 'Chinese', 'Russian', 'Bulgarian', 'German', 'French', 'Spanish']
            train_set = read_lego(train_num, languages)
        task = 'translation'

    elif stage_name == 'augmentation':
        # Stage 2: query translation or partial prompts
        if 'math' in task:
            train_set = read_math_train(train_num)
        elif "nli" in task:
            train_set = read_americas_xnli_train()
        elif 'csqa' in task:
            train_set = read_x_csqa_train()
        else:
            train_set = read_xnli_train()

    elif stage_name == 'reasoning':
        # Stage 3 but using NaiveMindReasoner => no soft prompts in forward
        print("Stage 3: Reasoning stage WITHOUT soft prompts => NaiveMindReasoner.")
        if 'math' in task:
            train_set = read_math_train(train_num)
        elif "nli" in task:
            train_set = read_americas_xnli_train()
        elif 'csqa' in task:
            train_set = read_x_csqa_train()
        else:
            train_set = read_xnli_train()
    else:
        raise ValueError(f"Unknown stage_name: {stage_name}")

    # Train/Dev split
    dev_set = train_set[:args.dev_size]
    train_set = train_set[args.dev_size:]

    # Wrap in dataset class
    train_set = MathDataset(train_set, task)
    dev_set = MathDataset(dev_set, task)

    # ---------------------------
    # 3) Training Hyperparameters
    # ---------------------------
    lr = args.lr
    epoch_num = args.epoch_num
    max_seq_len = args.max_seq_len
    max_gen_len = args.max_gen_len
    train_batch_size = args.train_batch_size
    eval_batch_size = args.eval_batch_size
    train_micro_batch_size_per_gpu = args.train_micro_batch_size_per_gpu

    gpu_num = torch.cuda.device_count()
    gradient_accumulation = train_batch_size // (train_micro_batch_size_per_gpu * gpu_num)
    assert train_micro_batch_size_per_gpu * gpu_num * gradient_accumulation == train_batch_size

    ds_config = get_train_ds_config(
        train_batch_size,
        train_micro_batch_size_per_gpu,
        lr,
        gradient_accumulation
    )

    # ---------------------------
    # 4) Tokenizers
    # ---------------------------
    tokenizer_m2m = AutoTokenizer.from_pretrained(mt_path)
    tokenizer_llm = AutoTokenizer.from_pretrained(llm_path, use_fast=True)
    tokenizer_llm.pad_token = tokenizer_llm.eos_token
    tokenizer_llm.padding_side = "left"

    print(json.dumps({
        'stage_name': stage_name,
        'task': task,
        'llm_path': llm_path,
        'mt_path': mt_path,
        'lr': lr,
        'epoch_num': epoch_num,
        'gradient_accumulation': gradient_accumulation,
        'train_set_size': len(train_set),
        'dev_set_size': len(dev_set),
        'max_seq_len': max_seq_len,
        'max_gen_len': max_gen_len,
        'train_batch_size': train_batch_size,
        'eval_batch_size': eval_batch_size,
        'output_model_path': output_model_path_base,
    }, indent=2))

    # ---------------------------
    # 5) Initialize NaiveMindReasoner
    # ---------------------------
    use_lora = (stage_name == 'reasoning')  # only use LoRA in stage 3
    if stage_name != 'mapping' and args.init_checkpoint is None:
        # fallback: assume stage 2 checkpoint
        args.init_checkpoint = f'./outputs/{save_name}/{task}/mapping/pytorch_model.bin'

    from modeling_naivemindreasoner import NaiveMindReasoner
    model = NaiveMindReasoner(
        mt_path=mt_path,
        llm_path=llm_path,
        max_gen_len=max_gen_len,
        llm_bos_token_id=tokenizer_llm.bos_token_id,
        llm_pad_token_id=tokenizer_llm.pad_token_id,
        use_lora=use_lora,
        lora_r=args.lora_r,
        lora_alpha=args.lora_alpha,
        lora_dropout=args.lora_dropout,
        target_modules=args.target_modules.split(",")
    )

    # (Re)Load mapping checkpoint if provided
    if args.init_checkpoint is not None:
        checkpoint = torch.load(args.init_checkpoint, map_location='cpu')
        model_dict = checkpoint.get('model_state_dict', checkpoint)
        model.mapping.load_state_dict(model_dict, strict=False)
        print(f"Mapping layer init from: {args.init_checkpoint}")

    # If stage 3, freeze mapping + MT encoder
    if stage_name == 'reasoning':
        for p in model.mapping.parameters():
            p.requires_grad = False
        for p in model.encoder_mt.parameters():
            p.requires_grad = False
        print("Froze the mapping & MT encoder => only LoRA is trainable.")

    # ---------------------------
    # 6) DeepSpeed Initialization
    # ---------------------------
    # This for a problem with mT5 models and DeepSpeed
    for param in model.parameters():
        if not param.data.is_contiguous():
            param.data = param.data.contiguous()

    parameters = filter(lambda p: p.requires_grad, model.parameters())
    model, optimizer, _, __ = deepspeed.initialize(
        config=ds_config,
        model=model,
        model_parameters=parameters,
        training_data=None
    )

    # Data Samplers
    train_sampler = DistributedSampler(train_set)
    dev_sampler = SequentialSampler(dev_set)
    train_dl = DataLoader(train_set, batch_size=train_micro_batch_size_per_gpu, sampler=train_sampler)
    dev_dl = DataLoader(dev_set, batch_size=eval_batch_size, sampler=dev_sampler, shuffle=False)

    # A globally accessible "langs_map" for your custom input features
    # (You presumably define it in your code or pass from arguments)
    global langs_map
    if 'nllb' in mt_path:
        # e.g. define your nllb map
        langs_map = {
            'English': 'eng_Latn','Swahili': 'swh_Latn','Chinese': 'zho_Hans','Bengali': 'ben_Beng',
            'German': 'deu_Latn','Spanish': 'spa_Latn','French': 'fra_Latn','Japanese': 'jpn_Jpan',
            'Russian': 'rus_Cyrl','Thai': 'tha_Thai','Amharic': 'amh_Ethi','Ewe': 'ewe_Latn','Hausa': 'hau_Latn',
            'Igbo': 'ibo_Latn','Kinyarwanda': 'kin_Latn','Lingala': 'lin_Latn','Luganda': 'lug_Latn',
            'Oromo': 'orm_Latn','Shona': 'sna_Latn','Sotho': 'sot_Latn','Twi': 'twi_Latn','Wolof': 'wol_Latn',
            'Xhosa': 'xho_Latn','Yoruba': 'yor_Latn','Zulu': 'zul_Latn',
            # AmericasNLI
            "Aymara": "ayr_Latn", "Guarani": "grn_Latn", "Quechua": "quy_Latn"
        }
    else:
        # define your m2m map
        langs_map = {
            'English':'en','Swahili':'sw','Chinese':'zh','Bengali':'bn','German':'de','Spanish':'es','French':'fr',
            'Japanese':'ja','Russian':'ru','Thai':'th','Greek':'el','Telugu':'te','Arabic':'ar','Bulgarian':'bg',
            'Croatian':'hr','Hungarian':'hu','Italian':'it','Lithuanian':'lt','Macedonian':'mk','Polish':'pl',
            'Portuguese':'pt','Albanian':'sq','Serbian':'sr','Turkish':'tr','Vietnamese':'vi','Hindi':'hi',
            'Flemish':'nl','Urdu':'ur','Amharic':'am','Ewe':'ee','Hausa':'ha','Igbo':'ig','Kinyarwanda':'rw',
            'Lingala':'ln','Luganda':'lg','Oromo':'om','Shona':'sn','Sotho':'st','Wolof':'wo','Twi':'tw',
            'Xhosa':'xh','Yoruba':'yo','Zulu':'zu',
            # AmericasNLI
            "Aymara": "ay", "Guarani": "gn", "Quechua": "qu"
        }

    # Evaluate initial perplexity
    global_rank = torch.distributed.get_rank()
    best_perplexity = evaluate_ppl(
        model, dev_dl, tokenizer_llm, tokenizer_m2m,
        max_seq_len, max_gen_len, langs_map, augmentation
    )
    print(f"Initial perplexity: {best_perplexity:.4f}")

    # ---------------------------
    # 7) Training Loop
    # ---------------------------
    eval_step = 10000
    for epoch in range(epoch_num):
        model.train()
        tr_loss, nb_tr_steps = 0.0, 0
        step_count = 0

        for batch in tqdm(train_dl, desc=f"Epoch {epoch} (NaiveMindReasoner)"):
            sources = batch['source']
            prompts = batch['prompt']  # We'll ignore them in forward pass
            targets = batch['target']
            source_languages = batch['source_language']

            # Prepare input for MT -> LLM
            input_ids_m2m, attention_mask_m2m = mt_input_features(
                sources, tokenizer_m2m, max_seq_len, source_languages, langs_map
            )

            # We add BOS? Typically no, but let's be consistent with your pipeline
            add_bos_token = False
            add_eos_token = True
            labels, mask_label = llm_input_features(
                targets, tokenizer_llm, max_gen_len, add_bos_token, add_eos_token
            )

            # Naive => do not pass prompts to forward
            loss = model(
                input_ids_mt=input_ids_m2m,
                attention_mask_mt=attention_mask_m2m,
                labels=labels,
                mask_label=mask_label,
                input_ids_prompt=None,
                mask_prompt=None
            )
            loss = loss.mean()
            tr_loss += loss.item()
            nb_tr_steps += 1

            # DeepSpeed
            model.backward(loss)
            model.step()

            step_count += 1

            if step_count % eval_step == 0 and step_count > 0:
                perplexity = evaluate_ppl(
                    model, dev_dl, tokenizer_llm, tokenizer_m2m,
                    max_seq_len, max_gen_len, langs_map, augmentation
                )
                print(f"[step={step_count}] ppl={perplexity:.4f}")
                if global_rank == 0 and perplexity < best_perplexity:
                    best_perplexity = perplexity
                    save_reasoner_model(output_model_path_base, model.mapping, is_lora=use_lora, lora_model=model.model_llm)
                    print("==> Saved new best model")

        # End of epoch evaluation
        perplexity = evaluate_ppl(
            model, dev_dl, tokenizer_llm, tokenizer_m2m,
            max_seq_len, max_gen_len, langs_map, augmentation
        )
        print(f"[epoch={epoch}] ppl={perplexity:.4f}")
        if global_rank == 0 and perplexity < best_perplexity:
            best_perplexity = perplexity
            save_reasoner_model(output_model_path_base, model.mapping, is_lora=use_lora, lora_model=model.model_llm)
            print("==> Saved new best model after epoch")

    print(f"Training completed. Best perplexity: {best_perplexity:.4f}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    # Typical arguments
    parser.add_argument("--llm_path", type=str, default="google/gemma-2-9b-it")
    parser.add_argument("--mt_path", type=str, default="facebook/nllb-200-distilled-600M")
    parser.add_argument("--save_name", type=str, default="NaiveMindReasoner")
    parser.add_argument("--task", type=str, default="math")
    parser.add_argument("--stage_name", type=str, default="reasoning")
    parser.add_argument("--lr", type=float, default=2e-5)
    parser.add_argument("--epoch_num", type=int, default=1)
    parser.add_argument("--train_num", type=int, default=3000)
    parser.add_argument("--train_batch_size", type=int, default=24)
    parser.add_argument("--train_micro_batch_size_per_gpu", type=int, default=1)
    parser.add_argument("--eval_batch_size", type=int, default=2)
    parser.add_argument("--max_seq_len", type=int, default=512)
    parser.add_argument("--max_gen_len", type=int, default=512)
    parser.add_argument("--dev_size", type=int, default=300)
    parser.add_argument("--init_checkpoint", type=str, default=None)
    parser.add_argument("--gpu", type=str, default="0")
    parser.add_argument("--local_rank", type=int, default=0)
    parser.add_argument("--augmentation", type=ast.literal_eval, default=False)

    # LoRA arguments
    parser.add_argument("--lora_r", type=int, default=16)
    parser.add_argument("--lora_alpha", type=int, default=32)
    parser.add_argument("--lora_dropout", type=float, default=0.05)
    parser.add_argument("--target_modules", type=str, default="q_proj,v_proj")

    parser = deepspeed.add_config_arguments(parser)
    args = parser.parse_args()

    # GPU setup
    os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu
    os.environ["TOKENIZERS_PARALLELISM"] = "false"
    set_seed(0)

    # Provide or load your language maps here if needed
    # e.g.:
    # global langs_map
    # langs_map = {...}  # or define inside main

    main(args)
