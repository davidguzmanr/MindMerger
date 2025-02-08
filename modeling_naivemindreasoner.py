# modeling_naivereasoner.py

from transformers import AutoModelForCausalLM, AutoModel
import torch
from torch import nn

# >>> If using Hugging Face PEFT <<<
try:
    from peft import LoraConfig, get_peft_model
    PEFT_AVAILABLE = True
except ImportError:
    PEFT_AVAILABLE = False

class MLP(nn.Module):
    def __init__(self, mt_dim, llm_dim):
        super(MLP, self).__init__()
        self.linear1 = nn.Linear(mt_dim, mt_dim * 2)
        self.linear2 = nn.Linear(mt_dim * 2, llm_dim)
        self.relu = nn.ReLU()

    def forward(self, mt_hidden_state):
        output = self.linear1(mt_hidden_state)
        output = self.relu(output)
        output = self.linear2(output)
        return output

class Mapping(nn.Module):
    def __init__(self, mt_dim, llm_dim):
        super(Mapping, self).__init__()
        self.mlp = MLP(mt_dim, llm_dim)
        self.end_boundary = nn.Parameter(
            torch.zeros(1, 1, llm_dim), requires_grad=True
        )

    def forward(self, hidden_states):
        hidden_states = self.mlp(hidden_states)
        return hidden_states

    def get_embed(self):
        return self.end_boundary


class NaiveMindReasoner(nn.Module):
    """
    Identical to MindReasoner except we do NOT incorporate the additional
    prompt embeddings (soft prompts) in the forward pass. We ignore
    input_ids_prompt/mask_prompt in stage 3.
    """
    def __init__(
        self,
        mt_path,
        llm_path,
        max_gen_len,
        llm_bos_token_id,
        llm_pad_token_id,
        use_lora=False,
        lora_r=16,
        lora_alpha=32,
        lora_dropout=0.05,
        target_modules=["q_proj", "v_proj"]
    ):
        super(NaiveMindReasoner, self).__init__()
        self.max_gen_len = max_gen_len

        # ============= 1) Initialize Multilingual Model =============
        model_mt = AutoModel.from_pretrained(mt_path)
        print('MT model size:', sum(param.numel() for param in model_mt.parameters()) / 1e6)
        self.model_mt = model_mt
        # Freeze all params in the MT model
        for name, parameter in self.model_mt.named_parameters():
            parameter.requires_grad = False

        # For an encoder-decoder model, we only want the encoder
        if 'bert' in mt_path or 'GPT' in mt_path:
            self.encoder_mt = self.model_mt
        else:
            self.encoder_mt = self.model_mt.get_encoder()

        # ============= 2) Initialize LLM =============
        model_llm = AutoModelForCausalLM.from_pretrained(llm_path, trust_remote_code=True, attn_implementation='eager')
        self.model_llm = model_llm
        self.llm_embedding_layer = self.model_llm.get_input_embeddings()
        # Freeze all params in the LLM by default
        for name, parameter in self.model_llm.named_parameters():
            parameter.requires_grad = False

        # ============= 3) Mapping Layer =============
        # Project from MT model dimension -> LLM dimension
        if 'bert' in mt_path:
            d_model = model_mt.config.hidden_size
        elif 'GPT' in mt_path:
            d_model = model_mt.config.n_embd
        else:
            d_model = model_mt.config.d_model
        self.mapping = Mapping(d_model, model_llm.config.hidden_size)

        self.llm_pad_token_id = llm_pad_token_id
        self.llm_bos_token_id = llm_bos_token_id

        print('Mapping layer size:', sum(param.numel() for param in self.mapping.parameters()) / 1e6)

        # ============= 4) Optionally Inject LoRA Into LLM =============
        self.use_lora = use_lora
        if self.use_lora and PEFT_AVAILABLE:
            lora_config = LoraConfig(
                r=lora_r,
                lora_alpha=lora_alpha,
                lora_dropout=lora_dropout,
                bias="none",
                task_type="CAUSAL_LM",
                target_modules=target_modules,
            )
            self.model_llm = get_peft_model(self.model_llm, lora_config)
            print("LoRA is injected into LLM with r={}, alpha={}, dropout={}".format(
                lora_r, lora_alpha, lora_dropout
            ))
        elif self.use_lora and not PEFT_AVAILABLE:
            raise ImportError("peft is not installed. Please install via `pip install peft`.")

    def squeeze_pad(self, hidden_states, masks):
        x_01 = (masks != 0).long()
        seq_num_len = x_01.size(1)
        offset = torch.arange(1, seq_num_len + 1, device=masks.device).unsqueeze(0).expand_as(x_01)
        x_01 *= offset
        _, idx = x_01.sort(1, descending=False)
        masks = masks.gather(1, idx)
        idx = idx.unsqueeze(dim=-1).expand_as(hidden_states)
        hidden_states = hidden_states.gather(1, idx)

        bs, seq_len, dim = hidden_states.size()
        masks_sum = torch.sum(masks, dim=0)
        idx = masks_sum > 0
        idx = idx.unsqueeze(dim=0).expand_as(masks)
        masks = masks[idx]
        idx_ex = idx.unsqueeze(dim=-1).expand_as(hidden_states)
        hidden_states = hidden_states[idx_ex]
        hidden_states = hidden_states.view(bs, -1, dim)
        masks = masks.view(bs, -1)

        return hidden_states, masks, idx

    def forward(
        self,
        input_ids_mt,
        attention_mask_mt,
        labels=None,
        mask_label=None,
        # ignoring input_ids_prompt & mask_prompt in naive version
        input_ids_prompt=None,
        mask_prompt=None
    ):
        """
        Does not incorporate additional prompt embeddings into LLM input.
        """
        # 1) BOS token
        bs = input_ids_mt.size(0)
        bos = torch.full((bs,), self.llm_bos_token_id, dtype=torch.long, device=input_ids_mt.device)
        bos_embedding = self.llm_embedding_layer(bos).view(bs, 1, -1)
        mask = bos_embedding.new_ones(bs, 1, dtype=torch.long)  # shape=[bs,1]

        llm_input_embedding = bos_embedding
        llm_input_mask = mask

        # 2) Encode with MT (frozen)
        with torch.no_grad():
            mt_outputs = self.encoder_mt(
                input_ids=input_ids_mt,
                attention_mask=attention_mask_mt,
                output_hidden_states=True
            )
            enc_hidden = mt_outputs[0]

        # 3) Mapping
        mapped_enc = self.mapping(enc_hidden)

        # 4) End boundary token
        end_boundary = self.mapping.get_embed().expand(bs, 1, -1)
        llm_input_embedding = torch.cat([llm_input_embedding, mapped_enc, end_boundary], dim=1)
        llm_input_mask = torch.cat([llm_input_mask, attention_mask_mt, mask], dim=1)

        # 5) SKIP the prompt concatenation (Naive)
        # We do NOT do: 
        # hidden_states_prompt = self.llm_embedding_layer(input_ids_prompt)
        # llm_input_embedding = torch.cat([...], dim=1)
        # llm_input_mask = ...

        # 6) If we have labels
        if labels is not None:
            pad_labels = llm_input_mask * -100 + (1 - llm_input_mask) * -100
            label_embedding = self.llm_embedding_layer(labels)
            llm_input_embedding = torch.cat([llm_input_embedding, label_embedding], dim=1)
            llm_input_mask = torch.cat([llm_input_mask, mask_label], dim=1)
            labels = labels * mask_label - 100 * (1 - mask_label)
            labels = torch.cat([pad_labels, labels], dim=1)

        # 7) Remove extra padding
        llm_input_embedding, llm_input_mask, cut_pad_idx = self.squeeze_pad(
            llm_input_embedding, llm_input_mask
        )

        # 8) Forward pass
        if labels is None:
            # Inference
            outputs = self.model_llm.generate(
                inputs_embeds=llm_input_embedding,
                attention_mask=llm_input_mask,
                max_new_tokens=self.max_gen_len,
                pad_token_id=self.llm_pad_token_id,
                do_sample=False
            )
            return outputs
        else:
            # Training
            bs, seq_len = labels.size()
            labels = labels[cut_pad_idx].view(bs, -1)
            output = self.model_llm(
                inputs_embeds=llm_input_embedding,
                attention_mask=llm_input_mask,
                labels=labels
            )
            return output.loss
