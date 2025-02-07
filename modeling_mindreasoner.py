from transformers import AutoModelForCausalLM, AutoModel
import torch
from torch import nn

# >>> Add these imports if using Hugging Face PEFT <<<
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

class MindReasoner(nn.Module):
    def __init__(self, mt_path, llm_path, max_gen_len, llm_bos_token_id,
                 llm_pad_token_id, use_lora=False, lora_r=16, lora_alpha=32, lora_dropout=0.05):
        super(MindReasoner, self).__init__()
        self.max_gen_len = max_gen_len

        # ============= 1) Initialize Multilingual Model =============
        model_mt = AutoModel.from_pretrained(mt_path)
        print('MT model size:', sum(param.numel() for param in model_mt.parameters()) / 1e6)
        self.model_mt = model_mt
        for name, parameter in self.model_mt.named_parameters():
            parameter.requires_grad = False
        if 'bert' in mt_path or 'GPT' in mt_path:
            self.encoder_mt = self.model_mt
        else:
            self.encoder_mt = self.model_mt.get_encoder()

        # ============= 2) Initialize LLM =============
        model_llm = AutoModelForCausalLM.from_pretrained(llm_path, trust_remote_code=True)
        self.model_llm = model_llm
        self.llm_embedding_layer = self.model_llm.get_input_embeddings()
        for name, parameter in self.model_llm.named_parameters():
            parameter.requires_grad = False  # default frozen

        # ============= 3) Initialize Mapping Layer =============
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
        # We'll do so only if "use_lora=True" or in stage3 scenario
        self.use_lora = use_lora
        if self.use_lora and PEFT_AVAILABLE:
            # Define LoRA config
            lora_config = LoraConfig(
                r=lora_r,
                lora_alpha=lora_alpha,
                lora_dropout=lora_dropout,
                bias="none",
                task_type="CAUSAL_LM",
                target_modules=["q_proj", "v_proj"]
            )
            self.model_llm = get_peft_model(self.model_llm, lora_config)
            # Now the LoRA layers inside model_llm will be trainable
            print("LoRA is injected into LLM with r={}, alpha={}, dropout={}".format(
                lora_r, lora_alpha, lora_dropout
            ))
        elif self.use_lora and not PEFT_AVAILABLE:
            raise ImportError("peft is not installed. Please install via `pip install peft`.")

    def squeeze_pad(self, hidden_states, masks):
        # same as your original code ...
        x_01 = (masks != 0).long()
        seq_num_len = x_01.size(1)
        offset = torch.tensor([(i + 1) for i in range(seq_num_len)], dtype=torch.long).to(x_01.device)
        offset = offset.unsqueeze(dim=0).expand_as(x_01)
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

    def forward(self, input_ids_mt, attention_mask_mt,
                labels=None, mask_label=None, input_ids_prompt=None, mask_prompt=None):
        """
        If self.use_lora==True, the LoRA modules inside self.model_llm will be trainable.
        The rest remains the same pipeline (mapping + encoder).
        """
        # 1) Soft "Boundary" token
        end_boundary = self.mapping.get_embed()
        bs = input_ids_mt.size(0)
        end_boundary = end_boundary.expand([bs, 1, end_boundary.size(-1)])

        # 2) BOS token
        bos = torch.tensor([self.llm_bos_token_id for i in range(bs)], dtype=torch.long).cuda()
        bos_embedding = self.llm_embedding_layer(bos)
        bos_embedding = bos_embedding.view(bs, 1, -1)
        mask = torch.ones([bs, 1], dtype=torch.long).cuda()
        llm_input_embedding = bos_embedding
        llm_input_mask = mask

        # 3) Multilingual encoder
        with torch.no_grad():
            mt_encoder_outputs = self.encoder_mt(input_ids=input_ids_mt,
                                                 attention_mask=attention_mask_mt,
                                                 output_hidden_states=True)
            encoder_last_hidden_state = mt_encoder_outputs[0]
        # 4) Mapping layer (either trainable or frozen, depending on your logic/stage)
        mt_hidden_state = self.mapping(encoder_last_hidden_state)

        # 5) Combine into LLM input
        llm_input_embedding = torch.cat([llm_input_embedding, mt_hidden_state, end_boundary], dim=1)
        llm_input_mask = torch.cat([llm_input_mask, attention_mask_mt, mask], dim=1)

        # 6) If we have an additional prompt
        if input_ids_prompt is not None:
            hidden_states_prompt = self.llm_embedding_layer(input_ids_prompt)
            llm_input_embedding = torch.cat([llm_input_embedding, hidden_states_prompt], dim=1)
            llm_input_mask = torch.cat([llm_input_mask, mask_prompt], dim=1)

        # 7) If we are training with labels
        if labels is not None:
            pad_labels = llm_input_mask * -100 + (1 - llm_input_mask) * -100
            label_embedding = self.llm_embedding_layer(labels)
            llm_input_embedding = torch.cat([llm_input_embedding, label_embedding], dim=1)
            llm_input_mask = torch.cat([llm_input_mask, mask_label], dim=1)
            labels = labels * mask_label - 100 * (1 - mask_label)
            labels = torch.cat([pad_labels, labels], dim=1)

        # 8) Remove extra padding
        llm_input_embedding, llm_input_mask, cut_pad_idx = \
            self.squeeze_pad(llm_input_embedding, llm_input_mask)

        # 9) Inference or training
        if labels is None:
            # generation mode
            generate_ids = self.model_llm.generate(
                inputs_embeds=llm_input_embedding,
                attention_mask=llm_input_mask,
                max_new_tokens=self.max_gen_len,
                pad_token_id=self.llm_pad_token_id,
                do_sample=False
            )
            return generate_ids
        else:
            # training mode
            bs, seq_len = labels.size()
            labels = labels[cut_pad_idx]
            labels = labels.view(bs, -1)
            output = self.model_llm(
                inputs_embeds=llm_input_embedding,
                attention_mask=llm_input_mask,
                labels=labels
            )
            return output.loss
