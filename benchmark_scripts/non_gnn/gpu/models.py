"""Native Transformers models with Model Zoo's preprocessed-batch loss contract.

No Model Zoo Trainer, cstorch.compile, or Model Zoo attention kernels are used.
The summed losses let the caller normalize over the entire accumulated batch.
"""

import torch.nn.functional as F
from torch import nn
from transformers import BertConfig, BertForPreTraining, LlamaConfig, LlamaForCausalLM


def bert_loss_sum(mlm_logits, nsp_logits, batch, mlm_weight):
    losses = F.cross_entropy(
        mlm_logits.float().flatten(0, 1),
        batch["labels"].long().flatten(),
        reduction="none",
    )
    loss = (losses * batch["masked_lm_mask"].flatten()).sum() * mlm_weight
    return loss + F.cross_entropy(
        nsp_logits.float(),
        batch["next_sentence_label"].long().flatten(),
        reduction="sum",
    )


def llama_loss_sum(logits, batch):
    # Model Zoo HDF5 labels are ALREADY shifted; HF's labels= would shift twice.
    # attention_mask is a LOSS mask here, not a Transformer attention mask.
    losses = F.cross_entropy(
        logits.float().flatten(0, 1), batch["labels"].long().flatten(), reduction="none"
    )
    return (losses * batch["attention_mask"].flatten()).sum()


class BertTrainingModel(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        config = BertConfig(
            vocab_size=cfg["vocab_size"],
            hidden_size=cfg["hidden_size"],
            num_hidden_layers=cfg["num_hidden_layers"],
            num_attention_heads=cfg["num_heads"],
            intermediate_size=cfg["filter_size"],
            hidden_act=cfg["encoder_nonlinearity"],
            hidden_dropout_prob=cfg["dropout_rate"],
            attention_probs_dropout_prob=cfg["attention_dropout_rate"],
            max_position_embeddings=cfg["max_position_embeddings"],
            layer_norm_eps=cfg["layer_norm_epsilon"],
            initializer_range=cfg.get("initializer_range", 0.02),
            pad_token_id=None,
        )
        config._attn_implementation = "sdpa"
        self.model = BertForPreTraining(config)
        self.mlm_weight = cfg["mlm_loss_weight"]
        # Match Model Zoo BERT's truncated-normal initialization distribution.
        for module in self.model.modules():
            if isinstance(module, (nn.Linear, nn.Embedding)):
                nn.init.trunc_normal_(
                    module.weight,
                    std=config.initializer_range,
                    a=-2 * config.initializer_range,
                    b=2 * config.initializer_range,
                )

    def forward(self, batch):
        output = self.model.bert(
            input_ids=batch["input_ids"].long(),
            attention_mask=batch["attention_mask"],
            token_type_ids=batch["token_type_ids"].long(),
            return_dict=True,
        )
        hidden = output.last_hidden_state
        positions = (
            batch["masked_lm_positions"]
            .long()
            .unsqueeze(-1)
            .expand(-1, -1, hidden.shape[-1])
        )
        # Apply the vocabulary head only at masked positions, as on CSX.
        selected = hidden.gather(1, positions)
        mlm_logits = self.model.cls.predictions(selected)
        nsp_logits = self.model.cls.seq_relationship(output.pooler_output)
        return bert_loss_sum(mlm_logits, nsp_logits, batch, self.mlm_weight)


class LlamaTrainingModel(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        scaling = {
            "rope_type": "llama3",
            "factor": cfg["pos_scaling_factor"],
            **cfg["pos_scaling_extra_args"],
        }
        config = LlamaConfig(
            vocab_size=cfg["vocab_size"],
            hidden_size=cfg["hidden_size"],
            intermediate_size=cfg["filter_size"],
            num_hidden_layers=cfg["num_hidden_layers"],
            num_attention_heads=cfg["num_heads"],
            num_key_value_heads=cfg["extra_attention_params"]["num_kv_groups"],
            hidden_act="silu",
            max_position_embeddings=cfg["max_position_embeddings"],
            rms_norm_eps=cfg["layer_norm_epsilon"],
            rope_theta=cfg["rope_theta"],
            rope_scaling=scaling,
            attention_dropout=cfg["attention_dropout_rate"],
            attention_bias=False,
            mlp_bias=False,
            tie_word_embeddings=cfg["share_embedding_weights"],
            initializer_range=cfg["initializer_range"],
            use_cache=False,
        )
        config._attn_implementation = "sdpa"
        self.model = LlamaForCausalLM(config)

    def forward(self, batch):
        logits = self.model(input_ids=batch["input_ids"].long(), use_cache=False).logits
        return llama_loss_sum(logits, batch)


def make_model(cfg):
    if cfg["name"] == "bert":
        return BertTrainingModel(cfg)
    if cfg["name"] == "llama":
        return LlamaTrainingModel(cfg)
    raise ValueError(f"Unsupported native model: {cfg['name']}")
