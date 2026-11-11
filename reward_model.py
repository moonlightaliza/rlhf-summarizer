import torch
import torch.nn as nn
from transformers import AutoModelForCausalLM


class RewardModel(nn.Module):
    """SFT backbone (no LM head) + a linear layer that outputs one number."""

    def __init__(self, sft_path):
        super().__init__()
        sft_model = AutoModelForCausalLM.from_pretrained(sft_path)
        self.backbone = sft_model.model              # transformer body only
        self.reward_head = nn.Linear(sft_model.config.hidden_size, 1)

    def forward(self, input_ids, attention_mask):
        hidden = self.backbone(
            input_ids=input_ids, attention_mask=attention_mask
        ).last_hidden_state                          # (B, L, H)

        # Last REAL token of each sequence (works for right-padded input)
        last_idx = attention_mask.sum(dim=1) - 1
        batch_idx = torch.arange(hidden.size(0), device=hidden.device)
        last_hidden = hidden[batch_idx, last_idx]    # (B, H)

        return self.reward_head(last_hidden).squeeze(-1)   # (B,)