import torch
import torch.nn as nn
from transformers import AutoModelForCausalLM


class RewardModel(nn.Module):
    """SFT backbone (no LM head) + a linear layer that outputs one number."""

    def __init__(self, sft_path):
        super().__init__()

        sft_model = AutoModelForCausalLM.from_pretrained(sft_path)

        # The SFT checkpoint is stored in half precision, so force everything
        # to fp32. Training in fp16 with AdamW is unreliable at lr 1e-5.
        self.backbone = sft_model.model.float()          # transformer body only
        self.reward_head = nn.Linear(sft_model.config.hidden_size, 1)

        del sft_model                                    # drop the unused LM head

    def forward(self, input_ids, attention_mask):
        hidden = self.backbone(
            input_ids=input_ids,
            attention_mask=attention_mask,
            use_cache=False,
        ).last_hidden_state                              # (B, L, H)

        # Last REAL token of each sequence (inputs must be right-padded)
        last_idx = attention_mask.sum(dim=1) - 1
        batch_idx = torch.arange(hidden.size(0), device=hidden.device)
        last_hidden = hidden[batch_idx, last_idx]        # (B, H)

        return self.reward_head(last_hidden).squeeze(-1)  # (B,)