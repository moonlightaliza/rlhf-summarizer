
import torch
import torch.nn as nn
from transformers import AutoModelForCausalLM


class RewardModel(nn.Module):

    def __init__(self, sft_path):
        super().__init__()

        sft_model = AutoModelForCausalLM.from_pretrained(
            sft_path,
            torch_dtype=torch.float16
        )

        self.backbone = sft_model.model
        hidden_size = sft_model.config.hidden_size

        # Reward head remains float32
        self.reward_head = nn.Linear(hidden_size, 1)

        del sft_model

    def forward(self, input_ids, attention_mask):

        outputs = self.backbone(
            input_ids=input_ids,
            attention_mask=attention_mask,
            use_cache=False
        )

        hidden_states = outputs.last_hidden_state

        # Last real token (right padding)
        last_indices = attention_mask.sum(dim=1) - 1

        batch_indices = torch.arange(
            hidden_states.size(0),
            device=hidden_states.device
        )

        last_hidden = hidden_states[
            batch_indices, last_indices
        ]

        # Match reward head precision
        last_hidden = last_hidden.to(
            self.reward_head.weight.dtype
        )

        return self.reward_head(last_hidden).squeeze(-1)
