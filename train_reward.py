import torch
import torch.nn as nn
from transformers import AutoModelForCausalLM


class RewardModel(nn.Module):

    def __init__(self, sft_path):
        super().__init__()
        sft_model = AutoModelForCausalLM.from_pretrained(sft_path)
        self.backbone = sft_model.model
        hidden_size = sft_model.config.hidden_size
        del sft_model.lm_head

        
        self.reward_head = nn.Linear(hidden_size, 1)

    def forward(self, input_ids, attention_mask):

        
        outputs = self.backbone(
            input_ids=input_ids,
            attention_mask=attention_mask
        )

        hidden_states = outputs.last_hidden_state

        # Find last non-padding token for each sequence
        last_token_indices = attention_mask.sum(dim=1) - 1

        # Extract its hidden state
        batch_indices = torch.arange(
            hidden_states.size(0),
            device=hidden_states.device
        )

        last_hidden = hidden_states[
            batch_indices,
            last_token_indices
        ]

   
        rewards = self.reward_head(last_hidden)

        return rewards.squeeze(-1)

device = "cuda" if torch.cuda.is_available() else "cpu"

sft_path = "/content/drive/MyDrive/rlhf-summarizer/checkpoints/sft/final"

reward_model = RewardModel(sft_path).to(device)