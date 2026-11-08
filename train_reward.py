import os
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from starter.models import load_tokenizer
from starter.data import load_preference_splits
from starter.batching import collate_pref
from starter.config import *
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

#Define the loss function
def loss_func(chosen_rewards, rejected_rewards):
    difference = chosen_rewards - rejected_rewards
    log_probability = F.logsigmoid(difference)
    loss = -log_probability
    return loss.mean()

tokenizer = load_tokenizer()
train_data, val_data = load_preference_splits(tokenizer)

print("Training pairs:", len(train_data))
print("Validation pairs:", len(val_data))

train_loader = DataLoader(
    train_data,
    batch_size=8,
    shuffle=True,
    collate_fn=lambda batch: collate_pref(batch, tokenizer.pad_token_id)
)

val_loader = DataLoader(
    val_data,
    batch_size=8,
    shuffle=False,
    collate_fn=lambda batch: collate_pref(batch, tokenizer.pad_token_id)
)

#Create optimizer 
optimizer = torch.optim.AdamW(
    params = reward_model.parameters(),
    lr = 1e-5
)
reward_model.train()
total_loss = 0

for step, batch in enumerate(train_loader):
    chosen_ids = batch["chosen_ids"].to(device)
    chosen_masks = batch["chosen_mask"].to(device)
    rejected_ids = batch["rejected_ids"].to(device)
    rejected_masks = batch["rejected_mask"].to(device)
    chosen_rewards = reward_model(chosen_ids, chosen_masks)
    rejected_rewards = reward_model(rejected_ids, rejected_masks)
    loss = loss_func(chosen_rewards, rejected_rewards)
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()
    total_loss += loss.item()
    if (step + 1) % 250 == 0:
        print(
            f"Step {step + 1} | "
            f"Loss: {total_loss / (step + 1):.4f}"
        )

print("Reward model training completed!")

reward_model.eval()
correct = 0
total = 0
with torch.no_grad():
    for batch in val_loader:
        chosen_ids = batch["chosen_ids"].to(device)
        chosen_mask = batch["chosen_mask"].to(device)
        rejected_ids = batch["rejected_ids"].to(device)
        rejected_mask = batch["rejected_mask"].to(device)
        chosen_rewards = reward_model(chosen_ids, chosen_mask)
        rejected_rewards = reward_model(rejected_ids, rejected_mask)
        correct += (chosen_rewards > rejected_rewards).sum().item()
        total += chosen_rewards.size(0)

accuracy = correct / total

print(f"Validation Pairwise Accuracy: {accuracy:.4f}")
print(f"Validation Accuracy: {accuracy * 100:.2f}%")
save_path = "/content/drive/MyDrive/rlhf-summarizer/checkpoints/reward"

os.makedirs(save_path, exist_ok=True)

torch.save(
    reward_model.state_dict(),
    os.path.join(save_path, "reward_model.pt")
)

print("Reward model saved successfully!")







