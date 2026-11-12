
import os
import torch
import torch.nn.functional as F

from torch.utils.data import DataLoader

from starter.models import load_tokenizer
from starter.data import load_preference_splits
from starter.batching import collate_pref

from reward_model import RewardModel

device = "cuda" if torch.cuda.is_available() else "cpu"

sft_path = "/content/drive/MyDrive/rlhf-summarizer/checkpoints/sft/final"

save_path = "/content/drive/MyDrive/rlhf-summarizer/checkpoints/reward"

os.makedirs(save_path, exist_ok=True)

print("Using device:", device)
print("SFT checkpoint:", sft_path)
print("Reward checkpoint directory:", save_path)

if not os.path.isdir(sft_path):
    raise FileNotFoundError(f"SFT checkpoint not found: {sft_path}")




reward_model = RewardModel(sft_path).to(device)

print("Reward model initialized successfully!")



def loss_func(chosen_rewards, rejected_rewards):

    difference = chosen_rewards - rejected_rewards

    return -F.logsigmoid(difference).mean()




tokenizer = load_tokenizer()

train_data, val_data = load_preference_splits(tokenizer)

print("Training pairs:", len(train_data))
print("Validation pairs:", len(val_data))




train_loader = DataLoader(
    train_data,
    batch_size=8,
    shuffle=True,
    collate_fn=lambda batch: collate_pref(
        batch, tokenizer.pad_token_id
    )
)

val_loader = DataLoader(
    val_data,
    batch_size=8,
    shuffle=False,
    collate_fn=lambda batch: collate_pref(
        batch, tokenizer.pad_token_id
    )
)




optimizer = torch.optim.AdamW(
    reward_model.parameters(),
    lr=1e-5
)




reward_model.train()

total_loss = 0

for step, batch in enumerate(train_loader):

    chosen_ids = batch["chosen_ids"].to(device)
    chosen_masks = batch["chosen_mask"].to(device)

    rejected_ids = batch["rejected_ids"].to(device)
    rejected_masks = batch["rejected_mask"].to(device)

    chosen_rewards = reward_model(
        chosen_ids, chosen_masks
    )

    rejected_rewards = reward_model(
        rejected_ids, rejected_masks
    )

    loss = loss_func(
        chosen_rewards,
        rejected_rewards
    )

    optimizer.zero_grad()
    loss.backward()
    optimizer.step()

    total_loss += loss.item()

    if (step + 1) % 250 == 0:

        average_loss = total_loss / (step + 1)

        print(
            f"Step {step + 1} | "
            f"Loss: {average_loss:.4f}"
        )

        checkpoint = {
            "step": step + 1,
            "model_state_dict": reward_model.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
            "loss": loss.item()
        }

        checkpoint_file = os.path.join(
            save_path,
            f"checkpoint_{step + 1}.pt"
        )

        torch.save(checkpoint, checkpoint_file)

        print("Checkpoint saved:", checkpoint_file)


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

        chosen_rewards = reward_model(
            chosen_ids, chosen_mask
        )

        rejected_rewards = reward_model(
            rejected_ids, rejected_mask
        )

        correct += (
            chosen_rewards > rejected_rewards
        ).sum().item()

        total += chosen_rewards.size(0)


accuracy = correct / total if total > 0 else 0

print("\n========== VALIDATION RESULTS ==========")
print(f"Correct predictions: {correct}/{total}")
print(f"Pairwise accuracy: {accuracy:.4f}")
print(f"Validation accuracy: {accuracy * 100:.2f}%")
print("========================================")



final_path = os.path.join(
    save_path,
    "reward_model.pt"
)

torch.save(
    reward_model.state_dict(),
    final_path
)

print("\nFinal reward model saved successfully!")
print("Saved at:", final_path)




print("\nSaved files:")

for filename in os.listdir(save_path):
    print(filename)
