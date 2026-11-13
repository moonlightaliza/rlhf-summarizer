
import os
import torch
import torch.nn.functional as F

from torch.utils.data import DataLoader

from starter.models import load_tokenizer
from starter.data import load_preference_splits
from starter.batching import collate_pref

from reward_model import RewardModel




device = "cuda" if torch.cuda.is_available() else "cpu"

ROOT = "/kaggle/working/rlhf-summarizer"

sft_path = os.path.join(
    ROOT, "checkpoints", "sft", "final"
)

save_path = os.path.join(
    ROOT, "checkpoints", "reward"
)

os.makedirs(save_path, exist_ok=True)

BATCH_SIZE = 8
LEARNING_RATE = 1e-5

print("Device:", device)
print("SFT path:", sft_path)
print("Reward save path:", save_path)




if not os.path.isdir(sft_path):
    raise FileNotFoundError(
        f"SFT checkpoint not found: {sft_path}"
    )



reward_model = RewardModel(sft_path).to(device)

print("Backbone dtype:",
      next(reward_model.backbone.parameters()).dtype)

print("Reward head dtype:",
      reward_model.reward_head.weight.dtype)



def loss_func(chosen_rewards, rejected_rewards):

    difference = chosen_rewards - rejected_rewards

    return -F.logsigmoid(difference).mean()



tokenizer = load_tokenizer()

train_data, val_data = load_preference_splits(tokenizer)

print("Training pairs:", len(train_data))
print("Validation pairs:", len(val_data))




train_loader = DataLoader(
    train_data,
    batch_size=BATCH_SIZE,
    shuffle=True,
    collate_fn=lambda batch: collate_pref(
        batch, tokenizer.pad_token_id
    )
)

val_loader = DataLoader(
    val_data,
    batch_size=BATCH_SIZE,
    shuffle=False,
    collate_fn=lambda batch: collate_pref(
        batch, tokenizer.pad_token_id
    )
)



optimizer = torch.optim.AdamW(
    reward_model.parameters(),
    lr=LEARNING_RATE
)


def evaluate():

    reward_model.eval()

    total_loss = 0.0
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

            loss = loss_func(
                chosen_rewards, rejected_rewards
            )

            total_loss += loss.item() * chosen_ids.size(0)

            correct += (
                chosen_rewards > rejected_rewards
            ).sum().item()

            total += chosen_ids.size(0)

    reward_model.train()

    return (
        total_loss / total,
        correct / total
    )



reward_model.train()

total_loss = 0.0
correct = 0
total = 0

for step, batch in enumerate(train_loader):

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

    difference = chosen_rewards - rejected_rewards

    loss = loss_func(
        chosen_rewards, rejected_rewards
    )

    optimizer.zero_grad()

    loss.backward()

    torch.nn.utils.clip_grad_norm_(
        reward_model.parameters(),
        max_norm=1.0
    )

    optimizer.step()

    total_loss += loss.item()

    correct += (difference > 0).sum().item()
    total += chosen_ids.size(0)

    if (step + 1) % 50 == 0:

        print(
            f"Step {step + 1} | "
            f"Train Loss: {total_loss / 50:.6f} | "
            f"Train Acc: {correct / total:.4f}"
        )

        total_loss = 0.0
        correct = 0
        total = 0

    # Save checkpoint every 250 steps
    if (step + 1) % 250 == 0:

        checkpoint = {
            "step": step + 1,
            "model_state_dict": reward_model.state_dict(),
            "optimizer_state_dict": optimizer.state_dict()
        }

        checkpoint_file = os.path.join(
            save_path,
            f"checkpoint_{step + 1}.pt"
        )

        torch.save(checkpoint, checkpoint_file)

        print("Checkpoint saved:", checkpoint_file)

        val_loss, val_acc = evaluate()

        print(
            f"Validation Loss: {val_loss:.6f} | "
            f"Validation Accuracy: {val_acc * 100:.2f}%"
        )


final_path = os.path.join(
    save_path, "reward_model.pt"
)

torch.save(
    reward_model.state_dict(),
    final_path
)

print("\nTraining completed!")
print("Final reward model saved:", final_path)

print("\nSaved checkpoint files:")

for filename in os.listdir(save_path):
    print(filename)
