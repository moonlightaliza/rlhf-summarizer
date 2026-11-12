import os
import json
import math
import torch
import torch.nn.functional as F
from functools import partial
from torch.utils.data import DataLoader

from starter.config import SEED
from starter.models import load_tokenizer, load_causal_lm
from starter.data import load_preference_splits
from starter.batching import collate_pref
from dpo import dpo_margin, dpo_loss

LR = 2e-6
BATCH_SIZE = 8
LOG_EVERY = 250

SFT_PATH = "/content/drive/MyDrive/rlhf-summarizer/checkpoints/sft/final"
SAVE_PATH = "/content/drive/MyDrive/rlhf-summarizer/checkpoints/dpo"
LOG_PATH = "results/dpo_log.jsonl"


@torch.no_grad()
def evaluate(policy, ref, val_loader, device):
    policy.eval()
    total_loss, total_correct, total = 0.0, 0, 0
    for batch in val_loader:
        margin = dpo_margin(policy, ref, batch, device)
        total_loss += -F.logsigmoid(margin).sum().item()
        total_correct += (margin > 0).sum().item()    # implicit reward accuracy
        total += margin.size(0)
    policy.train()
    return total_loss / total, total_correct / total


def main():
    torch.manual_seed(SEED)
    device = "cuda" if torch.cuda.is_available() else "cpu"

    tokenizer = load_tokenizer()
    train_data, val_data = load_preference_splits(tokenizer)
    collate = partial(collate_pref, pad_id=tokenizer.pad_token_id)
    train_loader = DataLoader(train_data, batch_size=BATCH_SIZE, shuffle=True, collate_fn=collate)
    val_loader = DataLoader(val_data, batch_size=BATCH_SIZE, shuffle=False, collate_fn=collate)

    policy = load_causal_lm(SFT_PATH, device).float()
    ref = load_causal_lm(SFT_PATH, device).float()
    ref.eval()
    for p in ref.parameters():
        p.requires_grad = False
    assert all(not p.requires_grad for p in ref.parameters())
    print("Reference model frozen: OK")

    optimizer = torch.optim.AdamW(policy.parameters(), lr=LR)

    os.makedirs("results", exist_ok=True)
    log_file = open(LOG_PATH, "w")
    policy.train()
    running_loss = 0.0

    for step, batch in enumerate(train_loader, start=1):
        margin = dpo_margin(policy, ref, batch, device)
        loss = dpo_loss(margin)

        # Check: policy == reference, so margin = 0 and loss = log 2
        if step == 1:
            print(f"First-step DPO loss: {loss.item():.4f} (expected {math.log(2):.4f})")
            assert abs(loss.item() - math.log(2)) < 1e-3

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        running_loss += loss.item()

        if step % LOG_EVERY == 0:
            train_loss = running_loss / LOG_EVERY
            running_loss = 0.0
            val_loss, val_acc = evaluate(policy, ref, val_loader, device)
            print(f"Step {step} | train loss {train_loss:.4f} | "
                  f"val loss {val_loss:.4f} | val implicit-reward acc {val_acc:.4f}")
            log_file.write(json.dumps({"step": step, "train_loss": train_loss,
                                       "val_loss": val_loss, "val_acc": val_acc}) + "\n")
            log_file.flush()

    os.makedirs(SAVE_PATH, exist_ok=True)
    policy.save_pretrained(SAVE_PATH)
    tokenizer.save_pretrained(SAVE_PATH)
    print("DPO model saved.")


if __name__ == "__main__":
    main()