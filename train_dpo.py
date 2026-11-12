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
from dpo import dpo_margin

LR = 2e-6
BATCH_SIZE = 8
MICRO_BATCH = 2          # pairs per forward pass; 4 of these make one batch of 8
LOG_EVERY = 250          # evaluate + save a checkpoint this often

BASE = "/content/drive/MyDrive/rlhf-summarizer/checkpoints"
SFT_PATH = f"{BASE}/sft/final"
SAVE_PATH = f"{BASE}/dpo"               # final model
CKPT_DIR = f"{BASE}/dpo_ckpt"           # resume checkpoint (overwritten each time)
STATE_PATH = f"{CKPT_DIR}/training_state.pt"
LOG_PATH = "/content/drive/MyDrive/rlhf-summarizer/results/dpo_log.jsonl"


def train_step(policy, ref, optimizer, batch, device):
    """One optimizer step on a batch of 8 pairs, done as several micro-batches."""
    B = batch["prompt_lens"].size(0)
    optimizer.zero_grad()
    total_loss = 0.0
    for start in range(0, B, MICRO_BATCH):
        sub = {k: v[start:start + MICRO_BATCH] for k, v in batch.items()}
        margin = dpo_margin(policy, ref, sub, device)
        loss = -F.logsigmoid(margin).sum() / B      # divide by the FULL batch size
        loss.backward()                              # gradients add up across micro-batches
        total_loss += loss.item()
    optimizer.step()
    return total_loss                                # = mean DPO loss of the full batch


def save_checkpoint(policy, optimizer, step):
    os.makedirs(CKPT_DIR, exist_ok=True)
    policy.save_pretrained(CKPT_DIR)                                   # weights
    torch.save({"step": step, "optimizer": optimizer.state_dict()},    # saved last
               STATE_PATH)


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

    # ---- resume? ----
    start_step, state = 0, None
    policy_source = SFT_PATH
    if os.path.exists(STATE_PATH):
        state = torch.load(STATE_PATH, map_location=device)
        start_step = state["step"]
        policy_source = CKPT_DIR
        print(f"Resuming from step {start_step}")

    tokenizer = load_tokenizer()
    train_data, val_data = load_preference_splits(tokenizer)

    # Fixed shuffle (same every run), so a resumed run skips the batches already used
    order = torch.randperm(len(train_data), generator=torch.Generator().manual_seed(SEED)).tolist()
    train_data = [train_data[i] for i in order][start_step * BATCH_SIZE:]

    collate = partial(collate_pref, pad_id=tokenizer.pad_token_id)
    train_loader = DataLoader(train_data, batch_size=BATCH_SIZE, shuffle=False, collate_fn=collate)
    val_loader = DataLoader(val_data, batch_size=4, shuffle=False, collate_fn=collate)

    policy = load_causal_lm(policy_source, device).float()   # trainable
    ref = load_causal_lm(SFT_PATH, device).float()           # frozen copy of SFT
    ref.eval()
    for p in ref.parameters():
        p.requires_grad = False
    assert all(not p.requires_grad for p in ref.parameters())
    print("Reference model frozen: OK")

    optimizer = torch.optim.AdamW(policy.parameters(), lr=LR)
    if state is not None:
        optimizer.load_state_dict(state["optimizer"])

    os.makedirs(os.path.dirname(LOG_PATH), exist_ok=True)
    log_file = open(LOG_PATH, "a" if start_step > 0 else "w")
    policy.train()
    running_loss = 0.0

    for step, batch in enumerate(train_loader, start=start_step + 1):
        loss_value = train_step(policy, ref, optimizer, batch, device)

        # Check (fresh run only): policy == reference, so margin = 0 and loss = log 2
        if step == 1:
            print(f"First-step DPO loss: {loss_value:.4f} (expected {math.log(2):.4f})")
            assert abs(loss_value - math.log(2)) < 1e-3

        running_loss += loss_value

        if step % LOG_EVERY == 0:
            train_loss = running_loss / LOG_EVERY
            running_loss = 0.0
            val_loss, val_acc = evaluate(policy, ref, val_loader, device)
            print(f"Step {step} | train loss {train_loss:.4f} | "
                  f"val loss {val_loss:.4f} | val implicit-reward acc {val_acc:.4f}")
            log_file.write(json.dumps({"step": step, "train_loss": train_loss,
                                       "val_loss": val_loss, "val_acc": val_acc}) + "\n")
            log_file.flush()
            save_checkpoint(policy, optimizer, step)       # right after logging
            print(f"Checkpoint saved at step {step}")

    os.makedirs(SAVE_PATH, exist_ok=True)
    policy.save_pretrained(SAVE_PATH)
    tokenizer.save_pretrained(SAVE_PATH)
    print("DPO model saved.")


if __name__ == "__main__":
    main()