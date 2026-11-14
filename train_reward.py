import os
import json
import torch
import torch.nn.functional as F
from functools import partial
from torch.utils.data import DataLoader

from starter.config import SEED
from starter.models import load_tokenizer
from starter.data import load_preference_splits
from starter.batching import collate_pref
from reward_model import RewardModel

# ---------------- settings (from the assignment) ----------------
LR = 1e-5
BATCH_SIZE = 8           # pairs per optimizer step
MICRO_BATCH = 2          # pairs per forward pass; 4 of these make one batch of 8
PRINT_EVERY = 50         # log training loss / accuracy
EVAL_EVERY = 250         # validate + save checkpoint

# ---------------- paths (change these if you run elsewhere) ----------------
ROOT = "/kaggle/working/rlhf-summarizer"
SFT_PATH = "/kaggle/input/datasets/mobeenrukhsar/sft-checkpoints/final"
SAVE_DIR = f"{ROOT}/checkpoints/reward"
CKPT_PATH = f"{SAVE_DIR}/checkpoint_latest.pt"     # one file, overwritten each time
FINAL_PATH = f"{SAVE_DIR}/reward_model.pt"         # final model (what PPO loads)
LOG_PATH = f"{ROOT}/results/reward_log.jsonl"


def train_step(model, optimizer, batch, device):
    """One optimizer step on a batch of 8 pairs, done as several micro-batches."""
    B = batch["prompt_lens"].size(0)
    optimizer.zero_grad(set_to_none=True)
    total_loss, n_correct = 0.0, 0

    for start in range(0, B, MICRO_BATCH):
        sub = {k: v[start:start + MICRO_BATCH].to(device) for k, v in batch.items()}
        chosen = model(sub["chosen_ids"], sub["chosen_mask"])
        rejected = model(sub["rejected_ids"], sub["rejected_mask"])
        diff = chosen - rejected

        # Bradley-Terry loss: -log sigma(r_w - r_l), divided by the FULL batch size
        loss = -F.logsigmoid(diff).sum() / B
        loss.backward()                              # gradients add up across micro-batches

        total_loss += loss.item()
        n_correct += (diff > 0).sum().item()

    torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
    optimizer.step()
    return total_loss, n_correct                     # loss = mean over the full batch


@torch.no_grad()
def evaluate(model, val_loader, device):
    """Validation loss and pairwise accuracy: fraction where r(chosen) > r(rejected)."""
    model.eval()
    total_loss, correct, total = 0.0, 0, 0
    for batch in val_loader:
        chosen = model(batch["chosen_ids"].to(device), batch["chosen_mask"].to(device))
        rejected = model(batch["rejected_ids"].to(device), batch["rejected_mask"].to(device))
        diff = chosen - rejected
        total_loss += -F.logsigmoid(diff).sum().item()
        correct += (diff > 0).sum().item()
        total += diff.size(0)
    model.train()
    return total_loss / total, correct / total


def save_checkpoint(model, optimizer, step):
    os.makedirs(SAVE_DIR, exist_ok=True)
    tmp = CKPT_PATH + ".tmp"
    torch.save({"step": step,
                "model_state_dict": model.state_dict(),
                "optimizer_state_dict": optimizer.state_dict()}, tmp)
    os.replace(tmp, CKPT_PATH)       # a half-written file never replaces a good one


def main():
    torch.manual_seed(SEED)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print("Device:", device)

    if not os.path.isdir(SFT_PATH):
        raise FileNotFoundError(f"SFT checkpoint not found: {SFT_PATH}")

    model = RewardModel(SFT_PATH).to(device)
    print("Backbone dtype:", next(model.backbone.parameters()).dtype)      # should be float32
    print("Reward head dtype:", model.reward_head.weight.dtype)

    optimizer = torch.optim.AdamW(model.parameters(), lr=LR)

    # ---- resume? ----
    start_step = 0
    if os.path.exists(CKPT_PATH):
        state = torch.load(CKPT_PATH, map_location=device)
        model.load_state_dict(state["model_state_dict"])
        optimizer.load_state_dict(state["optimizer_state_dict"])
        start_step = state["step"]
        print(f"Resuming from step {start_step}")

    tokenizer = load_tokenizer()
    train_data, val_data = load_preference_splits(tokenizer)
    print("Training pairs:", len(train_data), "| Validation pairs:", len(val_data))

    # Fixed shuffle (same every run), so a resumed run skips the batches already used
    order = torch.randperm(len(train_data), generator=torch.Generator().manual_seed(SEED)).tolist()
    train_data = [train_data[i] for i in order][start_step * BATCH_SIZE:]

    collate = partial(collate_pref, pad_id=tokenizer.pad_token_id)
    train_loader = DataLoader(train_data, batch_size=BATCH_SIZE, shuffle=False, collate_fn=collate)
    val_loader = DataLoader(val_data, batch_size=4, shuffle=False, collate_fn=collate)

    os.makedirs(os.path.dirname(LOG_PATH), exist_ok=True)
    log_file = open(LOG_PATH, "a" if start_step > 0 else "w")

    model.train()
    run_loss, run_correct, run_total = 0.0, 0, 0

    for step, batch in enumerate(train_loader, start=start_step + 1):
        loss_value, n_correct = train_step(model, optimizer, batch, device)
        run_loss += loss_value
        run_correct += n_correct
        run_total += batch["prompt_lens"].size(0)

        if step == 1:
            # The head is randomly initialised, so rewards are nearly equal: loss ~ 0.69
            print(f"First-step loss: {loss_value:.4f} (expected about 0.69)")

        if step % PRINT_EVERY == 0:
            train_loss = run_loss / PRINT_EVERY
            train_acc = run_correct / run_total
            run_loss, run_correct, run_total = 0.0, 0, 0
            print(f"Step {step} | train loss {train_loss:.4f} | train acc {train_acc:.4f}")
            log_file.write(json.dumps({"step": step, "train_loss": train_loss,
                                       "train_acc": train_acc}) + "\n")
            log_file.flush()

        if step % EVAL_EVERY == 0:
            val_loss, val_acc = evaluate(model, val_loader, device)
            print(f"Step {step} | VALIDATION loss {val_loss:.4f} | acc {val_acc * 100:.2f}%")
            log_file.write(json.dumps({"step": step, "val_loss": val_loss,
                                       "val_acc": val_acc}) + "\n")
            log_file.flush()
            save_checkpoint(model, optimizer, step)
            print(f"Checkpoint saved at step {step}")

    # ---- final evaluation + save ----
    val_loss, val_acc = evaluate(model, val_loader, device)
    print(f"\nFINAL validation loss {val_loss:.4f} | pairwise accuracy {val_acc * 100:.2f}%")
    log_file.write(json.dumps({"final": True, "val_loss": val_loss, "val_acc": val_acc}) + "\n")
    log_file.close()

    os.makedirs(SAVE_DIR, exist_ok=True)
    torch.save(model.state_dict(), FINAL_PATH)
    print("Final reward model saved:", FINAL_PATH)


if __name__ == "__main__":
    main()