import os
import json
import random
import torch

from starter.config import SEED
from starter.models import load_tokenizer, load_causal_lm
from starter.data import load_sft_splits
from reward_model import RewardModel
from ppo import PolicyWithValue, freeze, collect_rollout, ppo_epoch

LR = 1e-5
PPO_EPOCHS = 4
PROMPTS_PER_ROLLOUT = 16
ITERATIONS = 200
CKPT_EVERY = 10          # save a resume checkpoint every 10 iterations

BASE = "/content/drive/MyDrive/rlhf-summarizer"
SFT_PATH = f"{BASE}/checkpoints/sft/final"
REWARD_PATH = f"{BASE}/checkpoints/reward/reward_model.pt"
SAVE_PATH = f"{BASE}/checkpoints/ppo"                  # final policy
CKPT_PATH = f"{BASE}/checkpoints/ppo_ckpt/state.pt"    # resume checkpoint (overwritten)
LOG_PATH = f"{BASE}/results/ppo_log.jsonl"


def save_checkpoint(policy, optimizer, iteration):
    os.makedirs(os.path.dirname(CKPT_PATH), exist_ok=True)
    tmp = CKPT_PATH + ".tmp"
    torch.save({"iteration": iteration,
                "policy": policy.state_dict(),
                "optimizer": optimizer.state_dict()}, tmp)
    os.replace(tmp, CKPT_PATH)      # a half-written file never replaces a good one


def main():
    torch.manual_seed(SEED)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    tok = load_tokenizer()

    # Prompts only: 200 iterations x 16 prompts = 3200
    sft_train, _ = load_sft_splits(tok)
    prompts = random.Random(SEED).sample(sft_train, ITERATIONS * PROMPTS_PER_ROLLOUT)

    policy = PolicyWithValue(SFT_PATH, device)                       # trainable (fp32)
    ref = load_causal_lm(SFT_PATH, device).float()                   # frozen
    reward_model = RewardModel(SFT_PATH).to(device).float()          # frozen

    state = torch.load(REWARD_PATH, map_location=device)
    if "model_state_dict" in state:                                  # a checkpoint_*.pt file
        state = state["model_state_dict"]
    reward_model.load_state_dict(state)

    freeze(ref)
    freeze(reward_model)
    print("Reference and reward model frozen: OK")

    optimizer = torch.optim.AdamW(policy.parameters(), lr=LR)

    # ---- resume? ----
    start_it = 0
    if os.path.exists(CKPT_PATH):
        ckpt = torch.load(CKPT_PATH, map_location=device)
        policy.load_state_dict(ckpt["policy"])
        optimizer.load_state_dict(ckpt["optimizer"])
        start_it = ckpt["iteration"]
        print(f"Resuming after iteration {start_it}")

    os.makedirs(os.path.dirname(LOG_PATH), exist_ok=True)
    log_file = open(LOG_PATH, "a" if start_it > 0 else "w")

    for it in range(start_it + 1, ITERATIONS + 1):
        items = prompts[(it - 1) * PROMPTS_PER_ROLLOUT: it * PROMPTS_PER_ROLLOUT]
        rollout = collect_rollout(policy, ref, reward_model, items, tok, device)
        epoch_stats = [ppo_epoch(policy, optimizer, rollout) for _ in range(PPO_EPOCHS)]

        # Correctness checks
        first = epoch_stats[0]      # first epoch: policy == old policy, so ratio = 1
        if first["max_ratio_dev"] > 1e-3 or first["clipped"] > 0:
            print(f"WARNING it {it}: epoch-1 ratio not 1 "
                  f"(max dev {first['max_ratio_dev']:.2e}, clip {first['clipped']:.3f})")
        if it == 1:
            print(f"Iteration 1 KL to reference: {rollout['mean_kl']:.6f} (expected 0)")

        avg = lambda key: sum(e[key] for e in epoch_stats) / PPO_EPOCHS
        log = {
            "iteration": it,
            "mean_score": rollout["mean_score"],
            "mean_kl": rollout["mean_kl"],
            "mean_length": rollout["mean_length"],
            "clip_fraction": avg("clipped"),
            "policy_loss": avg("policy_loss"),
            "value_loss": avg("value_loss"),
            "entropy": avg("entropy"),
        }
        log_file.write(json.dumps(log) + "\n")
        log_file.flush()
        print(f"It {it:3d} | score {log['mean_score']:.3f} | KL {log['mean_kl']:.3f} | "
              f"len {log['mean_length']:.1f} | clip {log['clip_fraction']:.3f} | "
              f"pol {log['policy_loss']:.3f} | val {log['value_loss']:.3f} | ent {log['entropy']:.3f}")

        if it % CKPT_EVERY == 0:
            save_checkpoint(policy, optimizer, it)
            print(f"Checkpoint saved at iteration {it}")

    os.makedirs(SAVE_PATH, exist_ok=True)
    policy.lm.save_pretrained(SAVE_PATH)                     # without the value head
    tok.save_pretrained(SAVE_PATH)
    print("PPO policy saved.")


if __name__ == "__main__":
    main()