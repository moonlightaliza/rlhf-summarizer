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

SFT_PATH = "/content/drive/MyDrive/rlhf-summarizer/checkpoints/sft/final"
REWARD_PATH = "/content/drive/MyDrive/rlhf-summarizer/checkpoints/reward/reward_model.pt"
SAVE_PATH = "/content/drive/MyDrive/rlhf-summarizer/checkpoints/ppo"
LOG_PATH = "results/ppo_log.jsonl"


def main():
    torch.manual_seed(SEED)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    tok = load_tokenizer()

    # Prompts only: 200 iterations x 16 prompts = 3200
    sft_train, _ = load_sft_splits(tok)
    prompts = random.Random(SEED).sample(sft_train, ITERATIONS * PROMPTS_PER_ROLLOUT)

    policy = PolicyWithValue(SFT_PATH, device)               # trainable
    ref = load_causal_lm(SFT_PATH, device)                   # frozen
    reward_model = RewardModel(SFT_PATH).to(device)          # frozen
    reward_model.load_state_dict(torch.load(REWARD_PATH, map_location=device))
    freeze(ref)
    freeze(reward_model)
    print("Reference and reward model frozen: OK")

    optimizer = torch.optim.AdamW(policy.parameters(), lr=LR)
    os.makedirs("results", exist_ok=True)
    log_file = open(LOG_PATH, "w")

    for it in range(1, ITERATIONS + 1):
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

    os.makedirs(SAVE_PATH, exist_ok=True)
    policy.lm.save_pretrained(SAVE_PATH)                     # without the value head
    tok.save_pretrained(SAVE_PATH)
    print("PPO policy saved.")


if __name__ == "__main__":
    main()