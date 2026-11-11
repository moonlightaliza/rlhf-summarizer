import torch
import torch.nn as nn
import torch.nn.functional as F

from starter.models import load_causal_lm
from starter.batching import collate_sft, collate_prompts
from common import token_logprobs, make_summary_mask

# Algorithm settings (from the assignment)
CLIP_EPS = 0.2
BETA_KL = 0.05
C_VALUE = 0.1
C_ENTROPY = 0.01
MAX_NEW_TOKENS = 48
N_CHUNKS = 2    # process each rollout in halves to save GPU memory
# gamma = 1.0, so "return" = sum of all future rewards (reverse cumulative sum)


class PolicyWithValue(nn.Module):
    """SFT language model + a linear value head on the last hidden layer."""

    def __init__(self, sft_path, device):
        super().__init__()
        self.lm = load_causal_lm(sft_path, device)
        self.value_head = nn.Linear(self.lm.config.hidden_size, 1).to(device)

    def forward(self, input_ids, attention_mask):
        hidden = self.lm.model(input_ids=input_ids,
                               attention_mask=attention_mask).last_hidden_state
        logits = self.lm.lm_head(hidden)                    # (B, L, V)
        values = self.value_head(hidden).squeeze(-1)        # (B, L)
        return logits, values


def freeze(model):
    model.eval()
    for p in model.parameters():
        p.requires_grad = False
    assert all(not p.requires_grad for p in model.parameters())


# ---------- 4.1 Sampling ----------
@torch.no_grad()
def generate_summaries(policy, items, tok, device):
    """Sample one summary per prompt. Returns RIGHT-padded prompt+summary batch."""
    pad_id, eos_id = tok.pad_token_id, tok.eos_token_id

    # generate() needs LEFT-padded prompts
    prompts = collate_prompts(items, pad_id)
    prompt_ids = prompts["input_ids"].to(device)
    prompt_mask = prompts["attention_mask"].to(device)

    policy.eval()
    generated = policy.lm.generate(
        input_ids=prompt_ids, attention_mask=prompt_mask,
        do_sample=True, temperature=1.0, top_k=0, top_p=1.0,
        max_new_tokens=MAX_NEW_TOKENS,
        pad_token_id=pad_id, eos_token_id=eos_id,
    )[:, prompt_ids.size(1):].tolist()
    policy.train()

    # Cut each summary at its first EOS (keep the EOS), ignore what follows
    batch_items = []
    for item, summary in zip(items, generated):
        if eos_id in summary:
            summary = summary[: summary.index(eos_id) + 1]
        batch_items.append({"prompt_ids": item["prompt_ids"], "summary_ids": summary})

    # Rebuild RIGHT-padded sequences (same format as SFT). The reward model and
    # make_summary_mask both expect right padding.
    batch = collate_sft(batch_items, pad_id)
    return (batch["input_ids"].to(device),
            batch["attention_mask"].to(device),
            batch["prompt_lens"].to(device))


# ---------- 4.3 Per-token rewards ----------
def compute_rewards(old_logp, ref_logp, scores, mask, summary_mask):
    """r_t = -beta * (log pi_old - log pi_ref) at every token, + RM score at last token."""
    m = summary_mask.float()
    kl_per_token = (old_logp - ref_logp) * m
    rewards = -BETA_KL * kl_per_token

    B = old_logp.size(0)
    last_token = mask.sum(dim=1) - 2     # shifted index of the final real token
    rewards[torch.arange(B, device=old_logp.device), last_token] += scores
    return rewards, kl_per_token


# ---------- 4.4 Returns and advantages ----------
def compute_advantages(rewards, old_values, summary_mask):
    """R_t = sum of future rewards (gamma = 1), A_t = R_t - V(s_t), then normalise."""
    m = summary_mask.float()
    returns = rewards.flip(1).cumsum(1).flip(1)
    advantages = returns - old_values

    tokens = advantages[summary_mask]                       # generated tokens only
    advantages = (advantages - tokens.mean()) / (tokens.std() + 1e-8)
    return returns, advantages * m                          # zero outside generated tokens


# ---------- 4.1 - 4.4 together ----------
@torch.no_grad()
def collect_rollout(policy, ref, reward_model, items, tok, device):
    ids, mask, prompt_lens = generate_summaries(policy, items, tok, device)

    # 4.2 Store old log-probs, reference log-probs, values, reward-model scores
    logits, values = policy(ids, mask)
    old_logp = token_logprobs(logits, ids)                  # (B, L-1)
    old_values = values[:, :-1]                             # V(s_t): position just before a_t
    ref_logp = token_logprobs(ref(input_ids=ids, attention_mask=mask).logits, ids)
    scores = reward_model(ids, mask)                        # (B,)
    summary_mask = make_summary_mask(mask, prompt_lens)     # True on generated tokens

    rewards, kl_per_token = compute_rewards(old_logp, ref_logp, scores, mask, summary_mask)
    returns, advantages = compute_advantages(rewards, old_values, summary_mask)
    m = summary_mask.float()

    return {
        "ids": ids, "mask": mask, "summary_mask": m,
        "old_logp": old_logp, "returns": returns, "advantages": advantages,
        # for logging
        "mean_score": scores.mean().item(),
        "mean_kl": kl_per_token.sum(dim=1).mean().item(),
        "mean_length": m.sum(dim=1).mean().item(),
    }


# ---------- 4.5 PPO update ----------
def ppo_epoch(policy, optimizer, r):
    """One pass over the whole rollout (in chunks). Every loss is a mean over generated tokens."""
    n_tokens = r["summary_mask"].sum()
    B = r["ids"].size(0)
    chunk = B // N_CHUNKS

    stats = {"policy_loss": 0.0, "value_loss": 0.0, "entropy": 0.0,
             "clipped": 0.0, "max_ratio_dev": 0.0}
    optimizer.zero_grad()

    for start in range(0, B, chunk):
        s = slice(start, start + chunk)
        ids, mask, m = r["ids"][s], r["mask"][s], r["summary_mask"][s]

        logits, values = policy(ids, mask)
        log_probs_all = F.log_softmax(logits[:, :-1], dim=-1)               # (b, L-1, V)
        new_logp = log_probs_all.gather(-1, ids[:, 1:].unsqueeze(-1)).squeeze(-1)
        entropy = -(log_probs_all.exp() * log_probs_all).sum(dim=-1)        # (b, L-1)

        ratio = torch.exp(new_logp - r["old_logp"][s])                      # r_t(theta)
        adv = r["advantages"][s]
        unclipped = ratio * adv
        clipped = torch.clamp(ratio, 1 - CLIP_EPS, 1 + CLIP_EPS) * adv

        policy_loss = -(torch.min(unclipped, clipped) * m).sum() / n_tokens
        value_loss = (((values[:, :-1] - r["returns"][s]) ** 2) * m).sum() / n_tokens
        entropy_mean = (entropy * m).sum() / n_tokens

        loss = policy_loss + C_VALUE * value_loss - C_ENTROPY * entropy_mean
        loss.backward()                                                      # gradients add up over chunks

        stats["policy_loss"] += policy_loss.item()
        stats["value_loss"] += value_loss.item()
        stats["entropy"] += entropy_mean.item()
        stats["clipped"] += (((ratio - 1).abs() > CLIP_EPS).float() * m).sum().item() / n_tokens.item()
        stats["max_ratio_dev"] = max(stats["max_ratio_dev"], ((ratio - 1).abs() * m).max().item())

    torch.nn.utils.clip_grad_norm_(policy.parameters(), 1.0)
    optimizer.step()
    return stats