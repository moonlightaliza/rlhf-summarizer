import torch
import torch.nn.functional as F
from common import token_logprobs, make_summary_mask

BETA = 0.1


def sequence_logprob(model, ids, mask, prompt_lens):
    """log pi(y | x) = sum of log-probs of the SUMMARY tokens only. Shape (B,)."""
    logits = model(input_ids=ids, attention_mask=mask).logits
    token_lp = token_logprobs(logits, ids)                  # (B, L-1)
    summary_mask = make_summary_mask(mask, prompt_lens)     # (B, L-1)
    return (token_lp * summary_mask).sum(dim=1)


def dpo_margin(policy, ref, batch, device):
    """The expression inside sigma(...) in the DPO loss. Shape (B,)."""
    prompt_lens = batch["prompt_lens"].to(device)
    c_ids, c_mask = batch["chosen_ids"].to(device), batch["chosen_mask"].to(device)
    r_ids, r_mask = batch["rejected_ids"].to(device), batch["rejected_mask"].to(device)

    # trainable policy (with gradients)
    policy_chosen = sequence_logprob(policy, c_ids, c_mask, prompt_lens)
    policy_rejected = sequence_logprob(policy, r_ids, r_mask, prompt_lens)

    # frozen reference (no gradients)
    with torch.no_grad():
        ref_chosen = sequence_logprob(ref, c_ids, c_mask, prompt_lens)
        ref_rejected = sequence_logprob(ref, r_ids, r_mask, prompt_lens)

    chosen_ratio = policy_chosen - ref_chosen          # log(pi / pi_ref) for y_w
    rejected_ratio = policy_rejected - ref_rejected    # log(pi / pi_ref) for y_l
    return BETA * (chosen_ratio - rejected_ratio)


def dpo_loss(margin):
    return -F.logsigmoid(margin).mean()