import torch
import torch.nn.functional as F


def token_logprobs(logits, input_ids):
    """
    Calculate log probabilities of the actual next tokens.

    logits:   (batch_size, seq_length, vocab_size)
    input_ids: (batch_size, seq_length)

    Returns:
        (batch_size, seq_length - 1)
    """

    # Shift logits to predict the next token
    shift_logits = logits[:, :-1, :].contiguous()

    # Shift labels to contain the actual next tokens
    shift_labels = input_ids[:, 1:].contiguous()

    # Convert logits into log probabilities
    log_probs = F.log_softmax(shift_logits, dim=-1)

    # Select the log probability of each actual token
    token_log_probs = log_probs.gather(
        dim=-1,
        index=shift_labels.unsqueeze(-1)
    ).squeeze(-1)

    return token_log_probs

def make_summary_mask(attention_mask, prompt_lens):
    """
    Select summary tokens only.

    attention_mask: (B, L)
    prompt_lens: (B,)

    Returns:
        (B, L-1)
    """

    batch_size, seq_length = attention_mask.shape

    # Positions corresponding to next-token predictions
    positions = torch.arange(
        seq_length - 1,
        device=attention_mask.device
    )

    # Summary starts immediately after the prompt
    mask = positions.unsqueeze(0) >= (
        prompt_lens.unsqueeze(1) - 1
    )

    # Exclude padding tokens
    mask = mask & attention_mask[:, 1:].bool()

    return mask