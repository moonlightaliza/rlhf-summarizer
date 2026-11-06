import torch
from common import token_logprobs, make_summary_mask

# Test 1: token_logprobs
batch_size = 2
seq_length = 6
vocab_size = 10

logits = torch.randn(batch_size, seq_length, vocab_size)
input_ids = torch.randint(0, vocab_size, (batch_size, seq_length))

log_probs = token_logprobs(logits, input_ids)

print("Token log probabilities:", log_probs.shape)

assert log_probs.shape == (2, 5)


# Test 2: summary mask
attention_mask = torch.tensor([
    [1, 1, 1, 1, 1, 0],
    [1, 1, 1, 1, 0, 0]
])

prompt_lens = torch.tensor([3, 2])

summary_mask = make_summary_mask(attention_mask, prompt_lens)

print("\nSummary mask:")
print(summary_mask)

assert summary_mask.shape == (2, 5)

print("\nAll tests passed!")