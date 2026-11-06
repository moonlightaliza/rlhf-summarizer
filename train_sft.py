import os
import torch
import torch.nn.functional as F

from torch.utils.data import DataLoader
from functools import partial

from starter.config import (
    MODEL_NAME,
    SEED
)

from starter.models import (
    load_tokenizer,
    load_causal_lm
)

from starter.data import load_sft_splits

from starter.batching import collate_sft

from common import token_logprobs, make_summary_mask

# Set random seed
torch.manual_seed(SEED)

# Select device
device = "cuda" if torch.cuda.is_available() else "cpu"

print("Using device:", device)


# Load tokenizer
tokenizer = load_tokenizer()

# Load base language model
model = load_causal_lm(device=device)

print("Model loaded:", MODEL_NAME)