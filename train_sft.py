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

# Load SFT training and validation datasets
train_data, val_data = load_sft_splits(tokenizer)

print("Training examples:", len(train_data))
print("Validation examples:", len(val_data))



BATCH_SIZE = 16


# Create training DataLoader
train_loader = DataLoader(
    train_data,
    batch_size=BATCH_SIZE,
    shuffle=True,
    collate_fn=partial(
        collate_sft,
        pad_id=tokenizer.pad_token_id
    )
)


# Create validation DataLoader
val_loader = DataLoader(
    val_data,
    batch_size=BATCH_SIZE,
    shuffle=False,
    collate_fn=partial(
        collate_sft,
        pad_id=tokenizer.pad_token_id
    )
)


print("Training batches:", len(train_loader))
print("Validation batches:", len(val_loader))
batch = next(iter(train_loader))

print("\nBatch shapes:")
print("Input IDs:", batch["input_ids"].shape)
print("Attention mask:", batch["attention_mask"].shape)
print("Prompt lengths:", batch["prompt_lens"].shape)

def sft_loss(model, batch, device):
    """
    Compute the SFT loss for a batch.
    """
    input_ids = batch["input_ids"].to(device)
    attention_mask = batch["attention_mask"].to(device)
    prompt_lens = batch["prompt_lens"].to(device)

    # Forward pass
    outputs = model(
        input_ids=input_ids,
        attention_mask=attention_mask,
    )

    # Get logits
    logits = outputs.logits

    # Compute token log probabilities
    log_probs = token_logprobs(logits, input_ids)

    # Create summary mask
    summary_mask = make_summary_mask(attention_mask, prompt_lens)

    loss = -log_probs*summary_mask.float()
    loss = loss.sum() / summary_mask.sum().clamp(min=1.0)

    return loss

learning_rate = 5e-5
optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate)
@torch.no_grad()
def evaluate_loss(model, val_loader, device):

    model.eval()

    total_loss = 0.0

    for batch in val_loader:

        loss = sft_loss(model, batch, device)

        total_loss += loss.item()

    avg_loss = total_loss / len(val_loader)

    model.train()
    return avg_loss

def train_sft(model, train_loader, val_loader, optimizer, device):

    model.train()
    total_loss = 0.0

    for step, batch in enumerate(train_loader, start=1):
        optimizer.zero_grad()
        loss = sft_loss(model, batch, device)
        loss.backward()

        torch.nn.utils.clip_grad_norm_(
            model.parameters(),
            max_norm=1.0
        )

        optimizer.step()

        total_loss += loss.item()

        if step % 250 == 0:

            avg_train_loss = total_loss / 250

            val_loss = evaluate_loss(
                model,
                val_loader,
                device
            )

            print(
                f"Step {step} | "
                f"Train Loss: {avg_train_loss:.4f} | "
                f"Validation Loss: {val_loss:.4f}"
            )

            total_loss = 0.0

train_sft(
    model=model,
    train_loader=train_loader,
    val_loader=val_loader,
    optimizer=optimizer,
    device=device
)