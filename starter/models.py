import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from starter.config import MODEL_NAME


def load_tokenizer(name=MODEL_NAME):
    tok = AutoTokenizer.from_pretrained(name)

    if tok.pad_token is None:
        tok.pad_token = tok.eos_token

    return tok


def load_causal_lm(name_or_path=MODEL_NAME, device=None):
    """
    Plain pretrained causal LM (no heads added).
    Use it for the SFT policy, frozen reference model,
    reward model backbone, and PPO value head.
    """
    device = device or (
        "cuda" if torch.cuda.is_available() else "cpu"
    )

    model = AutoModelForCausalLM.from_pretrained(name_or_path)

    return model.to(device)