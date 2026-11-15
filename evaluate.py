%%writefile evaluate.py
import os
import json
import torch
from rouge_score import rouge_scorer

from starter.models import load_tokenizer, load_causal_lm
from starter.data import load_eval_posts
from starter.batching import collate_prompts, collate_sft
from common import token_logprobs, make_summary_mask
from reward_model import RewardModel

MAX_NEW_TOKENS = 48
GEN_BATCH = 16
SCORE_BATCH = 4

BASE = "/content/drive/MyDrive/rlhf-summarizer"
MODEL_PATHS = {
    "sft": f"{BASE}/checkpoints/sft/final",
    "ppo": f"{BASE}/checkpoints/ppo",
    "dpo": f"{BASE}/checkpoints/dpo",
}
REWARD_PATH = f"{BASE}/checkpoints/reward/reward_model.pt"
OUT_DIR = f"{BASE}/results"


@torch.no_grad()
def generate_summaries(model, tok, posts, device):
    """Greedy summaries as token-id lists, each cut at its first EOS (EOS kept)."""
    pad, eos = tok.pad_token_id, tok.eos_token_id
    summaries = []
    for start in range(0, len(posts), GEN_BATCH):
        batch = collate_prompts(posts[start:start + GEN_BATCH], pad)    # left-padded
        ids = batch["input_ids"].to(device)
        mask = batch["attention_mask"].to(device)
        out = model.generate(input_ids=ids, attention_mask=mask, do_sample=False,
                             max_new_tokens=MAX_NEW_TOKENS,
                             pad_token_id=pad, eos_token_id=eos)[:, ids.size(1):].tolist()
        for s in out:
            if eos in s:
                s = s[: s.index(eos) + 1]
            summaries.append(s)
    return summaries


def build_batch(posts, summaries, tok, device):
    """Right-padded prompt + summary (same format as SFT)."""
    items = [{"prompt_ids": p["prompt_ids"], "summary_ids": s}
             for p, s in zip(posts, summaries)]
    b = collate_sft(items, tok.pad_token_id)
    return (b["input_ids"].to(device), b["attention_mask"].to(device),
            b["prompt_lens"].to(device))


@torch.no_grad()
def reward_scores(reward_model, posts, summaries, tok, device):
    scores = []
    for start in range(0, len(posts), SCORE_BATCH):
        sl = slice(start, start + SCORE_BATCH)
        ids, mask, _ = build_batch(posts[sl], summaries[sl], tok, device)
        scores += reward_model(ids, mask).tolist()
    return scores


@torch.no_grad()
def kl_to_sft(model, sft, posts, summaries, tok, device):
    """Per post: sum over summary tokens of log pi(y_t) - log pi_sft(y_t), on the model's own summary."""
    kls = []
    for start in range(0, len(posts), SCORE_BATCH):
        sl = slice(start, start + SCORE_BATCH)
        ids, mask, plens = build_batch(posts[sl], summaries[sl], tok, device)
        lp = token_logprobs(model(input_ids=ids, attention_mask=mask).logits, ids)
        lp_ref = token_logprobs(sft(input_ids=ids, attention_mask=mask).logits, ids)
        m = make_summary_mask(mask, plens).float()
        kls += ((lp - lp_ref) * m).sum(dim=1).tolist()
    return kls


def mean(xs):
    return sum(xs) / len(xs)


def main():
    device = "cuda" if torch.cuda.is_available() else "cpu"
    tok = load_tokenizer()
    posts = load_eval_posts(tok)
    print("Evaluation posts:", len(posts))

    models = {name: load_causal_lm(path, device).float().eval()
              for name, path in MODEL_PATHS.items()}

    reward_model = RewardModel(MODEL_PATHS["sft"]).to(device).float()
    state = torch.load(REWARD_PATH, map_location=device)
    if "model_state_dict" in state:
        state = state["model_state_dict"]
    reward_model.load_state_dict(state)
    reward_model.eval()
    for p in reward_model.parameters():
        p.requires_grad = False

    scorer = rouge_scorer.RougeScorer(["rougeL"], use_stemmer=True)
    references = [p["summary"].strip() for p in posts]

    summaries, scores, kls, lengths, rouges, texts = {}, {}, {}, {}, {}, {}
    for name, model in models.items():
        print(f"Generating with {name} ...")
        summaries[name] = generate_summaries(model, tok, posts, device)
        texts[name] = [tok.decode(s, skip_special_tokens=True).strip() for s in summaries[name]]
        scores[name] = reward_scores(reward_model, posts, summaries[name], tok, device)
        kls[name] = ([0.0] * len(posts) if name == "sft"
                     else kl_to_sft(model, models["sft"], posts, summaries[name], tok, device))
        lengths[name] = [len(s) for s in summaries[name]]        # tokens, EOS included
        rouges[name] = [scorer.score(ref, hyp)["rougeL"].fmeasure
                        for ref, hyp in zip(references, texts[name])]

    # ---- Table 3 ----
    table = {}
    for name in models:
        wins = None
        if name != "sft":
            wins = mean([float(a > b) for a, b in zip(scores[name], scores["sft"])])
        table[name] = {"mean_reward": mean(scores[name]), "win_rate_vs_sft": wins,
                       "kl_to_sft": mean(kls[name]), "length": mean(lengths[name]),
                       "rougeL": mean(rouges[name])}

    os.makedirs(OUT_DIR, exist_ok=True)
    with open(f"{OUT_DIR}/table3.json", "w") as f:
        json.dump(table, f, indent=2)

    with open(f"{OUT_DIR}/eval_summaries.jsonl", "w") as f:
        for i, post in enumerate(posts):
            row = {"index": i, "prompt": post["prompt"], "reference": references[i]}
            for name in models:
                row[name] = texts[name][i]
                row[f"{name}_reward"] = scores[name][i]
            f.write(json.dumps(row) + "\n")

    print("\n| Policy | Mean r | Win rate vs SFT | KL to SFT | Length | ROUGE-L |")
    print("|---|---|---|---|---|---|")
    for name, r in table.items():
        win = "-" if r["win_rate_vs_sft"] is None else f"{r['win_rate_vs_sft']:.3f}"
        print(f"| {name.upper()} | {r['mean_reward']:.3f} | {win} | "
              f"{r['kl_to_sft']:.2f} | {r['length']:.1f} | {r['rougeL']:.4f} |")


if __name__ == "__main__":
    main()