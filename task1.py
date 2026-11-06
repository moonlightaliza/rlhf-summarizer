from starter.models import load_tokenizer
from starter.data import load_sft_splits, load_preference_splits, load_eval_posts

tok = load_tokenizer()

# Load datasets
sft_train, sft_val = load_sft_splits(tok)
pref_train, pref_val = load_preference_splits(tok)
eval_posts = load_eval_posts(tok)

print("Dataset sizes")
print("-----------------------")
print("SFT train:", len(sft_train))
print("SFT validation:", len(sft_val))
print("Preference train:", len(pref_train))
print("Preference validation:", len(pref_val))
print("Evaluation posts:", len(eval_posts))


# Calculate token length statistics
prompt_lengths = [
    len(item["prompt_ids"]) for item in sft_train
]

summary_lengths = [
    len(item["summary_ids"]) for item in sft_train
]

print("\nToken Length Statistics")
print("-----------------------")

print("Prompt mean:", sum(prompt_lengths) / len(prompt_lengths))
print("Prompt max:", max(prompt_lengths))

print("Summary mean:", sum(summary_lengths) / len(summary_lengths))
print("Summary max:", max(summary_lengths))