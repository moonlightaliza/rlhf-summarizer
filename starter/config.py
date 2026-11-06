MODEL_NAME = "HuggingFaceTB/SmolLM2-135M"

# Datasets
SFT_DATA = "CarperAI/openai_summarize_tldr"
PREF_DATA = "CarperAI/openai_summarize_comparisons"

# Token limits
MAX_PROMPT_TOKENS = 448  # Post + "TL;DR:" longer than this is dropped
MAX_SUMMARY_TOKENS = 48   # Summaries longer than this are dropped

# Random seed
SEED = 42

# Dataset sizes
N_SFT_TRAIN, N_SFT_VAL = 20_000, 500
N_PREF_TRAIN, N_PREF_VAL = 20_000, 1_000

# Evaluation
N_EVAL_POSTS = 300