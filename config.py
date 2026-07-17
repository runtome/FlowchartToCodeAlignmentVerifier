"""Central configuration for the Flowchart/Pseudocode -> Java alignment verifier.

Everything that changes between "run on my laptop with a tiny sample" and
"run on Kaggle T4x2 with the real competition data" lives here, so the rest of
the code never hard-codes a path or a hyper-parameter.
"""
from __future__ import annotations

import os
from pathlib import Path

# --------------------------------------------------------------------------- #
# Paths
# --------------------------------------------------------------------------- #
# On Kaggle the competition data is mounted read-only under /kaggle/input/<slug>.
# Locally, point DATA_DIR at a folder with the same layout for testing.
#
#   DATA_DIR/
#     train/case_01/{flowchart.png, pseudo.txt, solution.java}
#     test/case_34/{flowchart.png, pseudo.txt, solution.java}   (or all under one dir)
#     alignment_score_training.csv
#     id_definition.csv
#     sample_submission.csv
#
# The competition sometimes ships train and test cases under a single `cases/`
# directory instead of split train/test folders; CASE_SEARCH_DIRS lists every
# place we will look for a `case_xx` folder.
DATA_DIR = Path(os.environ.get("ALIGN_DATA_DIR", "/kaggle/input/flowchart-to-code"))

TRAIN_CSV = DATA_DIR / "alignment_score_training.csv"
ID_DEFINITION_CSV = DATA_DIR / "id_definition.csv"
SAMPLE_SUBMISSION_CSV = DATA_DIR / "sample_submission.csv"

# Directories that may contain `case_xx/` folders. First match wins per case.
CASE_SEARCH_DIRS = [
    DATA_DIR / "train",
    DATA_DIR / "test",
    DATA_DIR / "cases",
    DATA_DIR,
]

# Per-case file names inside a case folder.
FLOWCHART_NAME = "flowchart.png"
PSEUDO_NAME = "pseudo.txt"          # brief says pseudo.txt; some kits use pseudocode.txt
PSEUDO_NAME_ALT = "pseudocode.txt"
JAVA_NAME = "solution.java"

# Where to write the submission (Kaggle expects it in the working dir).
OUTPUT_DIR = Path(os.environ.get("ALIGN_OUTPUT_DIR", "/kaggle/working"))
SUBMISSION_PATH = OUTPUT_DIR / "submission.csv"

# --------------------------------------------------------------------------- #
# Model
# --------------------------------------------------------------------------- #
# Attach the model as a Kaggle *Model* or *Dataset* input and point MODEL_DIR at
# the mounted snapshot folder (the one containing config.json). Because the
# competition notebook has no internet, we must load from a local path.
MODEL_DIR = os.environ.get(
    "ALIGN_MODEL_DIR",
    "/kaggle/input/qwen2.5-vl-7b-instruct/transformers/default/1",
)

# T4 = Turing: no bfloat16, no FlashAttention-2.
TORCH_DTYPE = "float16"
ATTN_IMPLEMENTATION = "sdpa"        # "eager" if sdpa misbehaves on old transformers
LOAD_IN_4BIT = False                # flip to True (bitsandbytes) if fp16 OOMs with images
# Leave head-room on each 16 GB T4 for the KV-cache + image tokens.
MAX_MEMORY = {0: "14GiB", 1: "14GiB"}

# Qwen-VL visual token budget (multiples of 28*28 per patch). Caps OOM risk.
MIN_PIXELS = 256 * 28 * 28
MAX_PIXELS = 1280 * 28 * 28

# --------------------------------------------------------------------------- #
# Decoding / self-consistency
# --------------------------------------------------------------------------- #
MAX_NEW_TOKENS = 1024

# Primary deterministic pass.
GREEDY = dict(do_sample=False, temperature=None, top_p=None)

# Self-consistency sampling pass (metric is exact accuracy -> majority vote helps).
N_SAMPLES = 5                       # extra sampled votes on top of the greedy pass
SAMPLE_TEMPERATURE = 0.7
SAMPLE_TOP_P = 0.9

# --------------------------------------------------------------------------- #
# Pipeline toggles
# --------------------------------------------------------------------------- #
USE_OCR = True          # augment the flowchart image with detected text
TWO_PASS = False        # transcribe flowchart -> text, then score as text
USE_FEWSHOT = True      # prepend labeled anchors from the train CSV
N_FEWSHOT = 3           # anchors per prompt (kept small for the T4 token budget)
USE_PERSONAS = False    # add strict/lenient persona votes to the ensemble

VALID_SCORES = (0, 1, 2, 3)
DEFAULT_FALLBACK_SCORE = 1  # used only if every parse + retry fails

# --------------------------------------------------------------------------- #
# Offline env — import this module early so these are set before HF loads.
# --------------------------------------------------------------------------- #
os.environ.setdefault("HF_HUB_OFFLINE", "1")
os.environ.setdefault("TRANSFORMERS_OFFLINE", "1")
os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")
