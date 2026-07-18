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
# place we will look for a `case_xx` folder (search is recursive, so extra
# nesting like training/training/ or test/test/ is handled automatically).
# Default: the `datasets/` folder inside the cloned repo. Override with
# ALIGN_DATA_DIR to point at a Kaggle dataset mount.
DATA_DIR = Path(os.environ.get("ALIGN_DATA_DIR", "datasets"))

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
# With internet ON (default), MODEL_DIR is a Hugging Face repo id and the weights
# download on first use. For an offline run, set OFFLINE=1 and point MODEL_DIR at
# a local snapshot folder (the one containing config.json).
#
# Model options (the loader auto-detects the class):
#   "Qwen/Qwen3-VL-8B-Instruct"    <- default; better OCR/reasoning, splits over 2x T4
#   "Qwen/Qwen3-VL-4B-Instruct"    <- lighter, comfortable on a single T4
#   "Qwen/Qwen2.5-VL-7B-Instruct"  <- previous model
# Qwen3-VL needs a recent transformers (>= 4.57): pip install -U transformers.
MODEL_DIR = os.environ.get("ALIGN_MODEL_DIR", "Qwen/Qwen3-VL-8B-Instruct")

# Internet on by default. Set ALIGN_OFFLINE=1 to force HF offline mode.
OFFLINE = os.environ.get("ALIGN_OFFLINE", "0") == "1"

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
N_SAMPLES = 5                       # extra sampled votes (non-ensemble / single-baseline path)
SAMPLE_TEMPERATURE = 0.7
SAMPLE_TOP_P = 0.9

# --------------------------------------------------------------------------- #
# Ensemble (Baseline A + Baseline B + structural check)
# --------------------------------------------------------------------------- #
# Baseline A = direct image judge; Baseline B = image->Mermaid->judge (two-stage).
# For a flowchart case the final score is a majority vote over A, B, and the
# rule-based structural estimate. For a pseudocode case there is no image, so A
# and B collapse into one text judge (no Mermaid stage).
USE_BASELINE_A = True        # direct image (flowchart) / text (pseudocode) judge
USE_BASELINE_B = True        # two-stage: transcribe flowchart to Mermaid, then judge
USE_STRUCTURAL_VOTE = False  # OFF: the rule estimate maps to 0/3 and skews the vote to extremes
SAMPLES_PER_BASELINE = 2     # sampled votes per baseline on top of its greedy vote (balanced)
DEBUG_MERMAID = os.environ.get("ALIGN_DEBUG_MERMAID", "0") == "1"  # print Mermaid per case

# Break majority-vote ties toward the tied value nearest this score, then toward
# the greedy anchor. 2 is the plurality label (41%), so a gentle middle bias helps.
TIE_BREAK_TOWARD = 2

# --------------------------------------------------------------------------- #
# Pipeline toggles
# --------------------------------------------------------------------------- #
USE_OCR = False         # OFF: Thai Tesseract output is garbage & redundant (VLM reads Thai)
DEBUG_OCR = os.environ.get("ALIGN_DEBUG_OCR", "0") == "1"  # print OCR text per file
TWO_PASS = False        # DEPRECATED: superseded by USE_BASELINE_B (Mermaid two-stage)
USE_FEWSHOT = True      # prepend labeled anchors from the train CSV
N_FEWSHOT = 4           # anchors per prompt: aim for one per score level (0/1/2/3)
USE_PERSONAS = False    # add strict/lenient persona votes to the ensemble

# Contrastive few-shot: real train cases at each score level from ONE problem
# (the "sum 1..n" group), so the model sees an identical design scored differently
# by Java fidelity. Order: score 0, 1, 2, 3.
FEWSHOT_CASE_IDS = ["case_17", "case_20", "case_18", "case_16"]
# Skip these anchor cases when computing validation accuracy (they would be
# trivially correct = leakage). No effect on the real test set (case_34+).
EXCLUDE_FEWSHOT_FROM_VALIDATION = True

VALID_SCORES = (0, 1, 2, 3)
DEFAULT_FALLBACK_SCORE = 1  # used only if every parse + retry fails

# --------------------------------------------------------------------------- #
# HF env — import this module early so these are set before transformers loads.
# Internet-on by default; only force offline mode when explicitly requested.
# --------------------------------------------------------------------------- #
os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")
if OFFLINE:
    os.environ["HF_HUB_OFFLINE"] = "1"
    os.environ["TRANSFORMERS_OFFLINE"] = "1"
else:
    os.environ["HF_HUB_OFFLINE"] = "0"
    os.environ["TRANSFORMERS_OFFLINE"] = "0"
