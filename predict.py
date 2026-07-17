"""End-to-end alignment scorer for the Kaggle notebook.

Usage:
    python predict.py                # write submission.csv from id_definition.csv
    python predict.py --validate     # score the labeled train set, print accuracy
    python predict.py --dry-run 3    # run only the first 3 rows, verbose

Design: a single Qwen2.5-VL-7B-Instruct model acts as an LLM-judge. Each case is
routed by `representation_type` (flowchart image vs pseudocode text), scored with
a rubric chain-of-check prompt, and stabilized with self-consistency majority
voting (metric is exact accuracy, so voting directly helps). Robust JSON parsing
with a rule-based fallback guarantees every row gets a 0-3 score.
"""
from __future__ import annotations

import argparse
import json
import re
from collections import Counter
from pathlib import Path
from typing import Any

import pandas as pd

import config as C
from parser.java_signals import describe_java, fallback_score_from_signals
from feature.image_prep import preprocess_flowchart, ocr_flowchart
from llm import prompt as P


# --------------------------------------------------------------------------- #
# Data access
# --------------------------------------------------------------------------- #
_CASE_INDEX: dict[str, Path] | None = None


def _build_case_index() -> dict[str, Path]:
    """Index every `case_*` folder anywhere under the search roots, so we are
    robust to layout variants (train/, training/training/, test/test/, ...)."""
    index: dict[str, Path] = {}
    for base in C.CASE_SEARCH_DIRS:
        if not base.exists():
            continue
        for d in base.rglob("case_*"):
            if d.is_dir() and (d / C.JAVA_NAME).exists():
                index.setdefault(d.name, d)  # first match wins
    return index


def find_case_dir(case_id: str) -> Path:
    global _CASE_INDEX
    if _CASE_INDEX is None:
        _CASE_INDEX = _build_case_index()
    # Fast path: direct child of a search root.
    for base in C.CASE_SEARCH_DIRS:
        cand = base / case_id
        if cand.is_dir():
            return cand
    if case_id in _CASE_INDEX:
        return _CASE_INDEX[case_id]
    raise FileNotFoundError(f"Case folder not found for {case_id} under {C.CASE_SEARCH_DIRS}")


def read_java(case_dir: Path) -> str:
    return (case_dir / C.JAVA_NAME).read_text(encoding="utf-8", errors="ignore")


def read_pseudo(case_dir: Path) -> str:
    for name in (C.PSEUDO_NAME, C.PSEUDO_NAME_ALT):
        p = case_dir / name
        if p.exists():
            return p.read_text(encoding="utf-8", errors="ignore")
    return ""


def flowchart_path(case_dir: Path) -> Path:
    return case_dir / C.FLOWCHART_NAME


# --------------------------------------------------------------------------- #
# Few-shot anchors from the labeled train set
# --------------------------------------------------------------------------- #
def build_fewshot() -> list[dict[str, Any]]:
    if not (C.USE_FEWSHOT and C.TRAIN_CSV.exists()):
        return []
    df = pd.read_csv(C.TRAIN_CSV)
    anchors: list[dict[str, Any]] = []
    seen_scores: set[int] = set()
    # One anchor per score level, preferring pseudocode (text-only, cheap).
    df = df.sort_values("representation_type", ascending=False)  # pseudocode before flowchart
    for _, row in df.iterrows():
        score = int(row["alignment_score"])
        if score in seen_scores or len(anchors) >= C.N_FEWSHOT:
            continue
        try:
            case_dir = find_case_dir(str(row["case_id"]))
            java_code = read_java(case_dir)
            rep = str(row["representation_type"])
            if rep == "flowchart":
                design_text = read_pseudo(case_dir) or "(flowchart; see steps)"
            else:
                design_text = read_pseudo(case_dir)
        except FileNotFoundError:
            continue
        anchors.append(
            {
                "representation_type": rep,
                "design_text": design_text[:1500],
                "java_code": java_code[:2000],
                "java_hint": describe_java(java_code),
                "score": score,
            }
        )
        seen_scores.add(score)
    return P.build_fewshot_messages(anchors)


# --------------------------------------------------------------------------- #
# Model wrapper
# --------------------------------------------------------------------------- #
class Judge:
    def __init__(self) -> None:
        import torch
        from transformers import AutoProcessor
        try:
            from transformers import Qwen2_5_VLForConditionalGeneration as VLModel
        except Exception:  # older transformers
            from transformers import Qwen2VLForConditionalGeneration as VLModel

        self.torch = torch
        dtype = getattr(torch, C.TORCH_DTYPE)

        load_kwargs: dict[str, Any] = dict(
            torch_dtype=dtype,
            attn_implementation=C.ATTN_IMPLEMENTATION,
            device_map="auto",
            max_memory=C.MAX_MEMORY,
        )
        if C.LOAD_IN_4BIT:
            from transformers import BitsAndBytesConfig

            load_kwargs["quantization_config"] = BitsAndBytesConfig(
                load_in_4bit=True,
                bnb_4bit_compute_dtype=dtype,
                bnb_4bit_quant_type="nf4",
                bnb_4bit_use_double_quant=True,
            )
            load_kwargs.pop("torch_dtype", None)

        self.model = VLModel.from_pretrained(C.MODEL_DIR, **load_kwargs).eval()
        self.processor = AutoProcessor.from_pretrained(
            C.MODEL_DIR, min_pixels=C.MIN_PIXELS, max_pixels=C.MAX_PIXELS
        )

    def _prepare_inputs(self, messages: list[dict[str, Any]]):
        from qwen_vl_utils import process_vision_info

        text = self.processor.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True
        )
        image_inputs, video_inputs = process_vision_info(messages)
        inputs = self.processor(
            text=[text],
            images=image_inputs,
            videos=video_inputs,
            padding=True,
            return_tensors="pt",
        )
        return inputs.to(self.model.device)

    def generate(self, messages: list[dict[str, Any]], sample: bool) -> str:
        inputs = self._prepare_inputs(messages)
        gen_kwargs: dict[str, Any] = dict(max_new_tokens=C.MAX_NEW_TOKENS)
        if sample:
            gen_kwargs.update(
                do_sample=True, temperature=C.SAMPLE_TEMPERATURE, top_p=C.SAMPLE_TOP_P
            )
        else:
            gen_kwargs.update(do_sample=False)
        with self.torch.no_grad():
            out = self.model.generate(**inputs, **gen_kwargs)
        trimmed = out[:, inputs["input_ids"].shape[1]:]
        return self.processor.batch_decode(
            trimmed, skip_special_tokens=True, clean_up_tokenization_spaces=False
        )[0]


# --------------------------------------------------------------------------- #
# Output parsing
# --------------------------------------------------------------------------- #
_SCORE_RE = re.compile(r'"final_score"\s*:\s*([0-3])')
_LOOSE_RE = re.compile(r"\b([0-3])\b")


def parse_score(text: str) -> int | None:
    """Pull final_score from the model output. Try JSON, then regex, then a
    last-resort scan of the tail. Returns None if nothing valid is found."""
    m = _SCORE_RE.search(text)
    if m:
        return int(m.group(1))
    # Try to locate a JSON object and parse it.
    start = text.find("{")
    end = text.rfind("}")
    if start != -1 and end != -1 and end > start:
        try:
            obj = json.loads(text[start : end + 1])
            v = int(obj.get("final_score"))
            if v in C.VALID_SCORES:
                return v
        except Exception:
            pass
    # Last resort: last standalone 0-3 in the text.
    matches = _LOOSE_RE.findall(text)
    if matches:
        return int(matches[-1])
    return None


# --------------------------------------------------------------------------- #
# Scoring one case (with self-consistency)
# --------------------------------------------------------------------------- #
def score_case(
    judge: Judge,
    fewshot: list[dict[str, Any]],
    *,
    representation_type: str,
    case_dir: Path,
    verbose: bool = False,
) -> int:
    java_code = read_java(case_dir)
    java_hint = describe_java(java_code)
    design_text_for_fallback = read_pseudo(case_dir)

    image = None
    ocr_text = ""
    pseudo_text = ""

    if representation_type == "flowchart":
        image = preprocess_flowchart(flowchart_path(case_dir))
        if C.USE_OCR:
            ocr_text = ocr_flowchart(image)
            design_text_for_fallback = ocr_text or design_text_for_fallback
        if C.TWO_PASS:
            # Transcribe the flowchart to text, then score as a text task.
            trans_msgs = P.transcribe_flowchart_turn(image, ocr_text)
            pseudo_text = judge.generate(trans_msgs, sample=False).strip()
            design_text_for_fallback = pseudo_text or design_text_for_fallback
            image = None
            representation_type = "pseudocode"
    else:
        pseudo_text = read_pseudo(case_dir)

    personas = [None]
    if C.USE_PERSONAS:
        personas = [None, P.PERSONA_STRICT, P.PERSONA_LENIENT]

    votes: list[int] = []
    # Greedy pass per persona.
    for persona in personas:
        messages = P.build_messages(
            representation_type=representation_type,
            java_code=java_code,
            java_hint=java_hint,
            image=image,
            pseudo_text=pseudo_text,
            ocr_text=ocr_text,
            fewshot=fewshot,
            persona=persona,
        )
        out = judge.generate(messages, sample=False)
        s = parse_score(out)
        if s is None:  # one retry
            out = judge.generate(messages, sample=False)
            s = parse_score(out)
        if s is not None:
            votes.append(s)
        if verbose:
            print(f"  [greedy/{persona}] -> {s}\n  {out[:200]}")

    # Sampled self-consistency votes (single persona to bound cost).
    base_messages = P.build_messages(
        representation_type=representation_type,
        java_code=java_code,
        java_hint=java_hint,
        image=image,
        pseudo_text=pseudo_text,
        ocr_text=ocr_text,
        fewshot=fewshot,
        persona=None,
    )
    for _ in range(C.N_SAMPLES):
        out = judge.generate(base_messages, sample=True)
        s = parse_score(out)
        if s is not None:
            votes.append(s)

    if not votes:
        return fallback_score_from_signals(java_code, design_text_for_fallback)

    # Majority vote; tie-break toward the first (greedy) vote.
    counts = Counter(votes)
    top = counts.most_common()
    best_n = top[0][1]
    tied = [v for v, n in top if n == best_n]
    if len(tied) == 1:
        return tied[0]
    for v in votes:  # greedy vote comes first in `votes`
        if v in tied:
            return v
    return tied[0]


# --------------------------------------------------------------------------- #
# Runners
# --------------------------------------------------------------------------- #
def run_submission(judge: Judge, fewshot, dry_run: int | None = None) -> pd.DataFrame:
    ids = pd.read_csv(C.ID_DEFINITION_CSV)
    ids.columns = [c.strip().lower() for c in ids.columns]  # Id/Case_id/Representation_type
    rows = []
    n = len(ids) if dry_run is None else min(dry_run, len(ids))
    for i in range(n):
        row = ids.iloc[i]
        case_id = str(row["case_id"])
        rep = str(row["representation_type"]).strip().lower()
        case_dir = find_case_dir(case_id)
        score = score_case(
            judge, fewshot, representation_type=rep, case_dir=case_dir,
            verbose=dry_run is not None,
        )
        rows.append({"Id": int(row["id"]), "Alignment_score": int(score)})
        print(f"[{i+1}/{n}] {case_id} ({rep}) -> {score}")

    sub = pd.DataFrame(rows)
    assert sub["Alignment_score"].isin(C.VALID_SCORES).all(), "score out of range"
    return sub


def run_validation(judge: Judge, fewshot) -> None:
    df = pd.read_csv(C.TRAIN_CSV)
    y_true, y_pred = [], []
    for _, row in df.iterrows():
        case_id = str(row["case_id"])
        rep = str(row["representation_type"]).strip().lower()
        try:
            case_dir = find_case_dir(case_id)
        except FileNotFoundError:
            continue
        pred = score_case(judge, fewshot, representation_type=rep, case_dir=case_dir)
        y_true.append(int(row["alignment_score"]))
        y_pred.append(int(pred))
        print(f"{case_id} ({rep}): true={row['alignment_score']} pred={pred}")

    acc = sum(int(a == b) for a, b in zip(y_true, y_pred)) / max(len(y_true), 1)
    print(f"\nExact-match accuracy: {acc:.3f}  (n={len(y_true)})")
    # 4x4 confusion matrix.
    cm = [[0] * 4 for _ in range(4)]
    for a, b in zip(y_true, y_pred):
        cm[a][b] += 1
    print("Confusion matrix (rows=true, cols=pred):")
    print("      p0  p1  p2  p3")
    for t in range(4):
        print(f"  t{t}  " + " ".join(f"{cm[t][p]:3d}" for p in range(4)))


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--validate", action="store_true", help="score labeled train set")
    ap.add_argument("--dry-run", type=int, default=None, metavar="N", help="run first N rows, verbose")
    args = ap.parse_args()

    fewshot = build_fewshot()
    judge = Judge()

    if args.validate:
        run_validation(judge, fewshot)
        return

    sub = run_submission(judge, fewshot, dry_run=args.dry_run)
    C.OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    sub.to_csv(C.SUBMISSION_PATH, index=False)
    print(f"\nWrote {len(sub)} rows -> {C.SUBMISSION_PATH}")


if __name__ == "__main__":
    main()
