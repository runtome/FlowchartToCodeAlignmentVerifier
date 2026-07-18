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
import random
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
    """Curated contrastive anchors: the exact train cases in C.FEWSHOT_CASE_IDS
    (an identical design scored 0/1/2/3 by Java fidelity), each rendered as its
    pseudocode design + Java + a score-appropriate reasoning demonstration."""
    if not (C.USE_FEWSHOT and C.TRAIN_CSV.exists()):
        return []
    df = pd.read_csv(C.TRAIN_CSV)
    # case_id -> label (flowchart and pseudocode share a label; take the first).
    score_by_case = (
        df.groupby("case_id")["alignment_score"].first().astype(int).to_dict()
    )
    anchors: list[dict[str, Any]] = []
    for case_id in C.FEWSHOT_CASE_IDS:
        if case_id not in score_by_case:
            continue
        try:
            case_dir = find_case_dir(case_id)
            java_code = read_java(case_dir)
            design_text = read_pseudo(case_dir) or "(flowchart; see steps)"
        except FileNotFoundError:
            continue
        anchors.append(
            {
                "representation_type": "pseudocode",
                "design_text": design_text[:1500],
                "java_code": java_code[:2000],
                "java_hint": describe_java(java_code),
                "score": score_by_case[case_id],
            }
        )
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
    """Pull final_score from the model output. The model reasons first and emits
    the JSON last, so prefer the LAST match. Try the explicit key, then a JSON
    object, then a last-resort scan of the tail. None if nothing valid is found."""
    matches = _SCORE_RE.findall(text)
    if matches:
        return int(matches[-1])
    # Try to locate the last JSON object and parse it.
    end = text.rfind("}")
    start = text.rfind("{", 0, end) if end != -1 else -1
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
# Voting helpers
# --------------------------------------------------------------------------- #
def _collect_votes(judge, build_msgs, n_samples, *, tag="", verbose=False):
    """One greedy vote (with a single parse retry) + `n_samples` sampled votes
    over the same messages. Returns (votes, greedy_value)."""
    messages = build_msgs()
    out = judge.generate(messages, sample=False)
    greedy = parse_score(out)
    if greedy is None:  # one retry on unparseable output
        out = judge.generate(messages, sample=False)
        greedy = parse_score(out)
    if verbose:
        print(f"  [{tag} greedy] -> {greedy}  ::  {out[:140].replace(chr(10), ' ')}")

    votes: list[int] = [greedy] if greedy is not None else []
    for _ in range(n_samples):
        s = parse_score(judge.generate(messages, sample=True))
        if s is not None:
            votes.append(s)
    return votes, greedy


def _majority(votes: list[int], prefer: int | None) -> int:
    """Most common vote. Ties break toward the tied value nearest
    `C.TIE_BREAK_TOWARD` (a gentle middle bias); among values equally near, toward
    the greedy anchor `prefer`, else the one that appears first."""
    counts = Counter(votes)
    best_n = counts.most_common(1)[0][1]
    tied = [v for v, n in counts.most_common() if n == best_n]
    if len(tied) == 1:
        return tied[0]
    # nearest to the target; tie-break-within-tie by greedy anchor, then order.
    order = {v: i for i, v in enumerate(votes)}
    return min(
        tied,
        key=lambda v: (abs(v - C.TIE_BREAK_TOWARD), 0 if v == prefer else 1, order.get(v, 99)),
    )


# --------------------------------------------------------------------------- #
# Scoring one case — ensemble of Baseline A + Baseline B + structural vote
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

    is_flow = representation_type == "flowchart"
    image = None
    ocr_text = ""
    pseudo_text = ""

    if is_flow:
        image = preprocess_flowchart(flowchart_path(case_dir))
        if C.USE_OCR:
            ocr_text = ocr_flowchart(image)
            design_text_for_fallback = ocr_text or design_text_for_fallback
            if C.DEBUG_OCR:
                flat = " | ".join(t.strip() for t in ocr_text.splitlines() if t.strip())
                print(f'ocr:{case_dir.name}/{C.FLOWCHART_NAME} : "{flat or "<empty — no OCR engine or no text>"}"')
    else:
        pseudo_text = read_pseudo(case_dir)
        design_text_for_fallback = pseudo_text or design_text_for_fallback

    votes: list[int] = []
    greedy_ref: int | None = None

    personas = [None, P.PERSONA_STRICT, P.PERSONA_LENIENT] if C.USE_PERSONAS else [None]

    # ---- Baseline A: direct image (flowchart) or text (pseudocode) judge ----
    if C.USE_BASELINE_A:
        for persona in personas:
            n = C.SAMPLES_PER_BASELINE if persona is None else 0  # sample only the base persona
            v, g = _collect_votes(
                judge,
                lambda p=persona: P.build_messages(
                    representation_type=representation_type, java_code=java_code,
                    java_hint=java_hint, image=image, pseudo_text=pseudo_text,
                    ocr_text=ocr_text, fewshot=fewshot, persona=p,
                ),
                n, tag=f"A/{persona or 'base'}", verbose=verbose,
            )
            votes += v
            if greedy_ref is None:
                greedy_ref = g

    # ---- Baseline B: image -> Mermaid -> judge (flowchart cases only) ----
    if C.USE_BASELINE_B and is_flow:
        mermaid = P.extract_mermaid(
            judge.generate(P.flowchart_to_mermaid_turn(image, ocr_text), sample=False)
        )
        if C.DEBUG_MERMAID:
            flat = " ; ".join(l.strip() for l in mermaid.splitlines() if l.strip())
            print(f'mermaid:{case_dir.name}/{C.FLOWCHART_NAME} : "{flat[:400] or "<empty>"}"')
        if mermaid:
            v, g = _collect_votes(
                judge,
                lambda: P.build_messages(
                    representation_type="flowchart", java_code=java_code,
                    java_hint=java_hint, mermaid_text=mermaid, fewshot=fewshot,
                ),
                C.SAMPLES_PER_BASELINE, tag="B/mermaid", verbose=verbose,
            )
            votes += v
            if greedy_ref is None:
                greedy_ref = g

    # ---- Structural check: one rule-based vote ----
    if C.USE_STRUCTURAL_VOTE:
        struct = fallback_score_from_signals(java_code, design_text_for_fallback)
        votes.append(struct)
        if verbose:
            print(f"  [structural] -> {struct}")

    if not votes:
        return fallback_score_from_signals(java_code, design_text_for_fallback)
    return _majority(votes, greedy_ref)


# --------------------------------------------------------------------------- #
# Runners
# --------------------------------------------------------------------------- #
def _limit_rows(df: pd.DataFrame, dry_run: int | None, random_sample: bool, seed: int | None) -> pd.DataFrame:
    """Trim to `dry_run` rows. With `random_sample`, first reorder by a random
    CASE order (keeping each case's flowchart+pseudocode rows together and in
    their original order), so `--dry-run 6` picks 3 random folders, not the first
    3. The seed is printed so a run can be reproduced."""
    if random_sample:
        if seed is None:
            seed = random.randrange(1_000_000)
        print(f"[random dry-run] seed={seed}  (reuse with --seed {seed})")
        cases = list(dict.fromkeys(df["case_id"].tolist()))  # unique, keep order
        random.Random(seed).shuffle(cases)
        rank = {c: i for i, c in enumerate(cases)}
        df = (
            df.assign(_ord=df["case_id"].map(rank))
            .sort_values("_ord", kind="stable")
            .drop(columns="_ord")
            .reset_index(drop=True)
        )
    if dry_run is not None:
        df = df.head(dry_run)
    return df


def run_submission(
    judge: Judge, fewshot, dry_run: int | None = None,
    random_sample: bool = False, seed: int | None = None,
) -> pd.DataFrame:
    ids = pd.read_csv(C.ID_DEFINITION_CSV)
    ids.columns = [c.strip().lower() for c in ids.columns]  # Id/Case_id/Representation_type
    ids = _limit_rows(ids, dry_run, random_sample, seed)
    rows = []
    n = len(ids)
    for i, (_, row) in enumerate(ids.iterrows()):
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


def run_validation(
    judge: Judge, fewshot, dry_run: int | None = None,
    random_sample: bool = False, seed: int | None = None,
) -> float:
    df = pd.read_csv(C.TRAIN_CSV)
    skip = set(C.FEWSHOT_CASE_IDS) if (C.USE_FEWSHOT and C.EXCLUDE_FEWSHOT_FROM_VALIDATION) else set()
    if skip:
        df = df[~df["case_id"].isin(skip)]
        print(f"(excluding {len(skip)} few-shot anchor case(s) from accuracy: {sorted(skip)})")
    df = _limit_rows(df, dry_run, random_sample, seed)
    y_true, y_pred, y_rep = [], [], []
    for _, row in df.iterrows():
        case_id = str(row["case_id"])
        rep = str(row["representation_type"]).strip().lower()
        try:
            case_dir = find_case_dir(case_id)
        except FileNotFoundError:
            print(f"  (skip {case_id}: folder not found)")
            continue
        pred = score_case(judge, fewshot, representation_type=rep, case_dir=case_dir)
        t = int(row["alignment_score"])
        y_true.append(t)
        y_pred.append(int(pred))
        y_rep.append(rep)
        mark = "OK " if t == pred else "XX "
        print(f"{mark}{case_id} ({rep}): true={t} pred={pred}")

    n = max(len(y_true), 1)
    correct = sum(int(a == b) for a, b in zip(y_true, y_pred))
    within1 = sum(int(abs(a - b) <= 1) for a, b in zip(y_true, y_pred))
    acc = correct / n
    # "always predict 2" prior, the bar an ordinal judge must beat.
    prior = max((sum(int(t == k) for t in y_true) for k in range(4)), default=0) / n
    print("\n" + "=" * 40)
    print(f"EXACT-MATCH ACCURACY: {acc * 100:.1f}%   ({correct}/{len(y_true)})")
    print(f"  within +/-1        : {within1 / n * 100:.1f}%   ({within1}/{len(y_true)})")
    print(f"  majority-class prior: {prior * 100:.1f}%   (beat this)")

    # Per-representation-type accuracy (flowchart vs pseudocode).
    for rep in ("flowchart", "pseudocode"):
        idx = [i for i, r in enumerate(y_rep) if r == rep]
        if idx:
            c = sum(int(y_true[i] == y_pred[i]) for i in idx)
            print(f"  {rep:<11}: {c / len(idx) * 100:.1f}%   ({c}/{len(idx)})")
    print("=" * 40)

    # 4x4 confusion matrix.
    cm = [[0] * 4 for _ in range(4)]
    for a, b in zip(y_true, y_pred):
        cm[a][b] += 1
    print("Confusion matrix (rows=true, cols=pred):")
    print("      p0  p1  p2  p3")
    for t in range(4):
        print(f"  t{t}  " + " ".join(f"{cm[t][p]:3d}" for p in range(4)))
    return acc


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--validate", action="store_true", help="score labeled train set")
    ap.add_argument("--dry-run", type=int, default=None, metavar="N", help="run only N rows, verbose")
    ap.add_argument("--random", action="store_true", help="with --dry-run, pick random cases (folders) instead of the first ones")
    ap.add_argument("--seed", type=int, default=None, help="seed for --random (reproducible sampling)")
    ap.add_argument("--baseline", choices=["A", "B", "AB"], default=None,
                    help="override baselines for this run: A=image only, B=Mermaid only, AB=both")
    args = ap.parse_args()

    if args.baseline:  # override config for an A / B / A+B measurement run
        C.USE_BASELINE_A = args.baseline in ("A", "AB")
        C.USE_BASELINE_B = args.baseline in ("B", "AB")
        print(f"[baseline override] A={C.USE_BASELINE_A} B={C.USE_BASELINE_B}")

    fewshot = build_fewshot()
    judge = Judge()

    if args.validate:
        run_validation(judge, fewshot, dry_run=args.dry_run, random_sample=args.random, seed=args.seed)
        return

    sub = run_submission(judge, fewshot, dry_run=args.dry_run, random_sample=args.random, seed=args.seed)
    C.OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    sub.to_csv(C.SUBMISSION_PATH, index=False)
    print(f"\nWrote {len(sub)} rows -> {C.SUBMISSION_PATH}")


if __name__ == "__main__":
    main()
