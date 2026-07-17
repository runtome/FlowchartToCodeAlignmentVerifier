# Flowchart / Pseudocode → Java Alignment Verifier

SuperAI Season 6 hackathon (BUU). Scores **algorithmic consistency (0–3)** between a
program design and its Java source, for two sub-tasks:

- `flowchart.png` ↔ `solution.java`
- `pseudocode.txt` ↔ `solution.java`

Runs fully offline on **Kaggle T4×2** using an open-source VLM as an LLM-judge —
no commercial APIs, no hard-coded answers.

## Approach

A single **Qwen2.5-VL-7B-Instruct** grades each case with a rubric-grounded
chain-of-check prompt (summarize Java → summarize design → compare 6 dimensions →
emit strict JSON with `final_score`). **Self-consistency majority voting** stabilizes
the output, which is what the exact-match metric rewards. Robust JSON parsing with a
rule-based fallback guarantees every row gets a 0–3 score.

```
id_definition.csv → route by representation_type
   flowchart  → preprocess PNG (+OCR) →┐
   pseudocode → read TXT ──────────────┤→ Qwen2.5-VL judge → majority vote → submission.csv
```

## Files

| File | Role |
|------|------|
| `config.py` | Paths, model dir, decoding params, pipeline toggles, offline env flags |
| `parser/java_signals.py` | Java structural hints + rule-based fallback score |
| `feature/image_prep.py` | In-memory flowchart preprocessing (+ optional OCR) |
| `llm/prompt.py` | Rubric system prompt, flowchart/pseudocode templates, few-shot |
| `predict.py` | Model load, self-consistency inference, submission + validation |
| `notebook.ipynb` | Thin Kaggle entry point that calls the modules |
| `documents/pipeline.md` | Full walkthrough of how a case flows through the code |

## Data

The ~2 MB sample dataset **is committed** under `datasets/` so `git clone` + validation
works out of the box (any nesting is fine — case discovery is recursive):

```
datasets/
  alignment_score_training.csv   # train labels (case_id, representation_type, alignment_score)
  id_definition.csv              # test rows   (Id, Case_id, Representation_type)
  sample_submission.csv          # submission template (Id, Alignment_score)
  training/training/case_XX/{flowchart.png, pseudo.txt, solution.java}
  test/test/case_XX/{flowchart.png, pseudo.txt, solution.java}
```

To use a different data location (e.g. a Kaggle dataset mount), set `ALIGN_DATA_DIR`.

## Running on Kaggle (internet ON, GPU T4 ×2)

In the notebook sidebar: **Accelerator = GPU T4 x2**, **Internet = On**. Then run
`notebook.ipynb`, which does:

```python
!git clone https://github.com/<you>/FlowchartToCodeAlignmentVerifier.git
%cd FlowchartToCodeAlignmentVerifier
!pip install -q -U transformers accelerate qwen-vl-utils javalang
!python predict.py --validate     # train accuracy (%), downloads the model first run
!python predict.py                # writes /kaggle/working/submission.csv
```

The model (`Qwen/Qwen2.5-VL-7B-Instruct`, ~16 GB) downloads from Hugging Face on the
first call. No API keys and no attached-model step needed.

Local runs work the same way:

```bash
python predict.py --validate --dry-run 4   # first 4 rows, verbose
python predict.py --validate               # exact-match % accuracy + confusion matrix
python predict.py                          # full submission -> submission.csv
```

## Hardware notes — already handled in `config.py`

- **T4 has no bfloat16** → `torch_dtype=float16`.
- **T4 has no FlashAttention-2** → `attn_implementation="sdpa"`.
- **16 GB/GPU** → `device_map="auto"` with `max_memory` head-room; flip
  `LOAD_IN_4BIT=True` (bitsandbytes, ~7 GB) if fp16 + image tokens OOM.
- **Internet on by default**; set `ALIGN_OFFLINE=1` + a local `ALIGN_MODEL_DIR` for an
  offline run.
- Visual tokens capped via processor `min_pixels`/`max_pixels`.

## Tuning knobs (in `config.py`)

- `N_SAMPLES`, `SAMPLE_TEMPERATURE` — self-consistency strength.
- `USE_OCR` — flowchart OCR augmentation (biggest vision-side accuracy lever).
- `TWO_PASS` — transcribe flowchart → text, then score as text (decouples vision errors).
- `USE_FEWSHOT`, `N_FEWSHOT` — labeled anchors from the train CSV.
- `USE_PERSONAS` — add strict/lenient votes to the ensemble.

## Roadmap (post-baseline)

Self-consistency and OCR are wired in. Next levers, in ROI order: two-pass
transcription, strict/lenient persona ensemble, then (last, optional) QLoRA — noted
as high-overfit risk given the tiny label set.
