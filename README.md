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

## Data

The competition data is **not committed** (see `.gitignore`). Place it locally under
`datasets/` (any nesting works — case discovery is recursive) or point `ALIGN_DATA_DIR`
at its location. On Kaggle, attach it as a competition/dataset input instead.

## Running on Kaggle (internet OFF)

1. **Attach the model**: add *Qwen2.5-VL-7B-Instruct* as a Kaggle **Model** input.
   Set `ALIGN_MODEL_DIR` (or edit `config.MODEL_DIR`) to the snapshot folder that
   contains `config.json`.
2. **Attach this code** as a Kaggle **Dataset**, or paste the modules into the
   notebook. Set `ALIGN_DATA_DIR` to the competition data mount.
3. Run `notebook.ipynb`. It writes `/kaggle/working/submission.csv`
   (columns `Id,Alignment_score`).

Local dry run against a sample folder with the same layout:

```bash
export ALIGN_DATA_DIR=/path/to/sample
export ALIGN_MODEL_DIR=/path/to/qwen2.5-vl-7b
python predict.py --dry-run 3     # first 3 rows, verbose
python predict.py --validate      # exact accuracy + confusion matrix on train
python predict.py                 # full submission
```

## T4 (Turing) gotchas — already handled in `config.py`

- **No bfloat16** → `torch_dtype=float16`.
- **No FlashAttention-2** → `attn_implementation="sdpa"`.
- **16 GB/GPU** → `device_map="auto"` with `max_memory` head-room; flip
  `LOAD_IN_4BIT=True` (bitsandbytes, ~7 GB) if fp16 + image tokens OOM.
- **Internet off** → `HF_HUB_OFFLINE=1` / `TRANSFORMERS_OFFLINE=1` set on import;
  load the model from a local path only.
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
