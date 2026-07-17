"""Prompt construction for the LLM-as-judge.

One rubric-grounded system prompt, two user templates (flowchart image vs
pseudocode text) that share a single 6-dimension chain-of-check, and a few-shot
anchor builder that pulls one labeled example per score level from the train set.

The model is asked to emit STRICT JSON so `final_score` is trivially parseable.
All messages are returned in the OpenAI/Qwen chat format:
    [{"role": ..., "content": [{"type": "text"|"image", ...}]}]
so `predict.py` can feed them straight to the Qwen processor.
"""
from __future__ import annotations

from typing import Any

# --------------------------------------------------------------------------- #
# Rubric — transcribed from the competition brief (page 7-8), behavior over
# syntax. Kept verbatim in meaning so the model grades on the organizers' scale.
# --------------------------------------------------------------------------- #
RUBRIC = """\
You grade how well a program DESIGN (a flowchart or pseudocode) matches a Java
SOLUTION, on this 0-3 scale:

  0 = Inconsistent: clearly do not match, or they use a fundamentally different
      approach / algorithm to solve the problem.
  1 = Weakly consistent: some parts match, but there are several important
      differences (missing or wrong steps, conditions, or control flow).
  2 = Mostly consistent: the main approach matches; only minor differences or a
      few missing details remain.
  3 = Fully consistent: the concept, the steps, the conditions, and the order of
      execution all match.

Judge BEHAVIOR, not surface form:
  - The sequence/order of operations should correspond.
  - Conditions and branches should be equivalent.
  - Loops should match in kind, iteration count, and stop condition.
  - Input, computation, and output should mean the same thing.
  - Variable names, formatting, and exact wording do NOT need to be identical if
    the behavior is equivalent."""

_JSON_CONTRACT = """\
Respond with ONE JSON object and nothing else (no markdown, no prose outside it):
{
  "java_summary": "<ordered steps: inputs, loops (bounds+stop), branches, computation, outputs>",
  "design_summary": "<same schema, read from the design>",
  "dimension_findings": {
    "input":       "match|partial|mismatch",
    "output":      "match|partial|mismatch",
    "order":       "match|partial|mismatch",
    "loop":        "match|partial|mismatch",
    "condition":   "match|partial|mismatch",
    "computation": "match|partial|mismatch"
  },
  "mismatches": ["<meaningful difference>", "..."],
  "final_score": 0
}
Map the findings to the rubric: all/near-all "match" -> 3; mostly match, minor
gaps -> 2; several "mismatch"/"partial" on important axes -> 1; different
approach or pervasive mismatch -> 0. "final_score" MUST be an integer 0, 1, 2, or 3."""

# Persona flavor sentences for ensemble diversity (optional).
PERSONA_STRICT = "Grade strictly: any real difference in steps, conditions, or order should pull the score down."
PERSONA_LENIENT = "Grade for behavioral equivalence: reward matches in overall approach even if minor details differ."


def system_prompt(persona: str | None = None) -> str:
    parts = [
        "You are an expert grader of algorithmic consistency between program "
        "designs and source code.",
        RUBRIC,
    ]
    if persona:
        parts.append(persona)
    parts.append(_JSON_CONTRACT)
    return "\n\n".join(parts)


# --------------------------------------------------------------------------- #
# Few-shot anchors
# --------------------------------------------------------------------------- #
def build_fewshot_messages(anchors: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """`anchors` = list of {representation_type, design_text, java_code,
    java_hint, score}. We render TEXT-only anchors (even flowchart ones use a
    short textual design description) to avoid multiple images per request."""
    messages: list[dict[str, Any]] = []
    for a in anchors:
        user = (
            f"[EXAMPLE]\nDESIGN ({a['representation_type']}):\n{a['design_text']}\n\n"
            f"JAVA SOLUTION:\n{a['java_code']}\n\n{a.get('java_hint', '')}"
        )
        messages.append({"role": "user", "content": [{"type": "text", "text": user}]})
        # A minimal but well-formed JSON answer teaches the output shape.
        answer = (
            '{"java_summary": "...", "design_summary": "...", '
            '"dimension_findings": {"input": "match", "output": "match", '
            '"order": "match", "loop": "match", "condition": "match", '
            '"computation": "match"}, "mismatches": [], '
            f'"final_score": {int(a["score"])}}}'
        )
        messages.append({"role": "assistant", "content": [{"type": "text", "text": answer}]})
    return messages


# --------------------------------------------------------------------------- #
# Per-case user turns
# --------------------------------------------------------------------------- #
def flowchart_user_turn(image: Any, java_code: str, java_hint: str, ocr_text: str = "") -> dict[str, Any]:
    """Multimodal turn: the preprocessed flowchart image + Java + hints."""
    text = "Grade the alignment between the FLOWCHART (image below) and the JAVA SOLUTION.\n\n"
    if ocr_text:
        text += f"Text detected in the flowchart (OCR, may contain errors):\n{ocr_text}\n\n"
    text += f"JAVA SOLUTION:\n{java_code}\n\n{java_hint}\n\nReturn the JSON now."
    return {
        "role": "user",
        "content": [
            {"type": "image", "image": image},
            {"type": "text", "text": text},
        ],
    }


def pseudocode_user_turn(pseudo_text: str, java_code: str, java_hint: str) -> dict[str, Any]:
    text = (
        "Grade the alignment between the PSEUDOCODE and the JAVA SOLUTION.\n\n"
        f"PSEUDOCODE:\n{pseudo_text}\n\n"
        f"JAVA SOLUTION:\n{java_code}\n\n{java_hint}\n\nReturn the JSON now."
    )
    return {"role": "user", "content": [{"type": "text", "text": text}]}


def transcribe_flowchart_turn(image: Any, ocr_text: str = "") -> list[dict[str, Any]]:
    """Two-pass mode: ask the VLM to transcribe the flowchart into normalized
    text first (no scoring). Returns a full message list."""
    text = (
        "Transcribe this flowchart into a normalized, ordered list of steps. "
        "For each node write its kind (start/input/process/decision/output/end) "
        "and its text. Follow the arrows to preserve order and branches. "
        "Output plain text only."
    )
    if ocr_text:
        text += f"\n\nOCR text to help you (may contain errors):\n{ocr_text}"
    return [
        {"role": "system", "content": [{"type": "text", "text": "You transcribe flowcharts precisely."}]},
        {"role": "user", "content": [{"type": "image", "image": image}, {"type": "text", "text": text}]},
    ]


def build_messages(
    *,
    representation_type: str,
    java_code: str,
    java_hint: str,
    image: Any = None,
    pseudo_text: str = "",
    ocr_text: str = "",
    fewshot: list[dict[str, Any]] | None = None,
    persona: str | None = None,
) -> list[dict[str, Any]]:
    """Assemble the full chat message list for one scoring call."""
    messages: list[dict[str, Any]] = [
        {"role": "system", "content": [{"type": "text", "text": system_prompt(persona)}]}
    ]
    if fewshot:
        messages.extend(fewshot)
    if representation_type == "flowchart" and image is not None:
        messages.append(flowchart_user_turn(image, java_code, java_hint, ocr_text))
    else:
        # pseudocode, or flowchart already transcribed into pseudo_text (two-pass)
        messages.append(pseudocode_user_turn(pseudo_text, java_code, java_hint))
    return messages
