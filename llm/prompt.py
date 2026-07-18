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
SOLUTION, on this 0-3 scale. Decide it in TWO steps:

  STEP A - Is the control-flow STRUCTURE the same overall approach?
    (same kind of loop going the same direction, the same decision structure,
     the same order of operations). This sets the floor:
       - different structure / approach  -> score 0
       - same structure                  -> score is 1, 2, or 3 (go to Step B)

  STEP B - Given the same structure, how faithful are the DETAILS?
       3 = Fully consistent: steps, conditions, computations, order all match
           one-to-one.
       2 = Mostly consistent: same structure and steps, only MINOR differences
           (a loop bound like i<n vs i<=n, > vs >=, one small detail) -- score 2
           EVEN IF that small difference changes the numeric result.
       1 = Weakly consistent: the structure matches but a KEY operation or
           condition is wrong, or there are several such differences.

  0 = Inconsistent: a genuinely different algorithm / control-flow structure, or
      no meaningful correspondence at all."""

GRADING_PRINCIPLES = """\
IMPORTANT - how to score (worked examples for the design "sum 1..n":
`read n; sum=0; for i=1..n: sum=sum+i; print sum"):

  - `for i=1;i<=n;i++ sum+=i`  -> 3  (identical structure and details)
  - `for i=0;i<n;i++  sum+=i`  -> 2  (same structure; only the loop BOUNDS differ,
                                      a minor detail -- NOT a 0, even though the
                                      sum comes out different)
  - `for i=1;i<=n;i++ sum=i+n` -> 1  (same loop structure, but the BODY computes
                                      the wrong thing -- a key operation is wrong)
  - `while i>=1: sum+=i; i--`  -> 0  (counts DOWN with a while -- a different
                                      control-flow structure, even though the final
                                      sum is the same)

  Rules:
  - Do NOT give 0 just because the output would differ or a detail is wrong. 0 is
    reserved for a genuinely DIFFERENT structure / approach. Wrong details on top
    of the right structure are 1 or 2.
  - Structure differences that DO mean 0: up-counting vs down-counting loops;
    nested pairwise comparisons vs one combined condition; independent unchained
    ifs vs chained else-if; a fixed-count loop vs a different while stop condition.
  - Minor differences that stay at 2: loop bound off-by-one, > vs >=, < vs <=, one
    missing/extra trivial step, equivalent variable names, formatting.
  - Judge what each program ACTUALLY does, in order -- not whether both happen to
    solve the stated problem."""

THAI_GUIDE = """\
The design may be written in Thai (the code is Java/English). Interpret Thai
faithfully. Common terms:
  เริ่มต้น = start,  สิ้นสุด = end,  รับค่า = read/input,
  แสดงผล / แสดงค่า / พิมพ์ = print/output,  ถ้า = if,  มิฉะนั้น = else,
  มิฉะนั้นถ้า = else if,  จบเงื่อนไข = end if,  ทำซ้ำ / วนซ้ำ = loop/repeat,
  ตราบใด / ขณะที่ = while,  และ = and,  หรือ = or,  ไม่ = not,  กำหนดให้ = assign.
On a flowchart, a decision diamond's "Yes"/"ใช่" branch is the condition being
TRUE and "No"/"ไม่" is FALSE. Follow the arrows to recover the real order."""

_OUTPUT_CONTRACT = """\
First, compare the two step by step in plain text: summarize the code's algorithm,
summarize the design's algorithm, then go through input, output, order, loop,
condition, and computation, noting every real difference (especially structural
ones). THEN, on the LAST line, output ONLY this JSON object (no markdown fence):
{"dimension_findings": {"input": "match|partial|mismatch", "output": "...",
 "order": "...", "loop": "...", "condition": "...", "computation": "..."},
 "mismatches": ["..."], "final_score": <0|1|2|3>}
Mapping (apply Step A then Step B): different control-flow structure -> 0; else
one-to-one -> 3; same structure, only minor detail differences -> 2; same
structure but a key operation/condition wrong or several differences -> 1. A wrong
DETAIL on the right structure is 1 or 2, never 0. final_score MUST be an integer 0-3."""

# Persona flavor sentences for ensemble diversity (optional).
PERSONA_STRICT = "Grade strictly: any real difference in steps, conditions, or control-flow structure should pull the score down."
PERSONA_LENIENT = "Reward genuine step-by-step correspondence, but a different control-flow structure is still a mismatch even if the output is the same."


def system_prompt(persona: str | None = None) -> str:
    parts = [
        "You are a strict expert grader of algorithmic consistency between program "
        "designs and source code.",
        RUBRIC,
        GRADING_PRINCIPLES,
        THAI_GUIDE,
    ]
    if persona:
        parts.append(persona)
    parts.append(_OUTPUT_CONTRACT)
    return "\n\n".join(parts)


# --------------------------------------------------------------------------- #
# Few-shot anchors
# --------------------------------------------------------------------------- #
# Score-consistent demonstration answers. The point of the anchors is to teach
# the CALIBRATION (what each score feels like) and the reason-then-JSON format,
# so the findings must agree with the score -- never "all match" for a 0.
_ANCHOR_TEMPLATES: dict[int, tuple[str, str]] = {
    3: (
        "Step A: same control-flow structure as the design. Step B: every step, "
        "condition and computation corresponds one-to-one. No real differences.",
        '{"dimension_findings": {"input": "match", "output": "match", "order": "match", '
        '"loop": "match", "condition": "match", "computation": "match"}, '
        '"mismatches": [], "final_score": 3}',
    ),
    2: (
        "Step A: same control-flow structure. Step B: only a MINOR detail differs "
        "(e.g. a loop bound or > vs >=). Even if that changes the result, a minor "
        "detail on the right structure stays a 2 -- not a 0.",
        '{"dimension_findings": {"input": "match", "output": "match", "order": "match", '
        '"loop": "partial", "condition": "match", "computation": "match"}, '
        '"mismatches": ["minor loop-bound / operator difference"], "final_score": 2}',
    ),
    1: (
        "Step A: same control-flow structure. Step B: a KEY operation in the body "
        "is wrong (the loop matches but it computes the wrong thing), so the "
        "correspondence is only weak -- but the structure still matches, so not 0.",
        '{"dimension_findings": {"input": "match", "output": "partial", "order": "match", '
        '"loop": "match", "condition": "match", "computation": "mismatch"}, '
        '"mismatches": ["a key computation/operation differs from the design"], "final_score": 1}',
    ),
    0: (
        "Step A fails: the code uses a genuinely DIFFERENT control-flow structure "
        "than the design (e.g. a down-counting while loop instead of the design's "
        "up-counting for loop), even though the final result is the same. Different "
        "structure = inconsistent.",
        '{"dimension_findings": {"input": "match", "output": "match", "order": "mismatch", '
        '"loop": "mismatch", "condition": "mismatch", "computation": "match"}, '
        '"mismatches": ["different control-flow structure (counts down vs up)"], '
        '"final_score": 0}',
    ),
}


def build_fewshot_messages(anchors: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """`anchors` = list of {representation_type, design_text, java_code,
    java_hint, score}. Rendered TEXT-only (even flowchart anchors use the
    pseudocode text) to avoid multiple images per request."""
    messages: list[dict[str, Any]] = []
    for a in anchors:
        user = (
            f"[EXAMPLE]\nDESIGN ({a['representation_type']}):\n{a['design_text']}\n\n"
            f"JAVA SOLUTION:\n{a['java_code']}\n\n{a.get('java_hint', '')}"
        )
        messages.append({"role": "user", "content": [{"type": "text", "text": user}]})
        reasoning, json_line = _ANCHOR_TEMPLATES[int(a["score"])]
        messages.append(
            {"role": "assistant", "content": [{"type": "text", "text": f"{reasoning}\n{json_line}"}]}
        )
    return messages


# --------------------------------------------------------------------------- #
# Per-case user turns
# --------------------------------------------------------------------------- #
def flowchart_user_turn(image: Any, java_code: str, java_hint: str, ocr_text: str = "") -> dict[str, Any]:
    """Multimodal turn: the preprocessed flowchart image + Java + hints."""
    text = (
        "Grade the alignment between the FLOWCHART (image below) and the JAVA "
        "SOLUTION. Read the flowchart carefully: node text may be in Thai, while "
        "branch labels (Yes/No) are English. Trace the arrows to recover the exact "
        "order and decision structure, then compare it against the code's structure.\n\n"
    )
    if ocr_text:
        text += f"Text detected in the flowchart (OCR, may contain errors):\n{ocr_text}\n\n"
    text += (
        f"JAVA SOLUTION:\n{java_code}\n\n{java_hint}\n\n"
        "Compare them step by step, then output the JSON object on the last line."
    )
    return {
        "role": "user",
        "content": [
            {"type": "image", "image": image},
            {"type": "text", "text": text},
        ],
    }


def design_user_turn(design_text: str, java_code: str, java_hint: str, label: str = "DESIGN") -> dict[str, Any]:
    """Text-only scoring turn for any textual design (pseudocode, or a flowchart
    already transcribed to Mermaid). `label` names the design in the prompt."""
    text = (
        f"Grade the alignment between the {label} and the JAVA SOLUTION. The "
        "design may be in Thai; interpret it faithfully.\n\n"
        f"{label}:\n{design_text}\n\n"
        f"JAVA SOLUTION:\n{java_code}\n\n{java_hint}\n\n"
        "Compare them step by step, then output the JSON object on the last line."
    )
    return {"role": "user", "content": [{"type": "text", "text": text}]}


def pseudocode_user_turn(pseudo_text: str, java_code: str, java_hint: str) -> dict[str, Any]:
    return design_user_turn(pseudo_text, java_code, java_hint, label="PSEUDOCODE")


def mermaid_user_turn(mermaid_text: str, java_code: str, java_hint: str) -> dict[str, Any]:
    """Baseline B reasoning turn: the flowchart was transcribed to Mermaid; grade
    that graph against the code. The Mermaid preserves nodes, order, and Yes/No
    branches, so compare its decision STRUCTURE to the code's."""
    return design_user_turn(
        mermaid_text, java_code, java_hint,
        label="FLOWCHART (given as a Mermaid diagram)",
    )


_MERMAID_SYSTEM = (
    "You convert a flowchart image into Mermaid flowchart code, precisely and "
    "completely. Output ONLY one ```mermaid code block, nothing else."
)


def flowchart_to_mermaid_turn(image: Any, ocr_text: str = "") -> list[dict[str, Any]]:
    """Baseline B, stage 1 (perception): ask the VLM to transcribe the flowchart
    into Mermaid so the reasoning stage can compare graph structure to the code."""
    text = (
        "Convert this flowchart into Mermaid `flowchart TD` code. Rules:\n"
        "- Include EVERY node; keep its original text (Thai is fine).\n"
        "- Use node shapes to encode kind: ([start/end]), [/parallelogram I/O/], "
        "[process], {decision}.\n"
        "- Include EVERY edge, following the arrows to preserve order.\n"
        "- Label each decision's outgoing edges with its branch text "
        "(Yes/No/ใช่/ไม่).\n"
        "Output ONLY the ```mermaid code block."
    )
    if ocr_text:
        text += f"\n\nOCR text to help you (may contain errors):\n{ocr_text}"
    return [
        {"role": "system", "content": [{"type": "text", "text": _MERMAID_SYSTEM}]},
        {"role": "user", "content": [{"type": "image", "image": image}, {"type": "text", "text": text}]},
    ]


def extract_mermaid(text: str) -> str:
    """Pull the Mermaid body out of the model output, tolerating missing/extra
    fences. Returns the cleaned diagram text (never raises)."""
    if not text:
        return ""
    t = text.strip()
    # Prefer a fenced ```mermaid ... ``` block.
    lower = t.lower()
    start = lower.find("```mermaid")
    if start != -1:
        start += len("```mermaid")
        end = t.find("```", start)
        body = t[start:] if end == -1 else t[start:end]
        return body.strip()
    # Any generic fenced block.
    if t.startswith("```"):
        inner = t[3:]
        end = inner.find("```")
        return (inner if end == -1 else inner[:end]).strip()
    # Unfenced: keep from the first flowchart/graph keyword onward.
    for kw in ("flowchart", "graph "):
        i = lower.find(kw)
        if i != -1:
            return t[i:].strip()
    return t


def build_messages(
    *,
    representation_type: str,
    java_code: str,
    java_hint: str,
    image: Any = None,
    pseudo_text: str = "",
    ocr_text: str = "",
    mermaid_text: str = "",
    fewshot: list[dict[str, Any]] | None = None,
    persona: str | None = None,
) -> list[dict[str, Any]]:
    """Assemble the full chat message list for one scoring call.

    Routing: `mermaid_text` (Baseline B) wins if given; else a flowchart `image`
    (Baseline A) is used as a multimodal turn; else the `pseudo_text` (pseudocode
    case) is scored as text."""
    messages: list[dict[str, Any]] = [
        {"role": "system", "content": [{"type": "text", "text": system_prompt(persona)}]}
    ]
    if fewshot:
        messages.extend(fewshot)
    if mermaid_text:
        messages.append(mermaid_user_turn(mermaid_text, java_code, java_hint))
    elif representation_type == "flowchart" and image is not None:
        messages.append(flowchart_user_turn(image, java_code, java_hint, ocr_text))
    else:
        messages.append(pseudocode_user_turn(pseudo_text, java_code, java_hint))
    return messages
