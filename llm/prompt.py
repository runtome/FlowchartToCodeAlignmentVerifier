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

  0 = Inconsistent: they do not match, OR the code uses a different algorithm /
      control-flow STRUCTURE than the design -- even if both happen to produce the
      same output.
  1 = Weakly consistent: some parts match, but there are several important
      differences in steps, conditions, or control flow.
  2 = Mostly consistent: the structure and steps match the design; only MINOR
      differences remain (e.g. > vs >=, i<n vs i<=n-1, a missing detail).
  3 = Fully consistent: the concept, the steps, the conditions, and the order of
      execution all correspond one-to-one."""

GRADING_PRINCIPLES = """\
IMPORTANT - how to compare:
  - You are checking whether the design and the code describe the SAME algorithm
    STEP BY STEP: the same inputs, the same sequence of operations, the same
    decision structure (how many decisions and what each one tests), the same
    loops (kind, count, stop condition), and the same outputs.
  - Producing the same final RESULT through a DIFFERENT structure is NOT full
    consistency. Examples that should score LOW (0-1), not high:
      * design tests "a >= b AND a >= c" in one condition, but code uses nested
        pairwise comparisons (if a>b { if a>c ... }) -> different decision
        structure -> mismatch.
      * design uses chained else-if, but code uses independent unchained ifs.
      * design loops a fixed number of times, but code uses a while with a
        different stop condition.
  - Differences that are only MINOR (cap the score around 2, do not force 0):
      * > vs >=, < vs <=, or an off-by-one in a loop bound that keeps the same
        behavior; different but equivalent variable names; formatting.
  - Only variable names, formatting, and exact wording may differ freely. The
    control-flow structure must genuinely correspond for a 3.
  - Do NOT assume consistency just because both programs solve the stated problem.
    Compare what each one ACTUALLY does, in order."""

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
Mapping: structure+steps correspond one-to-one -> 3; same structure with only
minor differences -> 2; several real differences -> 1; different structure /
algorithm or pervasive mismatch -> 0. final_score MUST be an integer 0-3."""

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
        "The code follows the design step by step: same inputs, same decision "
        "structure, same order, same outputs. No real differences.",
        '{"dimension_findings": {"input": "match", "output": "match", "order": "match", '
        '"loop": "match", "condition": "match", "computation": "match"}, '
        '"mismatches": [], "final_score": 3}',
    ),
    2: (
        "The overall structure and steps match the design; only a minor difference "
        "remains (e.g. an operator like > vs >= or a small detail).",
        '{"dimension_findings": {"input": "match", "output": "match", "order": "match", '
        '"loop": "match", "condition": "partial", "computation": "match"}, '
        '"mismatches": ["minor operator/threshold difference"], "final_score": 2}',
    ),
    1: (
        "Parts overlap, but several steps or conditions differ from the design, "
        "so the correspondence is only weak.",
        '{"dimension_findings": {"input": "match", "output": "partial", "order": "mismatch", '
        '"loop": "match", "condition": "mismatch", "computation": "partial"}, '
        '"mismatches": ["some steps/conditions differ from the design"], "final_score": 1}',
    ),
    0: (
        "The code uses a different control-flow structure / algorithm than the "
        "design (e.g. nested pairwise comparisons instead of the design's combined "
        "conditions), even though the final result can look similar. Different "
        "structure = inconsistent.",
        '{"dimension_findings": {"input": "match", "output": "mismatch", "order": "mismatch", '
        '"loop": "mismatch", "condition": "mismatch", "computation": "mismatch"}, '
        '"mismatches": ["different decision/control-flow structure", "different algorithm"], '
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
