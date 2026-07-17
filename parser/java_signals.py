"""Lightweight structural extraction from Java source.

Two jobs:
  1. `describe_java(code)` -> a short human-readable hint string that we inject
     into the prompt so the LLM grounds its comparison ("Detected in Java: ...").
  2. `JavaSignals` -> a struct used by the rule-based fallback scorer when the
     LLM output cannot be parsed at all.

We prefer the `javalang` parser when available (accurate), and fall back to
regex heuristics that are good enough for the basic programming exercises in
this competition. Nothing here needs a GPU or network.
"""
from __future__ import annotations

import re
from dataclasses import dataclass, asdict

try:  # optional, more accurate; regex fallback covers the no-dependency case
    import javalang  # type: ignore

    _HAS_JAVALANG = True
except Exception:  # pragma: no cover - javalang not installed
    _HAS_JAVALANG = False


@dataclass
class JavaSignals:
    n_inputs: int = 0
    n_loops: int = 0
    n_branches: int = 0
    n_outputs: int = 0
    n_arith: int = 0
    has_array: bool = False
    has_method_call: bool = False

    def as_dict(self) -> dict:
        return asdict(self)


# --------------------------------------------------------------------------- #
# Regex heuristics (dependency-free path)
# --------------------------------------------------------------------------- #
# Count actual input *reads* (a call with parens), not the Scanner/BufferedReader
# class declarations -- otherwise every program over-counts its inputs.
_INPUT_RE = re.compile(r"\b(?:next(?:Int|Double|Line|Long|Float|Boolean|Short|Byte)?|readLine|nextToken)\s*\(")
_FOR_WHILE_RE = re.compile(r"\b(for|while|do)\b")
_IF_RE = re.compile(r"\b(if|else\s+if|switch|case)\b|\?\s*[^:]+:")  # if/switch/ternary
_OUTPUT_RE = re.compile(r"System\.out\.(print(?:ln|f)?)")
_ARITH_RE = re.compile(r"[+\-*/%](?![=+\-*/])|(?:\+=|-=|\*=|/=|%=)")
# Real array/list usage: allocation, indexing (a[i]), or a collection type.
# Deliberately does NOT match the empty `String[]` in `main(String[] args)`.
_ARRAY_RE = re.compile(r"\bnew\s+\w+\s*\[|\b\w+\s*\[\s*\w+\s*\]|\bArrayList\b|\bList<")


def _strip_comments_and_strings(code: str) -> str:
    """Remove comments and string/char literals so operators inside them do not
    inflate the arithmetic count."""
    code = re.sub(r"//[^\n]*", " ", code)
    code = re.sub(r"/\*.*?\*/", " ", code, flags=re.DOTALL)
    code = re.sub(r'"(?:\\.|[^"\\])*"', '""', code)
    code = re.sub(r"'(?:\\.|[^'\\])'", "''", code)
    return code


def _regex_signals(code: str) -> JavaSignals:
    clean = _strip_comments_and_strings(code)
    return JavaSignals(
        n_inputs=len(_INPUT_RE.findall(clean)),
        n_loops=len(_FOR_WHILE_RE.findall(clean)),
        n_branches=len(_IF_RE.findall(clean)),
        n_outputs=len(_OUTPUT_RE.findall(clean)),
        n_arith=len(_ARITH_RE.findall(clean)),
        has_array=bool(_ARRAY_RE.search(clean)),
        has_method_call=bool(re.search(r"\b\w+\s*\([^)]*\)\s*;", clean)),
    )


def _javalang_signals(code: str) -> JavaSignals:
    """More accurate counts via a real AST when javalang is installed."""
    try:
        tree = javalang.parse.parse(code)
    except Exception:
        return _regex_signals(code)

    sig = JavaSignals()
    for _, node in tree:
        name = type(node).__name__
        if name in ("ForStatement", "WhileStatement", "DoStatement", "ForControl", "EnhancedForControl"):
            if name.endswith("Statement"):
                sig.n_loops += 1
        elif name in ("IfStatement", "SwitchStatement", "TernaryExpression"):
            sig.n_branches += 1
        elif name == "MethodInvocation":
            member = getattr(node, "member", "") or ""
            qualifier = getattr(node, "qualifier", "") or ""
            if member.startswith("next") or member in ("readLine", "read"):
                sig.n_inputs += 1
            if member.startswith("print") and "out" in str(qualifier):
                sig.n_outputs += 1
            sig.has_method_call = True
        elif name == "BinaryOperation" and getattr(node, "operator", "") in ("+", "-", "*", "/", "%"):
            sig.n_arith += 1
        elif name in ("ArrayCreator", "ArrayReference"):
            sig.has_array = True

    # javalang sometimes misses Scanner-style input; backstop with regex.
    if sig.n_inputs == 0:
        sig.n_inputs = len(_INPUT_RE.findall(_strip_comments_and_strings(code)))
    return sig


def extract_java_signals(code: str) -> JavaSignals:
    if _HAS_JAVALANG:
        return _javalang_signals(code)
    return _regex_signals(code)


def describe_java(code: str) -> str:
    """One-line grounding hint for the prompt."""
    s = extract_java_signals(code)
    parts = [
        f"{s.n_inputs} input read(s)",
        f"{s.n_loops} loop(s)",
        f"{s.n_branches} branch/condition(s)",
        f"{s.n_outputs} print/output(s)",
        f"{s.n_arith} arithmetic operation(s)",
    ]
    if s.has_array:
        parts.append("uses an array/list")
    return "Detected in Java: " + ", ".join(parts) + "."


def fallback_score_from_signals(java_code: str, design_text: str) -> int:
    """Very rough estimate used ONLY when the LLM output is unparseable.

    Compares the coarse control-flow shape of the Java against keywords found in
    the design artifact (pseudocode text, or OCR text from the flowchart). This
    is deliberately conservative and never the primary signal.
    """
    js = extract_java_signals(java_code)
    d = (design_text or "").lower()

    design_has_loop = any(k in d for k in ("for", "while", "repeat", "loop", "วน", "ซ้ำ"))
    design_has_branch = any(k in d for k in ("if", "else", "condition", "เงื่อนไข", "ถ้า"))
    design_has_input = any(k in d for k in ("input", "read", "scan", "รับ", "อ่าน"))
    design_has_output = any(k in d for k in ("print", "output", "display", "แสดง", "พิมพ์"))

    checks = [
        (js.n_loops > 0) == design_has_loop,
        (js.n_branches > 0) == design_has_branch,
        (js.n_inputs > 0) == design_has_input,
        (js.n_outputs > 0) == design_has_output,
    ]
    matches = sum(checks)
    # Map 0..4 structural agreements onto the 0..3 rubric.
    return {0: 0, 1: 0, 2: 1, 3: 2, 4: 3}[matches]
