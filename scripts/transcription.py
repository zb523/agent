from __future__ import annotations
import re
from typing import List, Optional, Tuple

# ---------------------------
# Regex constants (unchanged semantics)
# ---------------------------

RE_STRICT_EOS = re.compile(r"(?:(?<![0-9٠-٩])\.(?![0-9٠-٩])|[!\?؟۔])")
RE_RELAXED_TAIL = re.compile(r"[،,]")

# ---------------------------
# Light Arabic helpers (moved as-is)
# ---------------------------

def _normalize_arabic(text: str) -> str:
    """Light Arabic normalization: remove diacritics & tatweel, unify alef & yeh,
    drop non-Arabic punctuation, collapse spaces."""
    if not text:
        return ""
    text = re.sub("\u0640", "", text)
    text = re.sub("[\u064B-\u065F\u0670]", "", text)
    text = re.sub("[\u0622\u0623\u0625\u0671]", "\u0627", text)
    text = re.sub("[\u0649]", "\u064A", text)
    text = re.sub(r"[^\u0600-\u06FF\s]", " ", text)
    text = re.sub(r"\s+", " ", text).strip()
    return text

def _tokenize_arabic(text: str) -> list[str]:
    norm = _normalize_arabic(text)
    return [t for t in norm.split(" ") if t]

# ---------------------------
# Normalization helpers
# ---------------------------

def normalize_spacing(s: str) -> str:
    if not s:
        return s
    s = re.sub(r"\s+([\.!\?؟۔،؛…:])", r"\1", s)
    s = re.sub(r"\s+", " ", s).strip()
    return s

def current_buffer_text(tokens: List[str]) -> str:
    return normalize_spacing(" ".join(tokens))

# ---------------------------
# Index finders (pure)
# ---------------------------

def find_strict_eos_idx(tokens: List[str], re_strict=RE_STRICT_EOS) -> Optional[int]:
    """Return the token index up to the last strict EOS, else None."""
    if not tokens:
        return None
    ends: List[int] = []
    pieces: List[str] = []
    for tok in tokens:
        if pieces:
            pieces.append(" ")
        pieces.append(tok)
        ends.append(len("".join(pieces)))
    s = "".join(pieces)
    last_end_char = None
    for m in re_strict.finditer(s):
        last_end_char = m.end()
    if last_end_char is None:
        return None
    for i, end in enumerate(ends):
        if end >= last_end_char:
            return i + 1
    return None

def find_relaxed_tail_idx(tokens: List[str], re_relaxed=RE_RELAXED_TAIL) -> Optional[int]:
    """Return the token index up to and including the last relaxed punctuation (comma), else None."""
    if not tokens:
        return None
    ends: List[int] = []
    pieces: List[str] = []
    for tok in tokens:
        if pieces:
            pieces.append(" ")
        pieces.append(tok)
        ends.append(len("".join(pieces)))
    s = "".join(pieces)
    last_end_char = None
    for m in re_relaxed.finditer(s):
        last_end_char = m.end()
    if last_end_char is None:
        return None
    for i, end in enumerate(ends):
        if end >= last_end_char:
            return i + 1
    return None

def smart_cut(tokens: List[str], min_words: int, max_words: int) -> int:
    """Prefer last relaxed punctuation; else cut within [min_words, max_words]."""
    relaxed_idx = find_relaxed_tail_idx(tokens)
    if relaxed_idx is not None and relaxed_idx >= min_words:
        return relaxed_idx
    return max(min_words, min(len(tokens), max_words))

# ---------------------------
# Commit brain (pure)
# ---------------------------

def _char_precise_split(tokens: List[str], re_pattern=RE_STRICT_EOS) -> Tuple[List[str], List[str]]:
    """Split exactly at the last punctuation character; keep punctuation with left side."""
    if not tokens:
        return [], []
    # Build mapping from tokens → char offsets in the joined string
    ends: List[int] = []
    starts: List[int] = []
    pieces: List[str] = []
    for tok in tokens:
        if pieces:
            pieces.append(" ")
        starts.append(len("".join(pieces)))
        pieces.append(tok)
        ends.append(len("".join(pieces)))
    s = "".join(pieces)
    last_end_char = None
    for m in re_pattern.finditer(s):
        last_end_char = m.end()
    if last_end_char is None:
        return tokens[:], []
    tok_idx = None
    for i, end in enumerate(ends):
        if end >= last_end_char:
            tok_idx = i
            break
    if tok_idx is None:
        return tokens[:], []
    token_text = tokens[tok_idx]
    token_start = starts[tok_idx]
    cut_in_token = last_end_char - token_start
    cut_in_token = max(0, min(len(token_text), cut_in_token))
    left_chunk = token_text[:cut_in_token].rstrip()
    right_chunk = token_text[cut_in_token:].lstrip()
    pre_tokens = tokens[:tok_idx]
    post_tokens = tokens[tok_idx + 1:]
    committed_tokens_cp = pre_tokens + ([left_chunk] if left_chunk else [])
    carry_tokens_cp = ([right_chunk] if right_chunk else []) + post_tokens
    return committed_tokens_cp, carry_tokens_cp

def decide_commit_split(
    tokens: List[str],
    idx: int,
    reason: str,
    min_words: int,
) -> Tuple[List[str], List[str], bool]:
    """
    Pure decision function mirroring Assistant._commit_at_index rules:
    - Enforces MIN_WORDS gating with elastic reasons
    - Performs char-precise split for HARD_PUNCT
    - Returns (commit_tokens, carry_tokens, allowed)
    """
    if idx <= 0 or idx > len(tokens):
        return [], tokens, False

    committed_tokens: List[str]
    carry_tokens: List[str]

    # HARD_PUNCT → char-precise split at last strict EOS
    # SOFT_COMMA_MAX → char-precise split at last comma
    committed_tokens_cp: Optional[List[str]] = None
    carry_tokens_cp: Optional[List[str]] = None
    if reason == "HARD_PUNCT" and tokens:
        committed_tokens_cp, carry_tokens_cp = _char_precise_split(tokens, RE_STRICT_EOS)
    elif reason == "SOFT_COMMA_MAX" and tokens:
        committed_tokens_cp, carry_tokens_cp = _char_precise_split(tokens, RE_RELAXED_TAIL)

    total_tokens = len(tokens)
    if idx < min_words:
        elastic_reasons = {"SOFT_COMMA_MAX", "MAX_WORDS", "STALL", "TTL_COMMA", "TTL_STALL"}
        if reason in elastic_reasons:
            if total_tokens >= min_words:
                idx = min_words
            else:
                return [], tokens, False
        else:
            return [], tokens, False

    if committed_tokens_cp is not None and carry_tokens_cp is not None:
        committed_tokens = committed_tokens_cp
        carry_tokens = carry_tokens_cp
    else:
        committed_tokens = tokens[:idx]
        carry_tokens = tokens[idx:]

    return committed_tokens, carry_tokens, True