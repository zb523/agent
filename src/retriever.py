"""
Quran detection and translation retrieval.

Ported from legacy/quran_matcher.py with additional translation helpers.
"""
from __future__ import annotations

import json
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Set, Tuple

# Optional RapidFuzz scoring
try:
    from rapidfuzz.fuzz import partial_ratio, token_set_ratio  # type: ignore
    _HAS_RAPIDFUZZ = True
except Exception:  # pragma: no cover
    import difflib

    def partial_ratio(a: str, b: str) -> float:  # crude fallback
        a = a or ""
        b = b or ""
        best = 0.0
        step = max(1, len(a) // 20)
        for i in range(0, max(1, len(b) - 1), step):
            sub = b[i : i + len(a)]
            sm = difflib.SequenceMatcher(None, a, sub)
            best = max(best, sm.ratio())
        return best * 100.0

    def token_set_ratio(a: str, b: str) -> float:  # simple token-set Jaccard proxy
        ta = set((a or "").split())
        tb = set((b or "").split())
        if not ta or not tb:
            return 0.0
        inter = len(ta & tb)
        union = len(ta | tb)
        return (inter / union) * 100.0

    _HAS_RAPIDFUZZ = False


# =============================================================================
# Normalization
# =============================================================================

_DIACRITICS = re.compile(r"[\u0610-\u061A\u064B-\u065F\u06D6-\u06ED\u0640]")
_PUNCT = re.compile(r"[\u0600-\u0605\u061B\u061F\u066A-\u066D\u06D4\.,;:!؟،\-\(\)\[\]\{\}\"']")


def normalize_arabic_strict(s: str) -> str:
    if not s:
        return ""
    s = s.replace("\u0670", "ا")  # small alef → alif
    s = _DIACRITICS.sub("", s)
    s = (
        s.replace("ٱ", "ا")
        .replace("أ", "ا")
        .replace("إ", "ا")
        .replace("آ", "ا")
        .replace("ى", "ي")
        .replace("ة", "ه")
        .replace("ؤ", "و")
        .replace("ئ", "ي")
        .replace("ﻻ", "لا")
    )
    s = _PUNCT.sub(" ", s)
    s = re.sub(r"\s+", " ", s).strip()
    return s


def tokenize(s: str) -> List[str]:
    return [t for t in s.split() if t]


def char_trigrams(s: str) -> Set[str]:
    s2 = s.replace(" ", "")
    if len(s2) < 3:
        return set()
    return {s2[i : i + 3] for i in range(0, len(s2) - 2)}


# =============================================================================
# Data classes
# =============================================================================

@dataclass
class Verse:
    ref: str  # "S:A"
    surah: int
    ayah: int
    text: str
    norm: str
    trigrams: Set[str]


@dataclass
class Detection:
    ref: str
    is_full: bool
    confidence: float
    pos: int  # earliest window start token index where this ref was observed


# =============================================================================
# QuranIndex - trigram-based fuzzy matching
# =============================================================================

class QuranIndex:
    def __init__(self) -> None:
        self.verses: Dict[str, Verse] = {}
        self.postings: Dict[str, Set[str]] = {}  # trigram -> {ref}

    @staticmethod
    def from_ar_json(path: Path | str) -> "QuranIndex":
        data = json.loads(Path(path).read_text(encoding="utf-8"))
        idx = QuranIndex()
        for ref, v in data.items():
            ar = str(v.get("text", ""))
            try:
                surah = int(v.get("surah") or int(str(ref).split(":")[0]))
                ayah = int(v.get("ayah") or int(str(ref).split(":")[1]))
            except Exception:
                parts = str(ref).split(":")
                surah = int(parts[0])
                ayah = int(parts[1])
            norm = normalize_arabic_strict(ar)
            tris = char_trigrams(norm)
            verse = Verse(ref=ref, surah=surah, ayah=ayah, text=ar, norm=norm, trigrams=tris)
            idx.verses[ref] = verse
            for tri in tris:
                idx.postings.setdefault(tri, set()).add(ref)
        return idx

    def candidates_for_window(self, win_norm: str, topk: int = 25) -> List[Tuple[str, float]]:
        tri = char_trigrams(win_norm)
        if not tri:
            return []
        cand_refs: Set[str] = set()
        for t in tri:
            cand_refs |= self.postings.get(t, set())
        if not cand_refs:
            return []
        scored: List[Tuple[str, float]] = []
        tri_len = len(tri)
        for ref in cand_refs:
            v = self.verses[ref]
            inter = len(tri & v.trigrams)
            union = tri_len + len(v.trigrams) - inter
            j = (inter / union) if union else 0.0
            if j > 0:
                scored.append((ref, j))
        scored.sort(key=lambda x: x[1], reverse=True)
        return scored[:topk]


# =============================================================================
# Detector - fuzzy matching
# =============================================================================

def _fuzzy_confidence(win_norm: str, verse_norm: str, jaccard: float) -> float:
    # RapidFuzz returns 0..100; convert to 0..1
    pr = partial_ratio(win_norm, verse_norm) / 100.0
    ts = token_set_ratio(win_norm, verse_norm) / 100.0
    cov = min(1.0, (len(win_norm.replace(" ", "")) + 1e-6) / (len(verse_norm.replace(" ", "")) + 1e-6))
    return 0.40 * pr + 0.35 * ts + 0.20 * jaccard + 0.05 * cov


def detect_refs(
    idx: QuranIndex,
    snippet_ar: str,
    *,
    min_tokens: int = 4,
    window_sizes: Sequence[int] = (4, 6, 8, 10, 12, 16, 20, 24),
    stride_tokens: int = 2,
    topk_per_window: int = 12,
    conf_strict: float = 0.90,
    conf_loose: float = 0.84,
    jaccard_loose: float = 0.45,
) -> List[Detection]:
    """Return ordered detections for the entire snippet (no clause splitting)."""
    norm = normalize_arabic_strict(snippet_ar)
    toks = tokenize(norm)
    if len(toks) < min_tokens:
        return []
    seen_best: Dict[str, Detection] = {}
    max_w = min(max(window_sizes) if window_sizes else 24, len(toks))
    sizes = [w for w in window_sizes if w >= min_tokens]
    if not sizes:
        sizes = [min_tokens]
    sizes = [min(w, max_w) for w in sizes]
    for start in range(0, len(toks) - min_tokens + 1, max(1, stride_tokens)):
        for w in sizes:
            end = start + w
            if end > len(toks):
                break
            win_norm = " ".join(toks[start:end])
            cands = idx.candidates_for_window(win_norm, topk=topk_per_window)
            for ref, j in cands:
                verse_norm = idx.verses[ref].norm
                conf = _fuzzy_confidence(win_norm, verse_norm, j)
                accept = (conf >= conf_strict) or (conf_loose <= conf < conf_strict and j >= jaccard_loose)
                if not accept:
                    continue
                det = seen_best.get(ref)
                if det is None or conf > det.confidence:
                    seen_best[ref] = Detection(ref=ref, is_full=(len(verse_norm) <= len(win_norm) + 2), confidence=conf, pos=start)
    if not seen_best:
        return []
    out = sorted(seen_best.values(), key=lambda d: (d.pos, -d.confidence))
    # Merge contiguous refs within same surah into ranges if they are adjacent
    merged: List[Detection] = []
    def _parse(ref: str) -> Tuple[int, int, int]:
        s, ab = ref.split(":", 1)
        if '-' in ab:
            a0, a1 = ab.split('-', 1)
            return int(s), int(a0), int(a1)
        return int(s), int(ab), int(ab)
    for d in out:
        if merged:
            ls, la0, la1 = _parse(merged[-1].ref)
            rs, ra0, ra1 = _parse(d.ref)
            if ls == rs and (la1 + 1) == ra0:
                merged[-1].ref = f"{ls}:{la0}-{ra1}"
                merged[-1].confidence = max(merged[-1].confidence, d.confidence)
                merged[-1].is_full = merged[-1].is_full and d.is_full
                continue
        merged.append(d)
    return merged


# =============================================================================
# DirectIndex - token-based deterministic matching
# =============================================================================

def _merge_proclitics(tokens: List[str]) -> List[str]:
    if not tokens:
        return tokens
    out: List[str] = []
    i = 0
    while i < len(tokens):
        t = tokens[i]
        if t in {"و", "ف", "ب", "ك", "ل", "س"} and (i + 1) < len(tokens):
            out.append(t + tokens[i + 1])
            i += 2
            continue
        out.append(t)
        i += 1
    return out


def _dedupe_stutters(tokens: List[str]) -> List[str]:
    out: List[str] = []
    prev: Optional[str] = None
    for t in tokens:
        if t == prev:
            continue
        out.append(t)
        prev = t
    return out


def _eq_token(a: str, b: str) -> bool:
    if a == b:
        return True
    if a.endswith("و") and (a + "ا") == b:
        return True
    if b.endswith("و") and (b + "ا") == a:
        return True
    return False


class DirectIndex:
    def __init__(self) -> None:
        self.surah_tokens: Dict[int, List[str]] = {}
        self.surah_verse_spans: Dict[int, List[Tuple[int, int, int]]] = {}
        self.bigram_index: Dict[Tuple[str, str], List[Tuple[int, int]]] = {}

    @staticmethod
    def build_from_qindex(qidx: QuranIndex) -> "DirectIndex":
        d = DirectIndex()
        by_surah: Dict[int, List[Tuple[int, Verse]]] = {}
        for ref, verse in qidx.verses.items():
            by_surah.setdefault(verse.surah, []).append((verse.ayah, verse))
        for sid, items in by_surah.items():
            items.sort(key=lambda x: x[0])
            toks: List[str] = []
            spans: List[Tuple[int, int, int]] = []
            for ay, v in items:
                vtoks = tokenize(v.norm)
                start = len(toks)
                toks.extend(vtoks)
                end = len(toks)
                spans.append((ay, start, end))
            d.surah_tokens[sid] = toks
            d.surah_verse_spans[sid] = spans
            for i in range(len(toks) - 1):
                bg = (toks[i], toks[i + 1])
                d.bigram_index.setdefault(bg, []).append((sid, i))
        return d


def detect_refs_direct(
    didx: DirectIndex,
    snippet_ar: str,
    *,
    min_tokens: int = 4,
) -> List[Detection]:
    norm = normalize_arabic_strict(snippet_ar)
    itoks_raw = tokenize(norm)
    itoks = _dedupe_stutters(_merge_proclitics(itoks_raw))
    if len(itoks) < min_tokens:
        return []
    out: List[Detection] = []
    i = 0
    while i < len(itoks):
        if i + 1 >= len(itoks):
            break
        cands = None
        for a in (itoks[i],):
            for b in (itoks[i + 1],):
                cands = didx.bigram_index.get((a, b))
                if cands:
                    break
            if cands:
                break
        best = None  # (sid, j, k)
        if cands:
            for sid, j in cands:
                stoks = didx.surah_tokens.get(sid, [])
                k = 0
                while (i + k) < len(itoks) and (j + k) < len(stoks) and _eq_token(itoks[i + k], stoks[j + k]):
                    k += 1
                if k >= min_tokens and (best is None or k > best[2]):
                    best = (sid, j, k)
        if best is None:
            i += 1
            continue
        sid, j, k = best
        v_spans = didx.surah_verse_spans.get(sid, [])
        v_start_id = v_spans[0][0] if v_spans else 1
        v_end_id = v_spans[-1][0] if v_spans else v_start_id
        for vid, a, b in v_spans:
            if a <= j < b:
                v_start_id = vid
                break
        for vid, a, b in v_spans:
            if a <= (j + k - 1) < b:
                v_end_id = vid
                break
        ref = f"{sid}:{v_start_id}" if v_start_id == v_end_id else f"{sid}:{v_start_id}-{v_end_id}"
        conf = min(0.99, 0.80 + 0.02 * max(0, k - min_tokens))
        out.append(Detection(ref=ref, is_full=False, confidence=conf, pos=i))
        i += k
    out.sort(key=lambda d: (d.pos, -d.confidence))
    merged: List[Detection] = []
    def _parse(ref: str) -> Tuple[int, int, int]:
        if '-' in ref:
            s, rng = ref.split(':'); a0,a1 = map(int, rng.split('-')); return int(s), a0, a1
        s,a = ref.split(':'); return int(s), int(a), int(a)
    for d in out:
        if merged:
            s0,a0,b0 = _parse(merged[-1].ref)
            s1,a1,b1 = _parse(d.ref)
            if s0==s1 and b0+1==a1:
                merged[-1].ref = f"{s0}:{a0}-{b1}"
                merged[-1].confidence = max(merged[-1].confidence, d.confidence)
                continue
        merged.append(d)
    return merged


def combine_detections(a: List[Detection], b: List[Detection]) -> List[Detection]:
    by_ref: Dict[str, Detection] = {}
    for d in a + b:
        e = by_ref.get(d.ref)
        if e is None or (d.pos, d.confidence) < (e.pos, e.confidence):
            by_ref[d.ref] = d
    out = sorted(by_ref.values(), key=lambda d: (d.pos, -d.confidence))
    merged: List[Detection] = []
    def _parse(ref: str) -> Tuple[int,int,int]:
        if '-' in ref:
            s,r=ref.split(':');a0,a1=map(int,r.split('-'));return int(s),a0,a1
        s,a=ref.split(':');return int(s),int(a),int(a)
    for d in out:
        if merged:
            s0,a0,b0=_parse(merged[-1].ref); s1,a1,b1=_parse(d.ref)
            if s0==s1 and b0+1==a1:
                merged[-1].ref=f"{s0}:{a0}-{b1}"
                merged[-1].confidence=max(merged[-1].confidence,d.confidence)
                continue
        merged.append(d)
    return merged


def expand_ref_range(ref: str) -> List[str]:
    """Expand a ref like '2:255-256' into ['2:255', '2:256']."""
    if ":" not in ref:
        return []
    s, ab = ref.split(":", 1)
    if '-' in ab:
        a0, a1 = ab.split('-', 1)
        return [f"{int(s)}:{a}" for a in range(int(a0), int(a1) + 1)]
    return [f"{int(s)}:{int(ab)}"]


# =============================================================================
# Translation loading and lookup
# =============================================================================

def load_quran_index(path: str = "jsons/Quran.json") -> QuranIndex:
    """Load QuranIndex from the Arabic Quran JSON."""
    return QuranIndex.from_ar_json(path)


def load_ar_map(path: str = "jsons/Quran.json") -> Dict[str, Dict]:
    """Load the raw Arabic Quran map for canonical text lookup."""
    return json.loads(Path(path).read_text(encoding="utf-8"))


def load_translation(lang_code: str, jsons_dir: str = "jsons") -> Dict[str, str]:
    """
    Load translation map for a language.
    Returns {ref: translation_text}.
    
    JSON format expected: {"ref": {"t": "translation"}} or {"ref": "translation"}
    """
    path = Path(jsons_dir) / f"{lang_code}.json"
    if not path.exists():
        return {}
    data = json.loads(path.read_text(encoding="utf-8"))
    result: Dict[str, str] = {}
    for ref, v in data.items():
        if isinstance(v, dict):
            result[ref] = str(v.get("t", ""))
        else:
            result[ref] = str(v)
    return result


def get_canonical_texts(
    ref: str,
    ar_map: Dict[str, Dict],
    lang_map: Dict[str, str],
) -> Tuple[str, str]:
    """
    Get canonical Arabic and translation texts for a reference.
    Handles ranges like '2:255-256'.
    Returns (ar_text, tl_text).
    """
    keys = expand_ref_range(ref)
    ar_parts = []
    tl_parts = []
    for k in keys:
        ar_entry = ar_map.get(k, {})
        ar_text = ar_entry.get("text", "") if isinstance(ar_entry, dict) else ""
        ar_parts.append(str(ar_text))
        tl_parts.append(str(lang_map.get(k, "")))
    return " ".join(ar_parts), " ".join(tl_parts)


def get_translation_for_ref(
    ref: str,
    lang_map: Dict[str, str],
) -> Tuple[str, bool]:
    """
    Get translation for a ref. Returns (text, ok).
    If any verse in a range is missing, returns ("", False).
    """
    keys = expand_ref_range(ref)
    parts: List[str] = []
    for k in keys:
        t = lang_map.get(k)
        if not t:
            return "", False
        parts.append(str(t))
    return " ".join(parts), True

