"""DATA-SPG-001: derive 1-3s text-speech training units from P1 character alignment.

Each unit is a contiguous token span whose mel span is in [86, 256] frames
(1-3s at hop=256, 22.05kHz). Units are traceable to the source utterance and
carry per-token mel boundaries for the alignment projection.

Splitting:
- phrases end at punctuation tokens
- phrases are split into ~target-length units at token boundaries
- units shorter than min_frames are merged with a neighbor
- no unit exceeds max_frames
"""

from __future__ import annotations

import json
import re
from dataclasses import dataclass, field
from typing import Iterable

from mini_multimodal.tts.alignment import alignment_projection

_PUNCT = set("，。！？；：、…—,.!?;:…")


def is_punct_char(char: str) -> bool:
    return char in _PUNCT or char.isspace()


@dataclass(slots=True)
class TokenSpan:
    char: str
    start_frame: int
    end_frame: int
    start_sample: int
    end_sample: int
    is_punct: bool


@dataclass(slots=True)
class Unit:
    unit_id: str
    source_utterance_id: str
    split: str
    pool: str
    text: str
    token_chars: list[str] = field(default_factory=list)
    token_bounds: list[tuple[int, int]] = field(default_factory=list)
    mel_start: int = 0
    mel_end: int = 0
    mel_frames: int = 0
    t_valid: int = 0
    token_durations: list[int] = field(default_factory=list)
    zero_duration_count: int = 0
    content_token_count: int = 0
    content_zero_duration_count: int = 0
    source_scene_id: str = ""
    source_audio_path: str = ""
    source_logmel80_path: str = ""

    @property
    def zero_duration_rate(self) -> float:
        n = len(self.token_durations)
        return self.zero_duration_count / n if n else 0.0

    @property
    def content_zero_duration_rate(self) -> float:
        return self.content_zero_duration_count / self.content_token_count if self.content_token_count else 0.0


def tokens_from_annotation(ann: dict) -> list[TokenSpan]:
    toks = []
    for a in ann.get("tokens", []):
        toks.append(
            TokenSpan(
                char=a["char"],
                start_frame=int(a["start_frame"]),
                end_frame=int(a["end_frame"]),
                start_sample=int(a.get("start_sample", a["start_frame"] * 256)),
                end_sample=int(a.get("end_sample", a["end_frame"] * 256)),
                is_punct=bool(a.get("is_punctuation", False)) or is_punct_char(a["char"]),
            )
        )
    return toks


def _split_phrase(phrase: list[TokenSpan], target: int, min_f: int, max_f: int) -> list[list[TokenSpan]]:
    """Split a phrase into units of ~target frames at token boundaries."""
    if not phrase:
        return []
    units: list[list[TokenSpan]] = []
    i, n = 0, len(phrase)
    while i < n:
        start_frame = phrase[i].start_frame
        j = i
        while j < n:
            span = phrase[j].end_frame - start_frame
            if span > max_f:
                break
            j += 1
            if span >= target:
                break
        last = max(j - 1, i)
        units.append(phrase[i : last + 1])
        i = last + 1
    return _merge_short(units, min_f, max_f)


def _merge_short(units: list[list[TokenSpan]], min_f: int, max_f: int) -> list[list[TokenSpan]]:
    """Merge units shorter than min_f with a neighbor (if it fits max_f)."""
    if not units:
        return []
    merged: list[list[TokenSpan]] = []
    for u in units:
        span = u[-1].end_frame - u[0].start_frame
        if span < min_f and merged:
            prev = merged[-1]
            combined_span = u[-1].end_frame - prev[0].start_frame
            if combined_span <= max_f:
                prev.extend(u)
                continue
        merged.append(list(u))
    # second pass: merge leading short unit with the next
    out: list[list[TokenSpan]] = []
    i = 0
    while i < len(merged):
        u = merged[i]
        span = u[-1].end_frame - u[0].start_frame
        if span < min_f and i + 1 < len(merged):
            nxt = merged[i + 1]
            combined_span = nxt[-1].end_frame - u[0].start_frame
            if combined_span <= max_f:
                out.append(u + nxt)
                i += 2
                continue
        out.append(u)
        i += 1
    return out


def derive_units_for_utterance(
    ann: dict,
    target: int = 130,
    min_f: int = 86,
    max_f: int = 256,
    t_max: int = 64,
    time_compression: int = 4,
) -> list[Unit]:
    """Derive units from one utterance's annotation."""
    toks = tokens_from_annotation(ann)
    if not toks:
        return []
    # split into phrases at punctuation
    phrases: list[list[TokenSpan]] = []
    cur: list[TokenSpan] = []
    for t in toks:
        cur.append(t)
        if t.is_punct:
            phrases.append(cur)
            cur = []
    if cur:
        phrases.append(cur)
    # split phrases into units
    raw_units: list[list[TokenSpan]] = []
    for p in phrases:
        raw_units.extend(_split_phrase(p, target, min_f, max_f))
    # build Unit objects
    scene_id = ann.get("scene_id", "")
    units: list[Unit] = []
    for idx, u in enumerate(raw_units):
        mel_start = u[0].start_frame
        mel_end = u[-1].end_frame
        mel_frames = mel_end - mel_start
        if mel_frames <= 0:
            continue
        rel_bounds = [(t.start_frame - mel_start, t.end_frame - mel_start) for t in u]
        ap = alignment_projection(rel_bounds, mel_frames, t_max, time_compression)
        text = "".join(t.char for t in u).strip()
        if not text:
            continue
        content_n = 0
        content_zd = 0
        for t, d in zip(u, ap.token_durations):
            if not t.is_punct:
                content_n += 1
                if d == 0:
                    content_zd += 1
        units.append(
            Unit(
                unit_id=f"{scene_id}#u{idx:03d}",
                source_utterance_id=scene_id,
                split="",
                pool="",
                text=text,
                token_chars=[t.char for t in u],
                token_bounds=ap.token_bounds,
                mel_start=mel_start,
                mel_end=mel_end,
                mel_frames=mel_frames,
                t_valid=ap.t_valid,
                token_durations=ap.token_durations,
                zero_duration_count=ap.zero_duration_count,
                content_token_count=content_n,
                content_zero_duration_count=content_zd,
                source_scene_id=scene_id,
            )
        )
    return units


def unit_to_dict(u: Unit) -> dict:
    return {
        "unit_id": u.unit_id,
        "source_utterance_id": u.source_utterance_id,
        "source_scene_id": u.source_scene_id,
        "split": u.split,
        "pool": u.pool,
        "text": u.text,
        "token_chars": u.token_chars,
        "token_bounds": [list(b) for b in u.token_bounds],
        "mel_start": u.mel_start,
        "mel_end": u.mel_end,
        "mel_frames": u.mel_frames,
        "t_valid": u.t_valid,
        "token_durations": u.token_durations,
        "zero_duration_count": u.zero_duration_count,
        "zero_duration_rate": u.zero_duration_rate,
        "content_token_count": u.content_token_count,
        "content_zero_duration_count": u.content_zero_duration_count,
        "content_zero_duration_rate": u.content_zero_duration_rate,
        "source_audio_path": u.source_audio_path,
        "source_logmel80_path": u.source_logmel80_path,
    }


def load_annotations(path: str) -> list[dict]:
    out = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                out.append(json.loads(line))
    return out
