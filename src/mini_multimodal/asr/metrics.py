"""Transparent CTC decoding, alignment spans, CER and domain-token WER."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Sequence

import torch


def _edit_distance(reference: Sequence, hypothesis: Sequence) -> int:
    previous = list(range(len(hypothesis) + 1))
    for i, ref in enumerate(reference, 1):
        current = [i]
        for j, hyp in enumerate(hypothesis, 1):
            current.append(min(current[-1] + 1, previous[j] + 1, previous[j - 1] + (ref != hyp)))
        previous = current
    return previous[-1]


_DOMAIN_LEXICON = tuple(sorted((
    "第一帧", "可见对象", "对象", "中等大小", "位于", "左上区域", "右上区域", "左下区域", "右下区域",
    "左侧区域", "右侧区域", "上方区域", "下方区域", "中央区域", "正方形", "三角形", "五边形", "矩形", "圆形",
    "红色", "蓝色", "绿色", "黄色", "紫色", "橙色", "一个", "二个", "三个", "是", "有", "的", "小", "大",
), key=len, reverse=True))


def domain_words(text: str) -> list[str]:
    """Deterministic longest-match segmentation for this closed teaching corpus."""
    result, index = [], 0
    while index < len(text):
        token = next((item for item in _DOMAIN_LEXICON if text.startswith(item, index)), text[index])
        result.append(token)
        index += len(token)
    return result


def ctc_greedy_decode(logits: torch.Tensor, lengths: torch.Tensor, tokenizer) -> list[str]:
    paths = logits.argmax(dim=-1).cpu()
    results = []
    for path, length in zip(paths, lengths.cpu().tolist()):
        collapsed, previous = [], None
        for token_id in path[:length].tolist():
            if token_id != previous and token_id != tokenizer.blank_id:
                collapsed.append(token_id)
            previous = token_id
        results.append(tokenizer.decode(collapsed))
    return results


def forced_greedy_spans(logits: torch.Tensor, length: int, tokenizer, frame_stride_seconds: float = 256 / 22050 * 8) -> list[dict]:
    """Return observable greedy-path spans; this is not forced alignment."""
    path = logits[:length].argmax(dim=-1).cpu().tolist()
    spans, start, previous = [], 0, path[0] if path else tokenizer.blank_id
    for index, token_id in enumerate(path[1:], 1):
        if token_id != previous:
            if previous != tokenizer.blank_id:
                spans.append({"token": tokenizer.tokens[previous], "start_step": start, "end_step": index, "start_seconds": start * frame_stride_seconds, "end_seconds": index * frame_stride_seconds})
            start, previous = index, token_id
    if path and previous != tokenizer.blank_id:
        spans.append({"token": tokenizer.tokens[previous], "start_step": start, "end_step": len(path), "start_seconds": start * frame_stride_seconds, "end_seconds": len(path) * frame_stride_seconds})
    return spans


@dataclass
class ASRMetricAccumulator:
    character_edits: int = 0
    characters: int = 0
    word_edits: int = 0
    words: int = 0
    exact: int = 0
    utterances: int = 0

    def update(self, references: Sequence[str], hypotheses: Sequence[str]) -> None:
        for reference, hypothesis in zip(references, hypotheses):
            ref_words, hyp_words = domain_words(reference), domain_words(hypothesis)
            self.character_edits += _edit_distance(reference, hypothesis)
            self.characters += len(reference)
            self.word_edits += _edit_distance(ref_words, hyp_words)
            self.words += len(ref_words)
            self.exact += reference == hypothesis
            self.utterances += 1

    def compute(self) -> dict:
        return {
            "cer": self.character_edits / max(1, self.characters),
            "domain_wer": self.word_edits / max(1, self.words),
            "exact_match_rate": self.exact / max(1, self.utterances),
            "utterance_count": self.utterances,
            "character_edits": self.character_edits,
            "reference_characters": self.characters,
            "domain_word_edits": self.word_edits,
            "reference_domain_words": self.words,
        }
