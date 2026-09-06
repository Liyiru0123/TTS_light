"""Mel -> latent alignment projection (nominal-bin convention, route A).

This is a deterministic, monotonic, reproducible projection convention, NOT a
recovery of the convolution encoder's true receptive-field ownership.

The P3-W02 encoder downsamples the time axis by 4 (256 -> 128 -> 64), so latent
frame j is assigned a nominal 4-frame temporal bin [4j, 4(j+1)) with center
c_j = 4j + 2 (mel frame coordinate). The frequency-axis 4x compression is NOT
used for time alignment.

Properties:
- deterministic, monotonic
- each valid latent frame belongs to exactly one token
- sum_i d_i = T_valid
- d_i = 0 is a representation-resolution loss (no latent center falls in the
  token's mel range), NOT an alignment error
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Sequence


def latent_center(j: int, time_compression: int = 4) -> int:
    """Nominal mel-frame center of latent frame j.

    c_j = time_compression * j + time_compression // 2
    For time_compression=4: c_j = 4j + 2.
    """
    if time_compression % 2 != 0:
        raise ValueError("time_compression must be even for a nominal center")
    return time_compression * j + time_compression // 2


def valid_latent_frames(
    mel_frames: int, t_max: int = 64, time_compression: int = 4
) -> list[int]:
    """Valid latent frame indices for a unit with `mel_frames` valid mel frames.

    J_valid = { j : 0 <= j < t_max, c_j < mel_frames }
    Returns the list of valid latent frame indices (a prefix [0, T_valid)).
    """
    if mel_frames <= 0:
        return []
    out: list[int] = []
    for j in range(t_max):
        if latent_center(j, time_compression) < mel_frames:
            out.append(j)
        else:
            break
    return out


def token_latent_durations(
    token_mel_bounds: Sequence[tuple[int, int]],
    mel_frames: int,
    t_max: int = 64,
    time_compression: int = 4,
) -> list[int]:
    """Per-token latent duration d_i.

    token_mel_bounds: list of (start, end) relative mel boundaries per token,
        where the token owns latent frame j iff c_j in [start, end).
    mel_frames: total valid mel length (L_mel) of the unit.

    Returns d_i for each token, with sum(d_i) == T_valid.
    """
    valid_js = valid_latent_frames(mel_frames, t_max, time_compression)
    durations = [0] * len(token_mel_bounds)
    for j in valid_js:
        c = latent_center(j, time_compression)
        assigned = False
        for i, (start, end) in enumerate(token_mel_bounds):
            if start <= c < end:
                durations[i] += 1
                assigned = True
                break
        if not assigned:
            # c is beyond all token bounds (should not happen if mel_frames is
            # the last token's end boundary); assign to the last token to keep
            # sum(d_i) == T_valid.
            if durations:
                durations[-1] += 1
    return durations


@dataclass(slots=True)
class AlignmentProjection:
    """Result of projecting a unit's mel alignment onto the latent grid."""

    mel_frames: int
    t_valid: int
    token_bounds: list[tuple[int, int]]
    token_durations: list[int]
    zero_duration_count: int

    @property
    def zero_duration_rate(self) -> float:
        if not self.token_durations:
            return 0.0
        return self.zero_duration_count / len(self.token_durations)


def alignment_projection(
    token_mel_bounds: Sequence[tuple[int, int]],
    mel_frames: int,
    t_max: int = 64,
    time_compression: int = 4,
) -> AlignmentProjection:
    """Compute the alignment projection for one unit.

    token_mel_bounds: relative (start, end) mel boundaries per token.
    mel_frames: valid mel length of the unit (L_mel).
    """
    if mel_frames <= 0:
        raise ValueError("mel_frames must be positive")
    if not token_mel_bounds:
        raise ValueError("token_mel_bounds must be non-empty")
    # sanity: bounds within [0, mel_frames]; start==end allowed (zero-width token, d_i=0)
    for start, end in token_mel_bounds:
        if start < 0 or end > mel_frames or start > end:
            raise ValueError(f"invalid token bound ({start},{end}) for mel_frames={mel_frames}")
    durations = token_latent_durations(token_mel_bounds, mel_frames, t_max, time_compression)
    t_valid = valid_latent_frames(mel_frames, t_max, time_compression)
    zero = sum(1 for d in durations if d == 0)
    return AlignmentProjection(
        mel_frames=mel_frames,
        t_valid=len(t_valid),
        token_bounds=[tuple(b) for b in token_mel_bounds],
        token_durations=durations,
        zero_duration_count=zero,
    )
