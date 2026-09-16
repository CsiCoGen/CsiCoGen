"""Typed schedule metadata for CsiCoGen-Turbo experiments."""

from __future__ import annotations

import math
import re
from dataclasses import dataclass

from modules.codec import (
    build_macro_sampling_plan,
    count_macro_refresh_calls_with_explicit_points,
    normalize_macro_refresh_mode,
)


@dataclass(frozen=True)
class TurboSchedule:
    timesteps: int = 100
    stride: int = 30
    tail_steps: int = 1
    codebook_size: int = 2048
    refresh_mode: str = "teacher"
    refresh_count: int = 1
    refresh_ratio: float = 0.5
    refresh_min_span: int = 8
    refresh_t_list: str = ""
    macro_eta: float = 1.0
    multi_index: bool = True

    @classmethod
    def from_spec(cls, spec: str, timesteps: int = 100) -> "TurboSchedule":
        tokens = [x.strip() for x in spec.split(":") if x.strip()]
        if len(tokens) < 2:
            raise ValueError(f"Bad schedule spec '{spec}', expected s30:k2048[:t1][:c1][:r0.5][:m8][:p84-54-19]")
        m_stride = re.fullmatch(r"s(\d+)", tokens[0])
        m_codebook = re.fullmatch(r"k(\d+)", tokens[1])
        if not m_stride or not m_codebook:
            raise ValueError(f"Bad schedule spec '{spec}', expected first tokens s<stride>:k<codebook>")

        values = {
            "timesteps": int(timesteps),
            "stride": int(m_stride.group(1)),
            "codebook_size": int(m_codebook.group(1)),
        }
        for token in tokens[2:]:
            for key, pattern, caster in (
                ("tail_steps", r"t(\d+)", int),
                ("refresh_count", r"c(\d+)", int),
                ("refresh_ratio", r"r([0-9]*\.?[0-9]+)", float),
                ("refresh_min_span", r"m(\d+)", int),
                ("macro_eta", r"e([0-9]*\.?[0-9]+)", float),
            ):
                match = re.fullmatch(pattern, token)
                if match:
                    values[key] = caster(match.group(1))
                    break
            else:
                match = re.fullmatch(r"p(\d+(?:-\d+)*)", token)
                if match:
                    values["refresh_t_list"] = match.group(1)
                    continue
                if token in {"none", "teacher"}:
                    values["refresh_mode"] = token
                    continue
                raise ValueError(f"Unsupported token '{token}' in schedule spec '{spec}'")
        return cls(**values)

    @property
    def bits_per_index(self) -> int:
        return int(math.ceil(math.log2(max(2, int(self.codebook_size)))))

    @property
    def boundaries(self) -> list[int]:
        denoise_steps, _, _ = build_macro_sampling_plan(self.timesteps, self.stride, self.tail_steps)
        return [int(x) for x in denoise_steps]

    @property
    def feedback_slots(self) -> int:
        _, _, slots = build_macro_sampling_plan(self.timesteps, self.stride, self.tail_steps)
        if not self.multi_index:
            return max(0, len(self.boundaries) - 1)
        return int(slots)

    @property
    def denoiser_calls(self) -> int:
        denoise_steps, transitions, _ = build_macro_sampling_plan(self.timesteps, self.stride, self.tail_steps)
        mode = normalize_macro_refresh_mode(self.refresh_mode)
        if self.stride <= 1:
            mode = "none"
        refresh_calls = count_macro_refresh_calls_with_explicit_points(
            transitions,
            mode,
            self.multi_index,
            self.refresh_min_span,
            self.refresh_ratio,
            self.refresh_count,
            self.refresh_t_list,
        )
        return int(len(denoise_steps) + refresh_calls)

    @property
    def feedback_bits(self) -> int:
        return int(self.feedback_slots * self.bits_per_index)
