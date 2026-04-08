#!/usr/bin/env python3
# Copyright    2026  Xiaomi Corp.        (authors:  Han Zhu)
#
# See ../../LICENSE for clarification regarding multiple authors
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Generate a WAV file from text with an exported OmniVoice ONNX graph.

This example keeps OmniVoice's text preprocessing, duration estimation,
iterative decoding, and audio-token decoding, but replaces the acoustic
forward pass with ONNX Runtime.

Notes:
  - The exported ONNX file only contains the tensor forward graph, so we still
    need the text tokenizer and the Higgs audio tokenizer/decoder.
  - This example focuses on text-only generation (auto voice / voice design).
    Voice cloning can be added later on top of the same pattern.
"""

from __future__ import annotations

import argparse
import logging
import math
from pathlib import Path

import numpy as np
import onnxruntime as ort
import soundfile as sf
import torch
import torch.nn.functional as F
import torchaudio
from transformers import (
    AutoFeatureExtractor,
    AutoTokenizer,
    HiggsAudioV2TokenizerModel,
)

from omnivoice.models.omnivoice import (
    _combine_text,
    _filter_top_k,
    _get_time_steps,
    _gumbel_sample,
    _resolve_instruct,
    _resolve_language,
    _tokenize_with_nonverbal_tags,
)
from omnivoice.utils.audio import fade_and_pad_audio, remove_silence
from omnivoice.utils.duration import RuleDurationEstimator
from omnivoice.utils.voice_design import _ZH_RE


NUM_AUDIO_CODEBOOK = 8
AUDIO_MASK_ID = 1024
DEFAULT_REF_TEXT = "Nice to meet you."
DEFAULT_REF_AUDIO_TOKENS = 25


def default_onnx_path() -> Path:
    repo_root = Path(__file__).resolve().parents[1]
    return repo_root / "artifacts" / "onnx" / "omnivoice.onnx"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Generate speech from text with OmniVoice ONNX Runtime.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--onnx",
        type=Path,
        default=default_onnx_path(),
        help="Path to exported omnivoice.onnx.",
    )
    parser.add_argument(
        "--text",
        type=str,
        required=True,
        help="Text to synthesize.",
    )
    parser.add_argument(
        "--output",
        type=Path,
        required=True,
        help="Output WAV file path.",
    )
    parser.add_argument(
        "--tokenizer",
        type=str,
        default="k2-fsa/OmniVoice",
        help="Path or Hugging Face repo for the OmniVoice text tokenizer.",
    )
    parser.add_argument(
        "--audio-tokenizer",
        type=str,
        default="eustlb/higgs-audio-v2-tokenizer",
        help="Path or Hugging Face repo for the audio tokenizer/decoder.",
    )
    parser.add_argument(
        "--device",
        type=str,
        choices=("auto", "cpu", "cuda"),
        default="auto",
        help="Execution provider for ONNX Runtime.",
    )
    parser.add_argument(
        "--audio-tokenizer-device",
        type=str,
        choices=("cpu", "cuda"),
        default="cpu",
        help="Torch device used by the Higgs audio tokenizer.",
    )
    parser.add_argument(
        "--language",
        type=str,
        default=None,
        help="Language name or ISO code, for example 'Russian' or 'ru'.",
    )
    parser.add_argument(
        "--instruct",
        type=str,
        default=None,
        help="Voice design prompt, for example 'female, low pitch, british accent'.",
    )
    parser.add_argument(
        "--duration",
        type=float,
        default=None,
        help="Exact output duration in seconds. Overrides --speed.",
    )
    parser.add_argument(
        "--speed",
        type=float,
        default=1.0,
        help="Speaking speed. Values > 1.0 are faster, < 1.0 are slower.",
    )
    parser.add_argument("--num-step", type=int, default=32)
    parser.add_argument("--guidance-scale", type=float, default=2.0)
    parser.add_argument("--t-shift", type=float, default=0.1)
    parser.add_argument("--layer-penalty-factor", type=float, default=5.0)
    parser.add_argument("--position-temperature", type=float, default=5.0)
    parser.add_argument("--class-temperature", type=float, default=0.0)
    parser.add_argument(
        "--seed",
        type=int,
        default=0,
        help="Random seed used by the Gumbel sampling steps.",
    )
    return parser.parse_args()


def pick_providers(device: str) -> list[str]:
    available = ort.get_available_providers()
    if device == "auto":
        if "CUDAExecutionProvider" in available:
            return ["CUDAExecutionProvider", "CPUExecutionProvider"]
        return ["CPUExecutionProvider"]
    if device == "cuda":
        if "CUDAExecutionProvider" not in available:
            raise RuntimeError(
                "CUDAExecutionProvider is not available. "
                f"Available providers: {available}"
            )
        return ["CUDAExecutionProvider", "CPUExecutionProvider"]
    return ["CPUExecutionProvider"]


class OmniVoiceOnnxGenerator:
    def __init__(
        self,
        onnx_path: Path,
        tokenizer_name: str,
        audio_tokenizer_name: str,
        device: str,
        audio_tokenizer_device: str,
    ) -> None:
        self.onnx_path = onnx_path
        self.providers = pick_providers(device)

        session_options = ort.SessionOptions()
        session_options.graph_optimization_level = (
            ort.GraphOptimizationLevel.ORT_ENABLE_ALL
        )
        self.session = ort.InferenceSession(
            str(onnx_path),
            sess_options=session_options,
            providers=self.providers,
        )

        self.text_tokenizer = AutoTokenizer.from_pretrained(tokenizer_name)
        self.audio_tokenizer = HiggsAudioV2TokenizerModel.from_pretrained(
            audio_tokenizer_name,
            device_map=audio_tokenizer_device,
        )
        self.feature_extractor = AutoFeatureExtractor.from_pretrained(
            audio_tokenizer_name
        )
        self.duration_estimator = RuleDurationEstimator()
        self.sampling_rate = self.feature_extractor.sampling_rate
        self.frame_rate = self.audio_tokenizer.config.frame_rate

    def estimate_target_tokens(
        self,
        text: str,
        speed: float,
        duration: float | None,
    ) -> int:
        if duration is not None:
            return max(1, int(duration * self.frame_rate))

        est = self.duration_estimator.estimate_duration(
            target_text=text,
            ref_text=DEFAULT_REF_TEXT,
            ref_duration=DEFAULT_REF_AUDIO_TOKENS,
        )
        if speed <= 0:
            raise ValueError("--speed must be > 0.")
        if speed != 1.0:
            est = est / speed
        return max(1, int(est))

    def prepare_inputs(
        self,
        text: str,
        num_target_tokens: int,
        language: str | None,
        instruct: str | None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        style_text = (
            f"<|lang_start|>{language or 'None'}<|lang_end|>"
            f"<|instruct_start|>{instruct or 'None'}<|instruct_end|>"
        )
        style_tokens = (
            self.text_tokenizer(style_text, return_tensors="pt")
            .input_ids.repeat(NUM_AUDIO_CODEBOOK, 1)
            .unsqueeze(0)
        )

        wrapped_text = f"<|text_start|>{_combine_text(text)}<|text_end|>"
        text_tokens = (
            _tokenize_with_nonverbal_tags(wrapped_text, self.text_tokenizer)
            .repeat(NUM_AUDIO_CODEBOOK, 1)
            .unsqueeze(0)
        )

        target_audio_tokens = torch.full(
            (1, NUM_AUDIO_CODEBOOK, num_target_tokens),
            AUDIO_MASK_ID,
            dtype=torch.long,
        )
        input_ids = torch.cat([style_tokens, text_tokens, target_audio_tokens], dim=2)

        audio_mask = torch.zeros((1, input_ids.size(-1)), dtype=torch.bool)
        audio_mask[0, -num_target_tokens:] = True
        return input_ids, audio_mask

    def run_onnx(
        self,
        input_ids: torch.Tensor,
        audio_mask: torch.Tensor,
        attention_mask: torch.Tensor,
        position_ids: torch.Tensor,
    ) -> torch.Tensor:
        outputs = self.session.run(
            None,
            {
                "input_ids": input_ids.cpu().numpy().astype(np.int64),
                "audio_mask": audio_mask.cpu().numpy().astype(np.bool_),
                "attention_mask": attention_mask.cpu().numpy().astype(np.int64),
                "position_ids": position_ids.cpu().numpy().astype(np.int64),
            },
        )
        return torch.from_numpy(outputs[0]).to(torch.float32)

    def predict_tokens(
        self,
        cond_logits: torch.Tensor,
        uncond_logits: torch.Tensor,
        guidance_scale: float,
        class_temperature: float,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        if guidance_scale != 0:
            cond_log_probs = F.log_softmax(cond_logits, dim=-1)
            uncond_log_probs = F.log_softmax(uncond_logits, dim=-1)
            log_probs = torch.log_softmax(
                cond_log_probs + guidance_scale * (cond_log_probs - uncond_log_probs),
                dim=-1,
            )
        else:
            log_probs = F.log_softmax(cond_logits, dim=-1)

        log_probs[..., AUDIO_MASK_ID] = -float("inf")

        if class_temperature > 0.0:
            filtered = _filter_top_k(log_probs, ratio=0.1)
            pred_tokens = _gumbel_sample(filtered, class_temperature).argmax(dim=-1)
        else:
            pred_tokens = log_probs.argmax(dim=-1)

        confidence_scores = log_probs.max(dim=-1)[0]
        return pred_tokens, confidence_scores

    def generate_tokens(self, args: argparse.Namespace) -> torch.Tensor:
        resolved_language = _resolve_language(args.language)
        resolved_instruct = None
        if args.instruct:
            resolved_instruct = _resolve_instruct(
                args.instruct,
                use_zh=bool(_ZH_RE.search(args.text)),
            )

        target_len = self.estimate_target_tokens(
            text=args.text,
            speed=args.speed,
            duration=args.duration,
        )
        logging.info("Target audio tokens: %s", target_len)

        cond_input_ids, cond_audio_mask = self.prepare_inputs(
            text=args.text,
            num_target_tokens=target_len,
            language=resolved_language,
            instruct=resolved_instruct,
        )
        cond_len = cond_input_ids.size(-1)

        batch_input_ids = torch.full(
            (2, NUM_AUDIO_CODEBOOK, cond_len),
            AUDIO_MASK_ID,
            dtype=torch.long,
        )
        batch_audio_mask = torch.zeros((2, cond_len), dtype=torch.bool)
        batch_attention_mask = torch.zeros((2, cond_len), dtype=torch.long)
        position_ids = torch.arange(cond_len, dtype=torch.long).unsqueeze(0).repeat(2, 1)

        batch_input_ids[0, :, :cond_len] = cond_input_ids[0]
        batch_audio_mask[0, :cond_len] = cond_audio_mask[0]
        batch_attention_mask[0, :cond_len] = 1

        batch_input_ids[1, :, :target_len] = cond_input_ids[0, :, -target_len:]
        batch_audio_mask[1, :target_len] = cond_audio_mask[0, -target_len:]
        batch_attention_mask[1, :target_len] = 1

        tokens = torch.full(
            (1, NUM_AUDIO_CODEBOOK, target_len),
            AUDIO_MASK_ID,
            dtype=torch.long,
        )
        layer_ids = torch.arange(NUM_AUDIO_CODEBOOK).view(1, -1, 1)

        timesteps = _get_time_steps(
            t_start=0.0,
            t_end=1.0,
            num_step=args.num_step + 1,
            t_shift=args.t_shift,
        ).tolist()
        total_mask = target_len * NUM_AUDIO_CODEBOOK
        remaining = total_mask
        schedule = []
        for step in range(args.num_step):
            if step == args.num_step - 1:
                num_to_fill = remaining
            else:
                span = timesteps[step + 1] - timesteps[step]
                num_to_fill = min(math.ceil(total_mask * span), remaining)
            schedule.append(int(num_to_fill))
            remaining -= int(num_to_fill)

        for step, num_to_fill in enumerate(schedule, start=1):
            logits = self.run_onnx(
                input_ids=batch_input_ids,
                audio_mask=batch_audio_mask,
                attention_mask=batch_attention_mask,
                position_ids=position_ids,
            )

            cond_logits = logits[0:1, :, cond_len - target_len : cond_len, :]
            uncond_logits = logits[1:2, :, :target_len, :]

            pred_tokens, scores = self.predict_tokens(
                cond_logits=cond_logits,
                uncond_logits=uncond_logits,
                guidance_scale=args.guidance_scale,
                class_temperature=args.class_temperature,
            )

            scores = scores - (layer_ids * args.layer_penalty_factor)
            if args.position_temperature > 0.0:
                scores = _gumbel_sample(scores, args.position_temperature)

            sample_tokens = tokens[:, :, :target_len]
            scores.masked_fill_(sample_tokens != AUDIO_MASK_ID, -float("inf"))

            if num_to_fill > 0:
                _, topk_idx = torch.topk(scores.flatten(), num_to_fill)
                flat_tokens = sample_tokens.flatten()
                flat_tokens[topk_idx] = pred_tokens.flatten()[topk_idx]
                sample_tokens.copy_(flat_tokens.view_as(sample_tokens))

                batch_input_ids[0:1, :, cond_len - target_len : cond_len] = sample_tokens
                batch_input_ids[1:2, :, :target_len] = sample_tokens

            logging.info(
                "Step %s/%s: filled %s tokens",
                step,
                args.num_step,
                num_to_fill,
            )

        return tokens[0]

    def decode_audio(self, tokens: torch.Tensor) -> torch.Tensor:
        tokenizer_device = self.audio_tokenizer.device
        audio = (
            self.audio_tokenizer.decode(tokens.to(tokenizer_device).unsqueeze(0))
            .audio_values[0]
            .detach()
            .cpu()
        )

        audio = remove_silence(
            audio,
            sampling_rate=self.sampling_rate,
            mid_sil=500,
            lead_sil=100,
            trail_sil=100,
        )
        peak = audio.abs().max()
        if peak > 1e-6:
            audio = audio / peak * 0.5
        return fade_and_pad_audio(audio, sample_rate=self.sampling_rate)


def main() -> None:
    args = parse_args()

    logging.basicConfig(
        format="%(asctime)s %(levelname)s [%(filename)s:%(lineno)d] %(message)s",
        level=logging.INFO,
        force=True,
    )

    if not args.onnx.is_file():
        raise FileNotFoundError(f"ONNX file not found: {args.onnx}")

    torch.manual_seed(args.seed)

    generator = OmniVoiceOnnxGenerator(
        onnx_path=args.onnx,
        tokenizer_name=args.tokenizer,
        audio_tokenizer_name=args.audio_tokenizer,
        device=args.device,
        audio_tokenizer_device=args.audio_tokenizer_device,
    )
    logging.info("ONNX providers: %s", generator.providers)

    tokens = generator.generate_tokens(args)
    audio = generator.decode_audio(tokens)

    args.output.parent.mkdir(parents=True, exist_ok=True)
    sf.write(
        str(args.output),
        audio.squeeze(0).numpy(),
        generator.sampling_rate,
    )
    logging.info("Saved audio to %s", args.output)


if __name__ == "__main__":
    main()
