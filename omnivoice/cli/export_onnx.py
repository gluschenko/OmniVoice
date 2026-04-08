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

"""Export OmniVoice forward graph to ONNX.

This exports the model's acoustic forward pass:

    input_ids, audio_mask, attention_mask, position_ids -> logits

It intentionally does not export the full Python generation pipeline, which
contains tokenizer/audio preprocessing and iterative decoding logic.
"""

import argparse
import logging
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn

from omnivoice.models.omnivoice import OmniVoice


INPUT_NAMES = [
    "input_ids",
    "audio_mask",
    "attention_mask",
    "position_ids",
]

CALIBRATION_METHODS = {
    "minmax": "MinMax",
    "entropy": "Entropy",
    "percentile": "Percentile",
    "distribution": "Distribution",
}

QUANTIZATION_PROFILES = {
    "qint8": {
        "kind": "dynamic",
        "weight_type": "QInt8",
        "per_channel": False,
        "reduce_range": False,
        "description": "Dynamic int8 weights.",
    },
    "quint8": {
        "kind": "dynamic",
        "weight_type": "QUInt8",
        "per_channel": False,
        "reduce_range": False,
        "description": "Dynamic uint8 weights.",
    },
    "qint8_per_channel": {
        "kind": "dynamic",
        "weight_type": "QInt8",
        "per_channel": True,
        "reduce_range": False,
        "description": "Dynamic int8 weights with per-channel quantization.",
    },
    "quint8_per_channel": {
        "kind": "dynamic",
        "weight_type": "QUInt8",
        "per_channel": True,
        "reduce_range": False,
        "description": "Dynamic uint8 weights with per-channel quantization.",
    },
    "qint8_reduce_range": {
        "kind": "dynamic",
        "weight_type": "QInt8",
        "per_channel": False,
        "reduce_range": True,
        "description": "Dynamic int8 weights with reduced range.",
    },
    "quint8_reduce_range": {
        "kind": "dynamic",
        "weight_type": "QUInt8",
        "per_channel": False,
        "reduce_range": True,
        "description": "Dynamic uint8 weights with reduced range.",
    },
    "qint8_per_channel_reduce_range": {
        "kind": "dynamic",
        "weight_type": "QInt8",
        "per_channel": True,
        "reduce_range": True,
        "description": "Dynamic int8 weights with per-channel reduced-range quantization.",
    },
    "quint8_per_channel_reduce_range": {
        "kind": "dynamic",
        "weight_type": "QUInt8",
        "per_channel": True,
        "reduce_range": True,
        "description": "Dynamic uint8 weights with per-channel reduced-range quantization.",
    },
    "qint16": {
        "kind": "dynamic",
        "weight_type": "QInt16",
        "per_channel": False,
        "reduce_range": False,
        "description": "Dynamic int16 weights.",
    },
    "quint16": {
        "kind": "dynamic",
        "weight_type": "QUInt16",
        "per_channel": False,
        "reduce_range": False,
        "description": "Dynamic uint16 weights.",
    },
    "qint16_per_channel": {
        "kind": "dynamic",
        "weight_type": "QInt16",
        "per_channel": True,
        "reduce_range": False,
        "description": "Dynamic int16 weights with per-channel quantization.",
    },
    "quint16_per_channel": {
        "kind": "dynamic",
        "weight_type": "QUInt16",
        "per_channel": True,
        "reduce_range": False,
        "description": "Dynamic uint16 weights with per-channel quantization.",
    },
    "static_qdq_u8s8": {
        "kind": "static",
        "quant_format": "QDQ",
        "activation_type": "QUInt8",
        "weight_type": "QInt8",
        "per_channel": False,
        "reduce_range": False,
        "description": "Static QDQ quantization with uint8 activations and int8 weights.",
    },
    "static_qdq_s8s8": {
        "kind": "static",
        "quant_format": "QDQ",
        "activation_type": "QInt8",
        "weight_type": "QInt8",
        "per_channel": False,
        "reduce_range": False,
        "description": "Static QDQ quantization with int8 activations and int8 weights.",
    },
    "static_qoperator_u8s8": {
        "kind": "static",
        "quant_format": "QOperator",
        "activation_type": "QUInt8",
        "weight_type": "QInt8",
        "per_channel": False,
        "reduce_range": False,
        "description": "Static QOperator quantization with uint8 activations and int8 weights.",
    },
    "static_qoperator_s8s8": {
        "kind": "static",
        "quant_format": "QOperator",
        "activation_type": "QInt8",
        "weight_type": "QInt8",
        "per_channel": False,
        "reduce_range": False,
        "description": "Static QOperator quantization with int8 activations and int8 weights.",
    },
    "static_qdq_u8s8_per_channel": {
        "kind": "static",
        "quant_format": "QDQ",
        "activation_type": "QUInt8",
        "weight_type": "QInt8",
        "per_channel": True,
        "reduce_range": False,
        "description": "Static QDQ quantization with uint8 activations and per-channel int8 weights.",
    },
    "static_qdq_s8s8_per_channel": {
        "kind": "static",
        "quant_format": "QDQ",
        "activation_type": "QInt8",
        "weight_type": "QInt8",
        "per_channel": True,
        "reduce_range": False,
        "description": "Static QDQ quantization with int8 activations and per-channel int8 weights.",
    },
    "static_qoperator_u8s8_per_channel": {
        "kind": "static",
        "quant_format": "QOperator",
        "activation_type": "QUInt8",
        "weight_type": "QInt8",
        "per_channel": True,
        "reduce_range": False,
        "description": "Static QOperator quantization with uint8 activations and per-channel int8 weights.",
    },
    "static_qoperator_s8s8_per_channel": {
        "kind": "static",
        "quant_format": "QOperator",
        "activation_type": "QInt8",
        "weight_type": "QInt8",
        "per_channel": True,
        "reduce_range": False,
        "description": "Static QOperator quantization with int8 activations and per-channel int8 weights.",
    },
}

DYNAMIC_QUANTIZATION_PROFILES = [
    name
    for name, profile in QUANTIZATION_PROFILES.items()
    if profile["kind"] == "dynamic"
]

STATIC_QUANTIZATION_PROFILES = [
    name
    for name, profile in QUANTIZATION_PROFILES.items()
    if profile["kind"] == "static"
]

QUANTIZATION_ALIASES = {
    "all_dynamic": DYNAMIC_QUANTIZATION_PROFILES,
    "all_static": STATIC_QUANTIZATION_PROFILES,
    "all": list(QUANTIZATION_PROFILES),
}


def get_best_device() -> str:
    """Auto-detect the best device available for export."""
    if torch.cuda.is_available():
        return "cuda"
    if torch.backends.mps.is_available():
        return "mps"
    return "cpu"


def parse_dtype(value: str) -> torch.dtype:
    """Map CLI dtype string to torch dtype."""
    mapping = {
        "float32": torch.float32,
        "fp32": torch.float32,
        "float16": torch.float16,
        "fp16": torch.float16,
        "bfloat16": torch.bfloat16,
        "bf16": torch.bfloat16,
    }
    key = value.lower()
    if key not in mapping:
        raise ValueError(f"Unsupported dtype: {value}")
    return mapping[key]


def dtype_name(dtype: torch.dtype) -> str:
    """Return a stable user-facing dtype name."""
    if dtype == torch.float16:
        return "float16"
    if dtype == torch.bfloat16:
        return "bfloat16"
    return "float32"


class OmniVoiceOnnxWrapper(nn.Module):
    """Thin wrapper exposing a tensor-only forward for ONNX export."""

    def __init__(self, model: OmniVoice):
        super().__init__()
        self.model = model

    def _build_llm_attention_mask(self, attention_mask: torch.Tensor) -> torch.Tensor:
        """Match OmniVoice's non-causal batched inference mask.

        During iterative decoding the PyTorch path uses a 4-D boolean mask where
        every valid token can attend to every other valid token. Padded query
        rows keep a self-diagonal ``True`` entry so the underlying LLM does not
        see a fully masked row. Export must mirror that layout; using a causal
        mask here severely degrades generation quality.
        """
        valid_tokens = attention_mask.ne(0)
        _, seq_len = valid_tokens.shape
        device = attention_mask.device

        full_attention = valid_tokens[:, None, :, None] & valid_tokens[:, None, None, :]

        pad_queries = ~valid_tokens[:, None, :, None]
        positions = torch.arange(seq_len, device=device)
        pad_diag = positions.view(1, 1, seq_len, 1).eq(
            positions.view(1, 1, 1, seq_len)
        )

        return full_attention | (pad_queries & pad_diag)

    def forward(
        self,
        input_ids: torch.Tensor,
        audio_mask: torch.Tensor,
        attention_mask: torch.Tensor,
        position_ids: torch.Tensor,
    ) -> torch.Tensor:
        return self.model(
            input_ids=input_ids,
            audio_mask=audio_mask,
            attention_mask=self._build_llm_attention_mask(attention_mask),
            position_ids=position_ids,
        ).logits


def build_dummy_inputs(
    model: OmniVoice,
    batch_size: int,
    seq_len: int,
    text_prefix_len: int,
    device: str,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """Construct shape-correct dummy inputs for tracing/export."""
    if seq_len < 2:
        raise ValueError("--seq-len must be at least 2.")

    if text_prefix_len < 1 or text_prefix_len >= seq_len:
        raise ValueError("--text-prefix-len must be in [1, seq_len - 1].")

    num_codebooks = model.config.num_audio_codebook
    text_vocab_size = model.get_input_embeddings().weight.shape[0]
    audio_vocab_size = model.config.audio_vocab_size

    input_ids = torch.zeros(
        (batch_size, num_codebooks, seq_len), dtype=torch.long, device=device
    )

    text_ids = torch.randint(
        low=0,
        high=text_vocab_size,
        size=(batch_size, text_prefix_len),
        dtype=torch.long,
        device=device,
    )
    input_ids[:, :, :text_prefix_len] = text_ids.unsqueeze(1).expand(
        -1, num_codebooks, -1
    )

    audio_len = seq_len - text_prefix_len
    audio_ids = torch.randint(
        low=0,
        high=audio_vocab_size,
        size=(batch_size, num_codebooks, audio_len),
        dtype=torch.long,
        device=device,
    )
    input_ids[:, :, text_prefix_len:] = audio_ids

    audio_mask = torch.zeros((batch_size, seq_len), dtype=torch.bool, device=device)
    audio_mask[:, text_prefix_len:] = True

    attention_mask = torch.ones(
        (batch_size, seq_len), dtype=torch.long, device=device
    )
    position_ids = torch.arange(seq_len, dtype=torch.long, device=device).unsqueeze(0)
    position_ids = position_ids.expand(batch_size, -1)

    return input_ids, audio_mask, attention_mask, position_ids


def maybe_check_onnx_installed() -> None:
    """Provide an actionable message if ONNX is missing."""
    try:
        import onnx  # noqa: F401
    except ImportError as exc:
        raise RuntimeError(
            "The `onnx` package is required for export. "
            "Install it with `pip install .[onnx]` or `pip install onnx`."
        ) from exc


def maybe_get_quantization_api():
    """Load ONNX Runtime quantization APIs lazily."""
    try:
        from onnxruntime.quantization import (
            CalibrationMethod,
            QuantFormat,
            QuantType,
            quantize_dynamic,
            quantize_static,
        )
    except ImportError as exc:
        raise RuntimeError(
            "ONNX quantization requires `onnxruntime` (or `onnxruntime-gpu`). "
            "Install it with `pip install onnxruntime` before using --quantize."
        ) from exc

    return CalibrationMethod, QuantFormat, QuantType, quantize_dynamic, quantize_static


def parse_calibration_method(value: str) -> str:
    """Parse the static-quantization calibration method."""
    key = value.strip().lower().replace("-", "_")
    if key not in CALIBRATION_METHODS:
        raise ValueError(
            "Unsupported calibration method: "
            f"{value}. Available methods: {', '.join(CALIBRATION_METHODS)}."
        )
    return CALIBRATION_METHODS[key]


def parse_quantization_profiles(values: list[str] | None) -> list[str]:
    """Parse requested quantization profiles from CLI tokens."""
    if not values:
        return []

    profiles: list[str] = []
    available = ", ".join(
        [*QUANTIZATION_PROFILES.keys(), *QUANTIZATION_ALIASES.keys()]
    )

    for value in values:
        for part in value.split(","):
            key = part.strip().lower().replace("-", "_")
            if not key:
                continue

            if key in QUANTIZATION_ALIASES:
                for profile_name in QUANTIZATION_ALIASES[key]:
                    if profile_name not in profiles:
                        profiles.append(profile_name)
                continue

            if key not in QUANTIZATION_PROFILES:
                raise ValueError(
                    f"Unsupported quantization profile: {part}. "
                    f"Available profiles: {available}, all."
                )

            if key not in profiles:
                profiles.append(key)

    return profiles


class SyntheticCalibrationDataReader:
    """Feed synthetic calibration samples into ONNX Runtime static quantization."""

    def __init__(self, samples: list[dict[str, np.ndarray]]):
        self.samples = samples
        self.start_index = 0
        self.end_index = len(samples)
        self.index = self.start_index

    def get_next(self) -> dict[str, np.ndarray] | None:
        if self.index >= self.end_index:
            return None

        item = self.samples[self.index]
        self.index += 1
        return item

    def __len__(self) -> int:
        return self.end_index - self.start_index

    def set_range(self, start_index: int, end_index: int):
        self.start_index = max(0, start_index)
        self.end_index = min(len(self.samples), end_index)
        self.index = self.start_index


def tensor_inputs_to_numpy(
    dummy_inputs: tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor],
) -> dict[str, np.ndarray]:
    """Convert traced PyTorch inputs to NumPy for ONNX Runtime."""
    return {
        name: tensor.detach().cpu().numpy()
        for name, tensor in zip(INPUT_NAMES, dummy_inputs, strict=True)
    }


def build_calibration_samples(
    model: OmniVoice,
    batch_size: int,
    seq_len: int,
    text_prefix_len: int,
    num_samples: int,
) -> list[dict[str, np.ndarray]]:
    """Generate synthetic calibration feeds matching the exported ONNX graph."""
    if num_samples < 1:
        raise ValueError("--calibration-samples must be at least 1 for static quantization.")

    samples: list[dict[str, np.ndarray]] = []
    for _ in range(num_samples):
        samples.append(
            tensor_inputs_to_numpy(
                build_dummy_inputs(
                    model=model,
                    batch_size=batch_size,
                    seq_len=seq_len,
                    text_prefix_len=text_prefix_len,
                    device="cpu",
                )
            )
        )
    return samples


def get_external_data_path(output_path: Path) -> Path:
    """Return the consolidated external data filename for an ONNX model."""
    return output_path.with_suffix(output_path.suffix + "_data")


def get_quantized_output_path(output_path: Path, profile_name: str) -> Path:
    """Append the quantization profile to the ONNX filename."""
    return output_path.with_name(
        f"{output_path.stem}.{profile_name}{output_path.suffix}"
    )


def collect_external_data_files(output_path: Path) -> set[Path]:
    """List current external tensor shard files referenced by the model."""
    import onnx
    from onnx import external_data_helper

    model = onnx.load(str(output_path), load_external_data=False)
    external_files: set[Path] = set()
    for tensor in external_data_helper._get_all_tensors(model):
        if tensor.data_location != onnx.TensorProto.EXTERNAL:
            continue

        location = external_data_helper.ExternalDataInfo(tensor).location
        if location:
            external_files.add((output_path.parent / location).resolve())

    return external_files


def repack_external_data(output_path: Path) -> Path:
    """Rewrite sharded external tensors into a single .onnx_data file."""
    import onnx

    external_data_path = get_external_data_path(output_path)
    temp_output_path = output_path.with_suffix(output_path.suffix + ".tmp")
    stale_external_files = collect_external_data_files(output_path)

    model = onnx.load(str(output_path), load_external_data=True)
    if external_data_path.exists():
        external_data_path.unlink()
    if temp_output_path.exists():
        temp_output_path.unlink()
    onnx.save_model(
        model,
        str(temp_output_path),
        save_as_external_data=True,
        all_tensors_to_one_file=True,
        location=external_data_path.name,
        size_threshold=0,
        convert_attribute=False,
    )
    onnx.checker.check_model(str(temp_output_path))

    output_path.unlink()
    temp_output_path.replace(output_path)

    for stale_file in stale_external_files:
        if stale_file.exists() and stale_file != external_data_path:
            stale_file.unlink()

    return external_data_path


def export_with_external_data_fallback(
    wrapper: nn.Module,
    dummy_inputs: tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor],
    output_path: Path,
    opset: int,
    dynamic_axes: dict | None,
    external_data: bool,
) -> None:
    """Call torch.onnx.export across compatible keyword variants."""
    export_kwargs = dict(
        export_params=True,
        opset_version=opset,
        do_constant_folding=True,
        input_names=[
            "input_ids",
            "audio_mask",
            "attention_mask",
            "position_ids",
        ],
        output_names=["logits"],
        dynamic_axes=dynamic_axes,
        dynamo=False,
    )

    try:
        torch.onnx.export(
            wrapper,
            dummy_inputs,
            str(output_path),
            external_data=external_data,
            **export_kwargs,
        )
        return
    except TypeError as exc:
        if "external_data" not in str(exc):
            raise

    torch.onnx.export(
        wrapper,
        dummy_inputs,
        str(output_path),
        use_external_data_format=external_data,
        **export_kwargs,
    )


def quantize_onnx_model(
    input_path: Path,
    output_path: Path,
    profile_name: str,
    external_data: bool,
    calibration_samples: list[dict[str, np.ndarray]] | None = None,
    calibration_method: str = "MinMax",
) -> None:
    """Create one quantized ONNX variant."""
    (
        CalibrationMethod,
        QuantFormat,
        QuantType,
        quantize_dynamic,
        quantize_static,
    ) = maybe_get_quantization_api()
    profile = QUANTIZATION_PROFILES[profile_name]

    if profile["kind"] == "dynamic":
        quantize_dynamic(
            str(input_path),
            str(output_path),
            weight_type=getattr(QuantType, profile["weight_type"]),
            per_channel=profile["per_channel"],
            reduce_range=profile["reduce_range"],
            use_external_data_format=external_data,
        )
        return

    if calibration_samples is None:
        raise RuntimeError(
            f"Static quantization profile `{profile_name}` requires calibration samples."
        )

    quantize_static(
        str(input_path),
        str(output_path),
        SyntheticCalibrationDataReader(calibration_samples),
        quant_format=getattr(QuantFormat, profile["quant_format"]),
        per_channel=profile["per_channel"],
        reduce_range=profile["reduce_range"],
        activation_type=getattr(QuantType, profile["activation_type"]),
        weight_type=getattr(QuantType, profile["weight_type"]),
        use_external_data_format=external_data,
        calibrate_method=getattr(CalibrationMethod, calibration_method),
    )


def get_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Export OmniVoice forward graph to ONNX.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--model",
        type=str,
        default="k2-fsa/OmniVoice",
        help="Model checkpoint path or HuggingFace repo id.",
    )
    parser.add_argument(
        "--output",
        type=str,
        required=True,
        help="Output ONNX file path.",
    )
    parser.add_argument(
        "--device",
        type=str,
        default=None,
        help="Export device. Defaults to CUDA if available, otherwise CPU.",
    )
    parser.add_argument(
        "--dtype",
        type=str,
        default="float32",
        choices=["float32", "fp32", "float16", "fp16", "bfloat16", "bf16"],
        help="Model dtype used during export.",
    )
    parser.add_argument(
        "--opset",
        type=int,
        default=17,
        help="ONNX opset version.",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=1,
        help="Dummy batch size used for tracing.",
    )
    parser.add_argument(
        "--seq-len",
        type=int,
        default=256,
        help="Dummy sequence length used for tracing.",
    )
    parser.add_argument(
        "--text-prefix-len",
        type=int,
        default=96,
        help="Dummy text-token prefix length inside the traced sequence.",
    )
    parser.add_argument(
        "--external-data",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Store large weights in external data files when needed.",
    )
    parser.add_argument(
        "--dynamic",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Export dynamic batch/sequence axes.",
    )
    parser.add_argument(
        "--quantize",
        nargs="*",
        default=None,
        metavar="PROFILE",
        help=(
            "Optional post-export ONNX Runtime quantization profiles to emit in "
            "addition to the base model. Accepts space- or comma-separated names: "
            f"{', '.join(QUANTIZATION_PROFILES)}, {', '.join(QUANTIZATION_ALIASES)}."
        ),
    )
    parser.add_argument(
        "--calibration-samples",
        type=int,
        default=8,
        help="Number of synthetic calibration batches for static quantization profiles.",
    )
    parser.add_argument(
        "--calibration-method",
        type=str,
        default="minmax",
        choices=list(CALIBRATION_METHODS),
        help="Calibration method for static quantization profiles.",
    )
    return parser


def main():
    formatter = "%(asctime)s %(levelname)s [%(filename)s:%(lineno)d] %(message)s"
    logging.basicConfig(format=formatter, level=logging.INFO, force=True)

    args = get_parser().parse_args()
    maybe_check_onnx_installed()
    quantization_profiles = parse_quantization_profiles(args.quantize)
    calibration_method = parse_calibration_method(args.calibration_method)

    device = args.device or get_best_device()
    export_dtype = parse_dtype(args.dtype)

    if device == "cpu" and export_dtype in (torch.float16, torch.bfloat16):
        logging.warning(
            "Exporting %s on CPU may fail for some operators. "
            "Use --device cuda for lower-precision export if available.",
            dtype_name(export_dtype),
        )

    output_path = Path(args.output).expanduser().resolve()
    output_path.parent.mkdir(parents=True, exist_ok=True)

    logging.info(
        "Loading model from %s on %s with dtype=%s ...",
        args.model,
        device,
        dtype_name(export_dtype),
    )
    model = OmniVoice.from_pretrained(
        args.model,
        device_map=device,
        dtype=export_dtype,
        train=True,
        attn_implementation="eager",
    )
    model.eval()

    wrapper = OmniVoiceOnnxWrapper(model).to(device)
    wrapper.eval()

    dummy_inputs = build_dummy_inputs(
        model=model,
        batch_size=args.batch_size,
        seq_len=args.seq_len,
        text_prefix_len=args.text_prefix_len,
        device=device,
    )

    dynamic_axes = None
    if args.dynamic:
        dynamic_axes = {
            "input_ids": {0: "batch", 2: "sequence"},
            "audio_mask": {0: "batch", 1: "sequence"},
            "attention_mask": {0: "batch", 1: "sequence"},
            "position_ids": {0: "batch", 1: "sequence"},
            "logits": {0: "batch", 2: "sequence"},
        }

    calibration_samples = None
    if any(
        QUANTIZATION_PROFILES[profile_name]["kind"] == "static"
        for profile_name in quantization_profiles
    ):
        logging.info(
            "Preparing %s synthetic calibration samples for static quantization "
            "(method=%s) ...",
            args.calibration_samples,
            args.calibration_method,
        )
        calibration_samples = build_calibration_samples(
            model=model,
            batch_size=args.batch_size,
            seq_len=args.seq_len,
            text_prefix_len=args.text_prefix_len,
            num_samples=args.calibration_samples,
        )

    logging.info("Exporting ONNX graph to %s ...", output_path)
    with torch.inference_mode():
        export_with_external_data_fallback(
            wrapper,
            dummy_inputs,
            output_path,
            args.opset,
            dynamic_axes,
            args.external_data,
        )

    logging.info("ONNX export complete: %s", output_path)
    if args.external_data:
        external_data_path = repack_external_data(output_path)
        logging.info("Packed external weights into: %s", external_data_path)
        logging.info(
            "The external weights are stored in a single .onnx_data file next "
            "to the main .onnx model."
        )

    for profile_name in quantization_profiles:
        quantized_output_path = get_quantized_output_path(output_path, profile_name)
        profile = QUANTIZATION_PROFILES[profile_name]
        logging.info(
            "Quantizing %s -> %s (%s)",
            output_path,
            quantized_output_path,
            profile["description"],
        )
        quantize_onnx_model(
            input_path=output_path,
            output_path=quantized_output_path,
            profile_name=profile_name,
            external_data=args.external_data,
            calibration_samples=calibration_samples,
            calibration_method=calibration_method,
        )
        if args.external_data:
            quantized_external_data_path = repack_external_data(quantized_output_path)
            logging.info(
                "Packed quantized external weights into: %s",
                quantized_external_data_path,
            )


if __name__ == "__main__":
    main()
