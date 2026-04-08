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

import torch
import torch.nn as nn

from omnivoice.models.omnivoice import OmniVoice


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


def get_external_data_path(output_path: Path) -> Path:
    """Return the consolidated external data filename for an ONNX model."""
    return output_path.with_suffix(output_path.suffix + "_data")


def repack_external_data(output_path: Path) -> Path:
    """Rewrite sharded external tensors into a single .onnx_data file."""
    import onnx

    external_data_path = get_external_data_path(output_path)
    temp_output_path = output_path.with_suffix(output_path.suffix + ".tmp")

    stale_external_files = [
        path
        for path in output_path.parent.iterdir()
        if path.is_file()
        and path.name not in {output_path.name, external_data_path.name, temp_output_path.name}
    ]

    if external_data_path.exists():
        external_data_path.unlink()
    if temp_output_path.exists():
        temp_output_path.unlink()

    model = onnx.load(str(output_path), load_external_data=True)
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
        if stale_file.exists():
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
    return parser


def main():
    formatter = "%(asctime)s %(levelname)s [%(filename)s:%(lineno)d] %(message)s"
    logging.basicConfig(format=formatter, level=logging.INFO, force=True)

    args = get_parser().parse_args()
    maybe_check_onnx_installed()

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


if __name__ == "__main__":
    main()
