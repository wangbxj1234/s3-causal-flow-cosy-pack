#!/usr/bin/env python3
"""
Gradio Web UI for S3-Causal-Flow codec inference.

Four inference modes:
  1. Self-Prompt (non-streaming) — reconstruct using same-speaker prompt
  2. Streaming — chunk-based streaming self-prompt inference
  3. Streaming Cross-Speaker — chunk-based streaming with a separate speaker voice
  4. Cross-Speaker (non-streaming) — reconstruct with another speaker's embedding

Usage:
  python webui.py [--port 8000]
"""
from __future__ import annotations

import argparse
import os
import subprocess
import sys
import tempfile
from pathlib import Path

import gradio as gr

REPO_ROOT = Path(__file__).resolve().parent
PYTHON = sys.executable

MODES = ["Self-Prompt", "Streaming", "Streaming Cross-Speaker", "Cross-Speaker"]

MODE_INSTRUCTIONS = {
    "Self-Prompt": (
        "Upload an audio file. The first portion (controlled by Prompt Ratio) "
        "serves as the prompt; the rest is reconstructed by the flow model."
    ),
    "Streaming": (
        "Upload an audio file for chunk-based streaming self-prompt inference. "
        "The first Prompt Duration (ms) of the audio conditions the model; "
        "the remainder is reconstructed in chunks."
    ),
    "Streaming Cross-Speaker": (
        "Upload a content audio and a separate speaker audio for chunk-based "
        "streaming cross-speaker inference. Speech tokens come from the content; "
        "the speaker embedding comes from the speaker audio."
    ),
    "Cross-Speaker": (
        "Upload a content audio and a separate speaker audio. "
        "Speech tokens come from the content; the speaker embedding comes "
        "from the speaker audio."
    ),
}


def _build_env() -> dict[str, str]:
    env = os.environ.copy()
    matcha = REPO_ROOT / "third_party" / "Matcha-TTS"
    env["PYTHONPATH"] = f"{REPO_ROOT}:{matcha}"
    return env


def run_inference(
    mode: str,
    input_audio: str | None,
    speaker_audio: str | None,
    prompt_ratio: float,
    chunk_ms: int,
    prompt_ms: int,
    token_overlap: int,
    n_timesteps: int,
):
    if not input_audio:
        raise gr.Error("Please upload an input audio file.")

    with tempfile.NamedTemporaryFile(suffix=".wav", delete=False, dir="/tmp") as tmp:
        out_path = tmp.name

    env = _build_env()

    if mode == "Self-Prompt":
        cmd = [
            PYTHON,
            str(REPO_ROOT / "tools" / "infer_flow_reconstruct_causal_s3tok25hz.py"),
            "--wav", input_audio,
            "--out_wav", out_path,
            "--prompt_ratio", str(prompt_ratio),
            "--n_timesteps", str(int(n_timesteps)),
        ]
    elif mode in ("Streaming", "Streaming Cross-Speaker"):
        cmd = [
            PYTHON,
            str(REPO_ROOT / "tools" / "infer_flow_streaming_s3tok25hz.py"),
            "--wav", input_audio,
            "--out_wav", out_path,
            "--chunk_ms", str(int(chunk_ms)),
            "--prompt_ms", str(int(prompt_ms)),
            "--token_overlap", str(int(token_overlap)),
            "--n_timesteps", str(int(n_timesteps)),
        ]
        if mode == "Streaming Cross-Speaker":
            if not speaker_audio:
                raise gr.Error("Streaming Cross-Speaker mode requires a speaker audio file.")
            cmd.extend(["--speaker_wav", speaker_audio])
    elif mode == "Cross-Speaker":
        if not speaker_audio:
            raise gr.Error("Cross-Speaker mode requires a speaker audio file.")
        cmd = [
            PYTHON,
            str(REPO_ROOT / "tools" / "infer_flow_reconstruct_cross_speaker_causal_s3tok25hz.py"),
            "--content_wav", input_audio,
            "--speaker_wav", speaker_audio,
            "--out_wav", out_path,
            "--n_timesteps", str(int(n_timesteps)),
        ]
    else:
        raise gr.Error(f"Unknown mode: {mode}")

    result = subprocess.run(
        cmd, capture_output=True, text=True, env=env, cwd=str(REPO_ROOT),
    )

    if result.returncode != 0:
        if os.path.exists(out_path):
            os.unlink(out_path)
        stderr = result.stderr.strip()
        stdout = result.stdout.strip()
        detail = stderr or stdout or "(no output)"
        raise gr.Error(f"Inference failed (exit {result.returncode}):\n{detail}")

    if not os.path.isfile(out_path):
        raise gr.Error("Output file was not generated.")

    return out_path


def on_mode_change(mode: str):
    is_self = mode == "Self-Prompt"
    is_stream_self = mode == "Streaming"
    is_stream_cross = mode == "Streaming Cross-Speaker"
    is_cross = mode == "Cross-Speaker"
    is_any_stream = is_stream_self or is_stream_cross
    needs_speaker = is_stream_cross or is_cross

    speaker_label = "Speaker Audio" if is_cross else "Speaker Audio (cross-speaker voice)"
    input_label = "Content Audio" if (is_cross or is_stream_cross) else "Input Audio"

    return (
        gr.update(value=MODE_INSTRUCTIONS[mode]),           # instruction_text
        gr.update(label=input_label),                       # input_audio
        gr.update(visible=needs_speaker,
                  label=speaker_label),                     # speaker_audio
        gr.update(visible=is_self),                         # prompt_ratio
        gr.update(visible=is_any_stream),                   # chunk_ms
        gr.update(visible=is_any_stream),                   # prompt_ms
        gr.update(visible=is_any_stream),                   # token_overlap
    )


def build_ui() -> gr.Blocks:
    with gr.Blocks(title="S3 Causal Flow — Codec Inference") as demo:
        gr.Markdown("## S3 Causal Flow — Codec Inference")

        with gr.Row():
            mode_radio = gr.Radio(
                choices=MODES,
                value=MODES[0],
                label="Inference Mode",
            )
            instruction_text = gr.Textbox(
                value=MODE_INSTRUCTIONS[MODES[0]],
                label="Instructions",
                interactive=False,
                scale=2,
            )

        with gr.Row():
            input_audio = gr.Audio(
                sources=["upload"],
                type="filepath",
                label="Input Audio",
            )
            speaker_audio = gr.Audio(
                sources=["upload"],
                type="filepath",
                label="Speaker Audio (optional — enables cross-speaker streaming)",
                visible=False,
            )

        with gr.Row():
            prompt_ratio = gr.Slider(
                minimum=0.1,
                maximum=0.9,
                value=0.35,
                step=0.05,
                label="Prompt Ratio",
                info="Fraction of aligned tokens used as prompt prefix (rest is reconstructed)",
            )
            chunk_ms = gr.Number(
                value=640,
                label="Chunk Size (ms)",
                minimum=100,
                maximum=2000,
                visible=False,
            )
            prompt_ms = gr.Number(
                value=1000,
                label="Prompt Duration (ms)",
                minimum=200,
                maximum=5000,
                visible=False,
            )
            token_overlap = gr.Number(
                value=10,
                label="Token Overlap",
                minimum=0,
                maximum=50,
                visible=False,
            )

        with gr.Row():
            n_timesteps = gr.Number(
                value=20,
                label="Flow ODE Steps (n_timesteps)",
                minimum=1,
                maximum=100,
            )

        generate_btn = gr.Button("Generate", variant="primary", size="lg")
        audio_output = gr.Audio(label="Output Audio", type="filepath")

        mode_radio.change(
            fn=on_mode_change,
            inputs=[mode_radio],
            outputs=[
                instruction_text,
                input_audio,
                speaker_audio,
                prompt_ratio,
                chunk_ms,
                prompt_ms,
                token_overlap,
            ],
        )

        generate_btn.click(
            fn=run_inference,
            inputs=[
                mode_radio,
                input_audio,
                speaker_audio,
                prompt_ratio,
                chunk_ms,
                prompt_ms,
                token_overlap,
                n_timesteps,
            ],
            outputs=[audio_output],
        )

    return demo


def main():
    parser = argparse.ArgumentParser(description="S3 Causal Flow Web UI")
    parser.add_argument("--port", type=int, default=8000)
    args = parser.parse_args()

    demo = build_ui()
    demo.queue(max_size=4, default_concurrency_limit=2)
    demo.launch(server_name="0.0.0.0", server_port=args.port)


if __name__ == "__main__":
    main()
