# Qwen3 ASR Transcriber

The **Qwen3 ASR Transcriber** is the primary inference node for the Qwen3-ASR model family. It converts speech from audio input into text and can optionally generate precise word-level timestamps when paired with a forced aligner.

## Parameters

- **audio**: The input audio stream (usually from a *Load Audio* node).
- **model_name**: The directory name of the Qwen3-ASR model (0.6B or 1.7B) located in `models/diffusion_models/Qwen3-ASR/`.
- **language**: The target language for transcription. Use `auto` to allow the model to automatically identify the spoken language.
- **device**: The hardware device to run the model on (`cuda` or `cpu`).
- **precision**: The floating-point precision. `bf16` is recommended for modern NVIDIA GPUs to save VRAM without losing accuracy.
- **max_new_tokens**: The maximum number of tokens to generate in the output text. Increase this for longer audio files.
- **forced_aligner** (Optional): An optional input from the **Qwen3 Forced Aligner Config** node. If connected, the node will calculate and output word-level timestamps.

## Outputs

- **text**: The raw transcription of the audio.
- **timestamps**: A formatted string containing word-level timestamps (e.g., `[0.00 - 0.50] Hello`). If no aligner is provided, this will return a status message.

## Usage Tips

- **Resampling**: This node automatically resamples audio to 16kHz internally using Torch, ensuring compatibility with the model even if your input audio is 44.1kHz or 48kHz.
- **Caching**: The model is cached in memory after the first run. Changing the `model_name`, `device`, or `precision` will trigger a reload.

## Example

1. Connect a **Load Audio** node to the **audio** input.
2. Select `Qwen3-ASR-1.7B` in **model_name**.
3. (Optional) Connect a **Qwen3 Forced Aligner Config** to the **forced_aligner** input to see timing data in the second output.