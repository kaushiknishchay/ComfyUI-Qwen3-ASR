# Qwen3 Forced Aligner Config

The **Qwen3 Forced Aligner Config** node is a helper node used to initialize the Qwen3-ForcedAligner-0.6B model. This configuration is required if you want the **Qwen3 ASR Transcriber** to output word-level timestamps.

## Parameters

- **model_name**: The directory name of the Forced Aligner model located in `models/diffusion_models/Qwen3-ASR/`.
- **device**: The hardware device to run the aligner on. This should generally match the device used by the Transcriber node.
- **precision**: The floating-point precision for the aligner model.

## Outputs

- **aligner_config**: A configuration object that bundles the model path and loading arguments. Connect this to the `forced_aligner` input of the **Qwen3 ASR Transcriber**.

## Why use this?

Standard ASR models generate text but often lack precise timing for when each word was spoken. The Forced Aligner is a specialized model that takes the generated text and the original audio to "align" them, providing highly accurate start and end times for every word.

## Requirements

Ensure you have downloaded the `Qwen3-ForcedAligner-0.6B` model into your `models/diffusion_models/Qwen3-ASR/` folder.