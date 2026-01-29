import torch
import gc
import numpy as np
import os
import torch.nn.functional as F
import folder_paths
import comfy.utils
from comfy import model_management
from qwen_asr import Qwen3ASRModel

# Global cache for models to prevent reloading
_QWEN3_MODEL_CACHE = {}

SUPPORTED_LANGUAGES = [
    "Chinese",
    "English",
    "Cantonese",
    "Arabic",
    "German",
    "French",
    "Spanish",
    "Portuguese",
    "Indonesian",
    "Italian",
    "Korean",
    "Russian",
    "Thai",
    "Vietnamese",
    "Japanese",
    "Turkish",
    "Hindi",
    "Malay",
    "Dutch",
    "Swedish",
    "Danish",
    "Finnish",
    "Polish",
    "Czech",
    "Filipino",
    "Persian",
    "Greek",
    "Romanian",
    "Hungarian",
    "Macedonian"
]

def get_qwen_model_list():
    all_files = folder_paths.get_filename_list("diffusion_models")
    qwen_models = set()
    for f in all_files:
        # Look for the specific structure: Qwen3-ASR/ModelFolder/
        if "Qwen3-ASR" in f:
            f = f.replace("\\", "/")
            parts = f.split("/")
            if len(parts) >= 2:
                # We want the path up to the second part (e.g. Qwen3-ASR/Qwen3-ASR-1.7B)
                qwen_models.add("/".join(parts[:2]))
    
    res = sorted(list(qwen_models))
    return res if res else ["None Found (Place in models/diffusion_models/Qwen3-ASR/)"]

def resolve_qwen_path(model_name):
    # Try standard Comfy resolution first
    path = folder_paths.get_full_path("diffusion_models", model_name)
    if path:
        return os.path.dirname(path) if os.path.isfile(path) else path
    
    # Fallback: manual join with configured diffusion_models paths for directory support
    for base in folder_paths.get_folder_paths("diffusion_models"):
        full_path = os.path.join(base, model_name)
        if os.path.exists(full_path):
            return full_path
    return model_name

class Qwen3ForcedAlignerConfig:
    """
    Provides configuration for the Qwen3 Forced Aligner.
    This config is passed to the main ASR node to enable timestamp generation.
    """
    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "model_name": (get_qwen_model_list(), {"tooltip": "The Qwen3 Forced Aligner model to use for generating timestamps."}),
                "device": (["cuda", "cpu"], {"default": "cuda", "tooltip": "The device to run the aligner on."}),
                "precision": (["bf16", "fp16", "fp32"], {"default": "bf16", "tooltip": "The numerical precision to use for the aligner model."}),
                "flash_attention_2": ("BOOLEAN", {"default": False, "tooltip": "Enable Flash Attention 2 for faster inference and lower VRAM usage (requires compatible GPU and bf16/fp16)."}),
            },
        }

    RETURN_TYPES = ("QWEN3_ALIGNER_CONF",)
    RETURN_NAMES = ("aligner_config",)
    FUNCTION = "get_config"
    CATEGORY = "Qwen3-ASR"

    def get_config(self, model_name, device, precision, flash_attention_2):
        dtype_map = {
            "bf16": torch.bfloat16,
            "fp16": torch.float16,
            "fp32": torch.float32
        }
        
        load_path = resolve_qwen_path(model_name)
        return ({
            "model_name": load_path,
            "kwargs": {
                "device_map": device,
                "dtype": dtype_map[precision],
                "attn_implementation": "flash_attention_2" if flash_attention_2 else "sdpa"
            }
        },)

class Qwen3ASRTranscriber:
    """
    Performs ASR inference using Qwen3-ASR models.
    Optionally uses a Forced Aligner config to generate word-level timestamps.
    """
    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "audio": ("AUDIO", {"tooltip": "The input audio to be transcribed."}),
                "model_name": (get_qwen_model_list(), {"tooltip": "The Qwen3 ASR model to use for transcription."}),
                "language": (["auto"] + SUPPORTED_LANGUAGES, {"default": "auto", "tooltip": "The language of the audio. Set to 'auto' for automatic language detection."}),
                "device": (["cuda", "cpu"], {"default": "cuda", "tooltip": "The device to run the ASR model on."}),
                "precision": (["bf16", "fp16", "fp32"], {"default": "bf16", "tooltip": "The numerical precision to use for the ASR model."}),
                "max_new_tokens": ("INT", {"default": 256, "min": 1, "max": 4096, "tooltip": "The maximum number of tokens to generate in the transcription."}),
                "flash_attention_2": ("BOOLEAN", {"default": False, "tooltip": "Enable Flash Attention 2 for faster inference and lower VRAM usage."}),
                "chunk_size": ("INT", {"default": 30, "min": 0, "max": 300, "tooltip": "Process audio in chunks of this many seconds. Set to 0 to disable chunking (not recommended for long audio)."}),
                "overlap": ("INT", {"default": 2, "min": 0, "max": 10, "tooltip": "Overlap between chunks in seconds to maintain context."}),
            },
            "optional": {
                "forced_aligner": ("QWEN3_ALIGNER_CONF", {"tooltip": "Optional configuration for the Qwen3 Forced Aligner to generate word-level timestamps."}),
            }
        }

    RETURN_TYPES = ("STRING", "STRING")
    RETURN_NAMES = ("text", "timestamps")
    FUNCTION = "transcribe"
    CATEGORY = "Qwen3-ASR"

    def transcribe(self, audio, model_name, language, device, precision, max_new_tokens, flash_attention_2, chunk_size, overlap, forced_aligner=None):
        global _QWEN3_MODEL_CACHE
        
        # Support for ComfyUI interruption (Cancel button)
        model_management.throw_exception_if_processing_interrupted()

        # Heuristic: If ComfyUI has cleared its internal model registry, 
        # we should clear our custom cache to respect the "Clear Cache" action.
        if len(model_management.current_loaded_models) == 0 and len(_QWEN3_MODEL_CACHE) > 0:
            print("[Qwen3-ASR] ComfyUI cache clear detected. Purging ASR models...")
            _QWEN3_MODEL_CACHE.clear()

        dtype_map = {
            "bf16": torch.bfloat16,
            "fp16": torch.float16,
            "fp32": torch.float32
        }
        dtype = dtype_map[precision]
        lang_param = None if not language or language.lower() == "auto" else language

        # Handle ComfyUI CPU mode or Low VRAM settings
        if model_management.cpu_mode():
            device = "cpu"
        elif device == "cuda":
            # Use the specific device assigned by ComfyUI (handles multi-GPU)
            device = model_management.get_torch_device()

        # Create a unique cache key based on model settings
        aligner_name = forced_aligner["model_name"] if forced_aligner else "none"
        cache_key = f"{model_name}_{str(device)}_{precision}_{aligner_name}_{flash_attention_2}"

        try:
            if cache_key not in _QWEN3_MODEL_CACHE:
                # Clear previous models to prevent OOM
                if model_management.vram_state in [model_management.VRAMState.LOW_VRAM, model_management.VRAMState.NO_VRAM]:
                    print(f"[Qwen3-ASR] Low VRAM mode active ({model_management.vram_state.name}). Clearing cache...")
                    _QWEN3_MODEL_CACHE.clear()
                    model_management.soft_empty_cache()

                load_path = resolve_qwen_path(model_name)
                print(f"[Qwen3-ASR] Loading model from: {load_path}...")
                
                # Ensure we have enough memory for the load
                if device != "cpu":
                    model_management.free_memory(model_management.minimum_inference_memory(), device)
                
                loader_kwargs = {
                    "pretrained_model_name_or_path": load_path,
                    "dtype": dtype,
                    "device_map": device,
                    "max_new_tokens": max_new_tokens,
                    "attn_implementation": "flash_attention_2" if flash_attention_2 else "sdpa",
                }

                if forced_aligner:
                    loader_kwargs["forced_aligner"] = forced_aligner["model_name"]
                    loader_kwargs["forced_aligner_kwargs"] = forced_aligner["kwargs"]

                _QWEN3_MODEL_CACHE[cache_key] = Qwen3ASRModel.from_pretrained(**loader_kwargs)
                # Clean up fragmentation after heavy load
                model_management.soft_empty_cache()

            model = _QWEN3_MODEL_CACHE[cache_key]

            # Prepare Audio
            # ComfyUI audio format: {"waveform": [Batch, Channels, Samples], "sample_rate": int}
            waveform = audio["waveform"]
            sample_rate = audio["sample_rate"]

            # Convert to mono if necessary and remove batch dim
            if waveform.ndim == 3:
                waveform = waveform[0]
            
            if waveform.shape[0] > 1:
                waveform = torch.mean(waveform, dim=0)
            else:
                waveform = waveform[0]

            # Resample to 16000Hz using torch to avoid buggy librosa calls in Python 3.13
            if sample_rate != 16000:
                # F.interpolate expects [batch, channels, length]
                w = waveform.view(1, 1, -1)
                w = F.interpolate(w, size=int(w.shape[-1] * 16000 / sample_rate), mode='linear', align_corners=False)
                waveform = w.view(-1)
                sample_rate = 16000

            # Chunking Logic
            full_waveform_np = waveform.cpu().numpy()
            total_samples = len(full_waveform_np)
            
            if chunk_size > 0:
                chunk_samples = chunk_size * sample_rate
                overlap_samples = overlap * sample_rate
                step_samples = chunk_samples - overlap_samples
                
                num_chunks = max(1, int(np.ceil((total_samples - overlap_samples) / step_samples)))
                pbar = comfy.utils.ProgressBar(num_chunks)
                
                all_texts = []
                all_timestamps = []
                
                for i in range(num_chunks):
                    model_management.throw_exception_if_processing_interrupted()
                    
                    start = i * step_samples
                    end = min(start + chunk_samples, total_samples)
                    chunk_np = full_waveform_np[start:end]
                    
                    # Run Inference on Chunk
                    results = model.transcribe(
                        audio=[(chunk_np, sample_rate)],
                        language=lang_param,
                        return_time_stamps=True if forced_aligner else False
                    )
                    
                    res = results[0]
                    all_texts.append(res.text)
                    
                    # Offset timestamps by the chunk start time
                    chunk_offset_s = start / sample_rate
                    if forced_aligner and hasattr(res, 'time_stamps') and res.time_stamps:
                        for ts in res.time_stamps:
                            all_timestamps.append(f"[{ts.start_time + chunk_offset_s:.2f} - {ts.end_time + chunk_offset_s:.2f}] {ts.text}")
                    
                    pbar.update(1)
                
                transcription_text = " ".join(all_texts)
                timestamp_output = "\n".join(all_timestamps) if all_timestamps else "No timestamps generated."
            else:
                # Single pass for short audio
                results = model.transcribe(
                    audio=[(full_waveform_np, sample_rate)],
                    language=lang_param,
                    return_time_stamps=True if forced_aligner else False
                )
                res = results[0]
                transcription_text = res.text
                
                timestamp_output = ""
                if forced_aligner and hasattr(res, 'time_stamps') and res.time_stamps:
                    ts_lines = [f"[{ts.start_time:.2f} - {ts.end_time:.2f}] {ts.text}" for ts in res.time_stamps]
                    timestamp_output = "\n".join(ts_lines)
                else:
                    timestamp_output = "No timestamps generated."

            # If in NO_VRAM mode, don't keep the model in cache after execution
            if model_management.vram_state == model_management.VRAMState.NO_VRAM:
                print("[Qwen3-ASR] NO_VRAM mode: Purging model after use.")
                _QWEN3_MODEL_CACHE.pop(cache_key, None)
                model_management.soft_empty_cache()

            return (transcription_text, timestamp_output)

        except Exception as e:
            if isinstance(e, model_management.InterruptProcessingException):
                # Cleanup on interrupt
                if model_management.vram_state == model_management.VRAMState.NO_VRAM:
                    _QWEN3_MODEL_CACHE.pop(cache_key, None)
                gc.collect()
                model_management.soft_empty_cache()
            raise

NODE_CLASS_MAPPINGS = {
    "Qwen3ASRTranscriber": Qwen3ASRTranscriber,
    "Qwen3ForcedAlignerConfig": Qwen3ForcedAlignerConfig
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "Qwen3ASRTranscriber": "Qwen3 ASR Transcriber",
    "Qwen3ForcedAlignerConfig": "Qwen3 Forced Aligner Config"
}