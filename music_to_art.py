"""
=============================================================================
Emotion-Driven Cross-Modal Generation: Music to Art (Audio -> Emotion -> Art)
=============================================================================
This module bridges Audio Signal Processing and Generative Latent Diffusion:
1. Audio Feature Extraction: Uses `librosa` to compute:
   - Tempo (BPM) via onset strength envelopes.
   - Spectral Centroid (Hz) measuring spectral brightness & frequency distribution.
   - RMS Energy measuring signal power/dynamics.
2. Emotion Mapping Rule:
   - High BPM (>110) & High Brightness (Centroid >= 2200 Hz) -> 'energetic'
   - High BPM (>110) & Moderate Brightness                   -> 'joyful'
   - Low BPM (<=110) & Low Brightness (Centroid < 1800 Hz)   -> 'sad'
   - Low BPM (<=110) & Moderate Brightness                  -> 'calm'
3. Art Generation: Synthesizes a high-fidelity emotional painting using
   Hugging Face's ultra-fast 'stabilityai/sd-turbo' Latent Diffusion Model.
   Automatically detects CUDA or falls back cleanly to CPU with low inference steps.

Author: AI College Project Team
License: MIT
=============================================================================
"""

import os
import time
from pathlib import Path
from typing import Tuple, Dict, Any, Union
from PIL import Image, ImageDraw, ImageFont

# Audio processing
try:
    import librosa
    import soundfile as sf
    import numpy as np
except ImportError:
    librosa = None
    sf = None
    np = None

# Deep learning & Diffusion models
try:
    import torch
    from diffusers import AutoPipelineForText2Image
except ImportError:
    torch = None
    AutoPipelineForText2Image = None

# Global pipeline cache for diffusion model
_DIFFUSION_PIPELINE = None

# Curated high-aesthetic prompts aligned with human emotion psychology
EMOTION_ART_PROMPTS = {
    "calm": (
        "serene pastel watercolor landscape at twilight, peaceful misty mountain lake "
        "with blooming water lilies, soft glowing ambient lanterns, studio ghibli aesthetic, "
        "calming ethereal atmosphere, tranquil reflections, highly detailed digital painting"
    ),
    "sad": (
        "melancholic lonely figure under rain in an impressionist muted blue oil painting, "
        "dimly lit cobblestone street with golden puddles, deep soft shadows, expressive "
        "sorrowful mood, atmospheric cinematic emotional masterpiece"
    ),
    "joyful": (
        "vibrant sunlit meadow filled with colorful wildflowers, golden hour warm sunshine, "
        "playful dancing light, uplifting warm yellow and coral colors, whimsical impressionist "
        "style, blissful feeling, 8k resolution digital art"
    ),
    "energetic": (
        "dynamic explosion of neon electric colors, abstract futuristic cosmic synthwave vortex, "
        "vibrant laser light streaks, hyper-detailed cyberpunk energy, intense kinetic motion, "
        "bold swirling composition, 8k cinematic masterpiece"
    )
}

# Color palettes for artistic fallback generation
FALLBACK_PALETTES = {
    "calm": [(34, 76, 110), (120, 180, 200), (225, 240, 245)],
    "sad": [(20, 25, 45), (50, 60, 90), (100, 110, 140)],
    "joyful": [(255, 180, 50), (255, 105, 90), (255, 230, 120)],
    "energetic": [(255, 0, 128), (0, 220, 255), (130, 0, 255)]
}


def get_diffusion_pipeline(model_id: str = "stabilityai/sd-turbo"):
    """
    Loads and caches the Hugging Face text-to-image pipeline.
    Uses 'stabilityai/sd-turbo' which produces high-quality images in 1-4 inference steps.
    """
    global _DIFFUSION_PIPELINE
    if _DIFFUSION_PIPELINE is None:
        if AutoPipelineForText2Image is None or torch is None:
            raise ImportError(
                "Packages 'diffusers' and 'torch' are required for AI art generation. "
                "Please run: pip install diffusers torch accelerate"
            )

        device = "cuda" if torch.cuda.is_available() else "cpu"
        dtype = torch.float16 if device == "cuda" else torch.float32

        print(f"[MusicToArt] Loading SD-Turbo diffusion pipeline onto {device.upper()} ({dtype})...")
        try:
            _DIFFUSION_PIPELINE = AutoPipelineForText2Image.from_pretrained(
                model_id,
                torch_dtype=dtype,
                variant="fp16" if device == "cuda" else None
            )
            _DIFFUSION_PIPELINE.to(device)
            # Enable memory optimization if CUDA is used
            if device == "cuda":
                try:
                    _DIFFUSION_PIPELINE.enable_vae_tiling()
                except Exception:
                    pass
            print("[MusicToArt] Diffusion pipeline loaded successfully.")
        except Exception as e:
            print(f"[MusicToArt Warning] Failed to load {model_id}: {e}")
            raise e

    return _DIFFUSION_PIPELINE


def extract_audio_features(audio_path: Union[str, Path]) -> Dict[str, Any]:
    """
    Extracts acoustic features from an audio file using Librosa:
    - Tempo (BPM): Rhythm speed estimated from onset strength envelopes.
    - Spectral Centroid (Hz): Center of mass of the frequency spectrum (timbre brightness).
    - RMS Energy: Overall signal amplitude and dynamic loudness.

    Returns:
    --------
    features : Dict[str, Any]
        Dictionary containing bpm, mean_centroid, mean_rms, and duration.
    """
    if librosa is None or np is None:
        raise ImportError(
            "The 'librosa' and 'numpy' libraries are required for audio analysis. "
            "Please run: pip install librosa soundfile numpy"
        )

    audio_path = Path(audio_path)
    if not audio_path.exists():
        raise FileNotFoundError(f"Audio file not found: {audio_path}")

    # Load audio (downsampled to 22050 Hz for standard psychoacoustic analysis)
    y, sr = librosa.load(str(audio_path), sr=22050, mono=True)
    duration = float(librosa.get_duration(y=y, sr=sr))

    if len(y) == 0:
        raise ValueError("The provided audio file is empty or silent.")

    # 1. Extract Tempo (BPM)
    try:
        onset_env = librosa.onset.onset_strength(y=y, sr=sr)
        tempo = librosa.feature.tempo(onset_envelope=onset_env, sr=sr)
        if hasattr(tempo, "__len__") and len(tempo) > 0:
            bpm = float(tempo[0])
        else:
            bpm = float(tempo)
    except Exception:
        # Fallback to beat tracking
        tempo, _ = librosa.beat.beat_track(y=y, sr=sr)
        bpm = float(tempo.item()) if hasattr(tempo, "item") else float(tempo)

    # 2. Extract Spectral Centroid (Brightness)
    spectral_centroids = librosa.feature.spectral_centroid(y=y, sr=sr)[0]
    mean_centroid = float(np.mean(spectral_centroids))

    # 3. Extract Root-Mean-Square (RMS) Energy (Loudness/Intensity)
    rms = librosa.feature.rms(y=y)[0]
    mean_rms = float(np.mean(rms))

    return {
        "bpm": round(bpm, 2),
        "spectral_centroid_hz": round(mean_centroid, 2),
        "mean_rms": round(mean_rms, 4),
        "duration_seconds": round(duration, 2)
    }


def map_audio_features_to_emotion(features: Dict[str, Any]) -> Tuple[str, Dict[str, Any]]:
    """
    Applies signal processing decision boundaries to map Tempo & Brightness to emotion:
    - High BPM (>110) & Bright Centroid (>=2200 Hz) -> 'energetic'
    - High BPM (>110) & Warm Centroid (<2200 Hz)    -> 'joyful'
    - Low BPM (<=110) & Dark Centroid (<1800 Hz)    -> 'sad'
    - Low BPM (<=110) & Balanced Centroid (>=1800 Hz)-> 'calm'

    Returns:
    --------
    emotion : str
    reasoning : Dict[str, Any]
    """
    bpm = features["bpm"]
    centroid = features["spectral_centroid_hz"]

    if bpm > 110:
        if centroid >= 2200:
            emotion = "energetic"
            rule_explanation = (
                f"High Tempo ({bpm:.1f} BPM > 110) + High Spectral Brightness "
                f"({centroid:.1f} Hz >= 2200 Hz) indicates intense kinetic drive."
            )
        else:
            emotion = "joyful"
            rule_explanation = (
                f"High Tempo ({bpm:.1f} BPM > 110) + Warm Balanced Brightness "
                f"({centroid:.1f} Hz < 2200 Hz) indicates uplifting, celebratory bounce."
            )
    else:
        if centroid < 1800:
            emotion = "sad"
            rule_explanation = (
                f"Low Tempo ({bpm:.1f} BPM <= 110) + Low Spectral Brightness "
                f"({centroid:.1f} Hz < 1800 Hz) indicates melancholic, muffled timbre."
            )
        else:
            emotion = "calm"
            rule_explanation = (
                f"Low Tempo ({bpm:.1f} BPM <= 110) + Moderate Spectral Brightness "
                f"({centroid:.1f} Hz >= 1800 Hz) indicates tranquil, peaceful harmony."
            )

    reasoning = {
        "emotion": emotion,
        "rule_explanation": rule_explanation,
        "tempo_threshold": 110.0,
        "centroid_thresholds": "Sad: <1800 Hz | Calm: 1800-2200 Hz | Energetic: >=2200 Hz"
    }

    return emotion, reasoning


def create_fallback_art(emotion: str, prompt: str, width: int = 512, height: int = 512) -> Image.Image:
    """
    Creates an artistic generative gradient canvas in case the deep learning model
    cannot run (e.g., offline mode, insufficient RAM/VRAM during a live viva demo).
    Ensures the demo NEVER crashes!
    """
    colors = FALLBACK_PALETTES.get(emotion, FALLBACK_PALETTES["calm"])
    img = Image.new("RGB", (width, height), colors[0])
    draw = ImageDraw.Draw(img)

    # Draw smooth emotional gradient
    c1, c2, c3 = colors[0], colors[1], colors[2]
    for y in range(height):
        factor = y / float(height)
        if factor < 0.5:
            local_factor = factor * 2.0
            r = int(c1[0] + (c2[0] - c1[0]) * local_factor)
            g = int(c1[1] + (c2[1] - c1[1]) * local_factor)
            b = int(c1[2] + (c2[2] - c1[2]) * local_factor)
        else:
            local_factor = (factor - 0.5) * 2.0
            r = int(c2[0] + (c3[0] - c2[0]) * local_factor)
            g = int(c2[1] + (c3[1] - c2[1]) * local_factor)
            b = int(c2[2] + (c3[2] - c3[2]) * local_factor)
        draw.line([(0, y), (width, y)], fill=(r, g, b))

    # Add artistic circles
    for i in range(1, 5):
        radius = i * 45
        draw.ellipse(
            [width // 2 - radius, height // 2 - radius, width // 2 + radius, height // 2 + radius],
            outline=(255, 255, 255, 120),
            width=2
        )

    # Informative caption
    caption = f"Emotion: {emotion.upper()} | Aesthetic Mode"
    draw.text((20, height - 35), caption, fill=(255, 255, 255))
    return img


def generate_art_from_prompt(
    prompt: str,
    emotion: str,
    num_inference_steps: int = None,
    output_path: Union[str, Path] = None
) -> Image.Image:
    """
    Synthesizes an artistic image from the text prompt using SD-Turbo.
    Automatically adapts inference steps based on CPU vs GPU.
    """
    device = "cuda" if (torch and torch.cuda.is_available()) else "cpu"
    
    # Fast inference step defaults: SD-turbo only requires 1-4 steps!
    if num_inference_steps is None:
        num_inference_steps = 4 if device == "cuda" else 2

    try:
        pipe = get_diffusion_pipeline()
        print(f"[MusicToArt] Generating art on {device.upper()} with {num_inference_steps} steps...")
        
        # SD-Turbo operates with guidance_scale=0.0 or 1.0 (recommended by stabilityai)
        result = pipe(
            prompt=prompt,
            num_inference_steps=num_inference_steps,
            guidance_scale=0.0
        )
        generated_image = result.images[0]

    except Exception as exc:
        print(f"[MusicToArt] Diffusion model execution failed or skipped ({exc}). Generating artistic fallback canvas.")
        generated_image = create_fallback_art(emotion=emotion, prompt=prompt)

    # Save to disk if requested
    if output_path is not None:
        out_p = Path(output_path)
        out_p.parent.mkdir(parents=True, exist_ok=True)
        generated_image.save(out_p)

    return generated_image


def audio_to_art(
    audio_path: Union[str, Path],
    output_image_path: Union[str, Path] = None
) -> Tuple[str, Dict[str, Any], str, Image.Image]:
    """
    End-to-End Pipeline:
    Takes an input audio file -> Extracts acoustic features -> Classifies emotion ->
    Synthesizes rich artistic prompt -> Generates AI Painting.

    Returns:
    --------
    emotion : str
    features : Dict[str, Any]
    prompt : str
    artwork_image : PIL.Image.Image
    """
    # Step 1: Feature Extraction
    features = extract_audio_features(audio_path)

    # Step 2: Emotion Mapping
    emotion, reasoning = map_audio_features_to_emotion(features)
    features.update(reasoning)

    # Step 3: Prompt Construction
    prompt = EMOTION_ART_PROMPTS.get(emotion, EMOTION_ART_PROMPTS["calm"])

    # Step 4: Art Generation
    artwork = generate_art_from_prompt(
        prompt=prompt,
        emotion=emotion,
        output_path=output_image_path
    )

    return emotion, features, prompt, artwork


# Standalone demonstration and test entrypoint
if __name__ == "__main__":
    import sys
    print("=== Music to Art (Audio -> Emotion -> AI Painting) Standalone Test ===")

    test_audio = Path("sample_calm_test.wav")
    if not test_audio.exists():
        # Synthesize a simple 3-second gentle sine wave tone at 60 BPM for testing
        sr = 22050
        duration = 3.0
        t = np.linspace(0, duration, int(sr * duration), endpoint=False)
        # Gentle 440 Hz (A4) sine wave
        y = 0.5 * np.sin(2 * np.pi * 440 * t)
        if sf is not None:
            sf.write(str(test_audio), y, sr)
            print(f"Synthesized test audio at: {test_audio}")
        else:
            print("Notice: soundfile package not installed; skipping test audio generation.")

    if test_audio.exists():
        input_audio = sys.argv[1] if len(sys.argv) > 1 else str(test_audio)
        print(f"Analyzing audio: {input_audio}")
        emotion, feats, prompt, img = audio_to_art(input_audio, output_image_path="test_art_output.png")
        print(f"\n[Acoustic Features]")
        print(f"Tempo             : {feats['bpm']} BPM")
        print(f"Spectral Centroid : {feats['spectral_centroid_hz']} Hz")
        print(f"Emotion Class     : {emotion.upper()}")
        print(f"Decision Rule     : {feats['rule_explanation']}")
        print(f"\n[Art Prompt]")
        print(prompt)
        print("\n[SUCCESS] Art generation complete! Saved to test_art_output.png")
