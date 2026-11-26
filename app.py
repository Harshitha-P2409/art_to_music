"""
=============================================================================
Emotion-Driven Cross-Modal Generation (Art & Music)
Interactive Web Application powered by Gradio
=============================================================================
This application demonstrates bi-directional cross-modal AI generation:
1. Art -> Music : Translates an uploaded artwork into an expressive 15-second
                  MIDI music composition using CLIP zero-shot emotion detection
                  and algorithmic musical synthesis.
2. Music -> Art : Translates an uploaded audio track into an AI-generated
                  painting by extracting acoustic features (Tempo, Spectral Centroid)
                  and conditioning a Latent Diffusion Model (SD-Turbo).

College Course: Artificial Intelligence & Deep Learning
Target Platforms: Local Machine (Windows, macOS, Linux) & Google Colab
=============================================================================
"""

import os
import sys
import argparse
from pathlib import Path
from PIL import Image

import gradio as gr

# Import our custom modular cross-modal pipelines
from art_to_music import image_to_music, SUPPORTED_EMOTIONS
from music_to_art import audio_to_art, EMOTION_ART_PROMPTS


# ---------------------------------------------------------------------------
# Tab 1 Handler: Art -> Music
# ---------------------------------------------------------------------------
def process_art_to_music(image):
    """
    Gradio callback for Image -> Emotion -> MIDI generation.
    Handles image input, runs CLIP zero-shot classification, and writes MIDI.
    """
    if image is None:
        return (
            "No Image Provided",
            {},
            "Please upload or drop an image first.",
            None
        )

    try:
        # Run the full image_to_music pipeline
        emotion, confidence, all_scores, midi_path, meta = image_to_music(image)

        # Format detection header
        status_text = f"**Detected Mood:** `{emotion.upper()}` ({confidence * 100:.1f}% confidence)"

        # Format readable music theory explanation for students/evaluators
        music_summary = f"""
### 🎵 Musical Synthesis Profile
- **Tempo:** `{meta['bpm']} BPM`
- **Scale / Mode:** `{meta['scale']}`
- **Duration:** `{meta['duration_seconds']}s` ({meta['total_beats']} beats)
- **Composition Style:** {meta['description']}
- **File Format:** Standard MIDI (.mid) playable in Windows Media Player, VLC, GarageBand, or any DAW.
        """

        return status_text, all_scores, music_summary, midi_path

    except Exception as exc:
        error_msg = f"⚠️ An error occurred during processing: {str(exc)}"
        print(f"[App Error - Art to Music]: {exc}")
        return error_msg, {}, "Failed to generate music.", None


# ---------------------------------------------------------------------------
# Tab 2 Handler: Music -> Art
# ---------------------------------------------------------------------------
def process_music_to_art(audio_file):
    """
    Gradio callback for Audio -> Emotion -> Art generation.
    Extracts acoustic features with Librosa and renders painting via SD-Turbo.
    """
    if audio_file is None:
        return (
            "No Audio Provided",
            "Please upload an audio file or record via microphone.",
            "No prompt generated.",
            None
        )

    try:
        # Run audio_to_art pipeline
        emotion, features, prompt_used, artwork = audio_to_art(audio_file)

        # Format detected mood status
        status_text = f"**Detected Mood:** `{emotion.upper()}`"

        # Format acoustic signal features for viva explanation
        acoustic_details = f"""
### 📊 Extracted Acoustic Features (Librosa)
- **Estimated Tempo (BPM):** `{features['bpm']} BPM`
- **Spectral Centroid (Brightness):** `{features['spectral_centroid_hz']} Hz`
- **Mean RMS Energy:** `{features['mean_rms']}`
- **Audio Duration:** `{features['duration_seconds']}s`
- **Decision Logic:** {features.get('rule_explanation', 'Feature-based mood mapping')}
        """

        return status_text, acoustic_details, prompt_used, artwork

    except Exception as exc:
        error_msg = f"⚠️ An error occurred during audio analysis: {str(exc)}"
        print(f"[App Error - Music to Art]: {exc}")
        return error_msg, "Failed to analyze audio.", "N/A", None


# ---------------------------------------------------------------------------
# Custom CSS for modern university project presentation
# ---------------------------------------------------------------------------
CUSTOM_CSS = """
.main-header {
    text-align: center;
    padding: 20px;
    background: linear-gradient(135deg, #1e3c72 0%, #2a5298 100%);
    color: white;
    border-radius: 12px;
    margin-bottom: 20px;
}
.main-header h1 {
    color: #ffffff !important;
    font-size: 2.2rem;
    font-weight: 700;
    margin-bottom: 8px;
}
.main-header p {
    color: #e0e8f9 !important;
    font-size: 1.05rem;
    margin: 0;
}
.project-badge {
    display: inline-block;
    background-color: rgba(255, 255, 255, 0.2);
    padding: 4px 12px;
    border-radius: 16px;
    font-size: 0.85rem;
    margin-top: 8px;
}
"""

# ---------------------------------------------------------------------------
# Build Gradio Blocks User Interface
# ---------------------------------------------------------------------------
def create_ui():
    with gr.Blocks(css=CUSTOM_CSS, title="Emotion-Driven Cross-Modal AI (Art & Music)") as demo:
        # Header banner
        gr.HTML("""
        <div class="main-header">
            <h1>🎨 Emotion-Driven Cross-Modal Generation 🎵</h1>
            <p>Bridging Visual Arts, Symbolic Music, and Generative Latent Diffusion through Emotional Semantics</p>
            <div class="project-badge">Artificial Intelligence Course Project &bull; Zero-Shot CLIP &bull; Librosa &bull; MIDIUtil &bull; SD-Turbo</div>
        </div>
        """)

        with gr.Tabs():
            # =======================================================================
            # TAB 1: Art -> Music (Image to MIDI)
            # =======================================================================
            with gr.TabItem("🖼️ ➔ 🎼 Art → Music (Image to MIDI)"):
                gr.Markdown("""
                Upload an image or painting. The system analyzes visual emotion using **OpenAI CLIP (zero-shot classification)** 
                and composes a synchronized **15-second algorithmic MIDI piece** mirroring that mood.
                """)
                
                with gr.Row():
                    # Left Column: Inputs
                    with gr.Column(scale=1):
                        image_input = gr.Image(
                            type="pil",
                            label="Input Image (Art / Photograph)",
                            sources=["upload", "clipboard"]
                        )
                        btn_art_to_music = gr.Button("🎵 Generate Music from Art", variant="primary", size="lg")
                        clear_art_btn = gr.ClearButton(components=[image_input], value="Reset")

                    # Right Column: Outputs
                    with gr.Column(scale=1):
                        emotion_output_art = gr.Markdown(value="*Upload an image and click 'Generate Music'*")
                        scores_output = gr.Label(num_top_classes=4, label="Emotion Confidence Distribution (CLIP)")
                        music_info_output = gr.Markdown()
                        midi_file_output = gr.File(label="📥 Download Synthesized MIDI File (.mid)")

                # Event binding
                btn_art_to_music.click(
                    fn=process_art_to_music,
                    inputs=[image_input],
                    outputs=[emotion_output_art, scores_output, music_info_output, midi_file_output]
                )

            # =======================================================================
            # TAB 2: Music -> Art (Audio to Painting)
            # =======================================================================
            with gr.TabItem("🎼 ➔ 🖼️ Music → Art (Audio to Painting)"):
                gr.Markdown("""
                Upload an audio track or record via microphone. The system extracts acoustic features 
                (**Tempo/BPM** and **Spectral Centroid**) via **Librosa**, maps them to an emotional mood, 
                and generates a high-resolution painting using **Stable Diffusion Turbo (SD-Turbo)**.
                """)
                
                with gr.Row():
                    # Left Column: Inputs
                    with gr.Column(scale=1):
                        audio_input = gr.Audio(
                            sources=["upload", "microphone"],
                            type="filepath",
                            label="Input Audio File (.wav or .mp3)"
                        )
                        btn_music_to_art = gr.Button("🎨 Synthesize Art from Audio", variant="primary", size="lg")
                        clear_music_btn = gr.ClearButton(components=[audio_input], value="Reset")

                    # Right Column: Outputs
                    with gr.Column(scale=1):
                        emotion_output_music = gr.Markdown(value="*Upload an audio clip and click 'Synthesize Art'*")
                        features_output = gr.Markdown()
                        prompt_output = gr.Textbox(
                            label="Synthesized Prompt for Generative Diffusion",
                            lines=3,
                            interactive=False
                        )
                        art_image_output = gr.Image(label="Synthesized AI Artwork", type="pil")

                # Event binding
                btn_music_to_art.click(
                    fn=process_music_to_art,
                    inputs=[audio_input],
                    outputs=[emotion_output_music, features_output, prompt_output, art_image_output]
                )

        # =======================================================================
        # Course Viva Reference & Documentation Section
        # =======================================================================
        with gr.Accordion("📚 Project Architecture & College Viva Reference Guide", open=False):
            gr.Markdown("""
            ### Core Theoretical Concepts for Examination / Viva

            #### 1. What is Cross-Modal Generation?
            Cross-modal learning maps information between disparate sensory modalities (e.g., Vision $\\leftrightarrow$ Audio). 
            In this project, human **emotion** acts as the latent semantic bridge connecting both realms.

            #### 2. CLIP Zero-Shot Classification (`art_to_music.py`)
            - **Model:** `openai/clip-vit-base-patch32`
            - **Mechanism:** CLIP projects images and text captions into a shared embedding space. By comparing the cosine similarity between the image embedding and candidate emotion prompt embeddings (`"joyful"`, `"sad"`, `"calm"`, `"energetic"`), it performs zero-shot prediction without task-specific retraining.

            #### 3. Acoustic Signal Processing (`music_to_art.py`)
            - **Tempo (BPM):** Estimated by computing the onset strength envelope of the audio waveform.
            - **Spectral Centroid ($f_c$):** Represents the "center of mass" of the frequency spectrum:
              $$f_c = \\frac{\\sum_{k} S(k) \\cdot f(k)}{\\sum_{k} S(k)}$$
              Higher values correspond to "bright/sharp" sounds (brass, cymbals, distorted guitars); lower values correspond to "mellow/dark" sounds (piano, cello, warm pads).

            #### 4. Latent Diffusion Models (SD-Turbo)
            - **Architecture:** Latent Diffusion Model (LDM) distilled using Adversarial Diffusion Distillation (ADD).
            - **Advantage:** Generates high-fidelity artwork in only 1–4 inference steps (unlike classical diffusion models requiring 50 steps), making it lightweight enough to run on college laptops and CPU environments.

            #### 5. Symbolic MIDI Synthesis (`midiutil`)
            - Generates event-based symbolic music (pitches, velocities, timestamps, durations) rather than raw waveforms, allowing lightweight files (~1 KB) with infinite audio fidelity when rendered by any standard sound synthesizer.
            """)

    return demo


# ---------------------------------------------------------------------------
# Application Entrypoint
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Emotion-Driven Cross-Modal Generation Web App")
    parser.add_argument("--share", action="store_true", default=True, help="Create a public Gradio shareable link (ideal for Google Colab)")
    parser.add_argument("--port", type=int, default=7860, help="Local port to bind server")
    args = parser.parse_args()

    app = create_ui()
    print("==================================================================")
    print(" Starting Emotion-Driven Cross-Modal Generation Web Interface")
    print(f" Local URL : http://127.0.0.1:{args.port}")
    print(f" Share Mode: {args.share}")
    print("==================================================================")

    # Launch Gradio app
    app.launch(
        server_name="0.0.0.0",
        server_port=args.port,
        share=args.share
    )
