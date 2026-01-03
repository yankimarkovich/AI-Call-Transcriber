"""
AI Call Transcriber - Gradio UI
"""

import os
import gradio as gr
from src.agent import create_agent

# Load API keys from environment
OPENAI_API_KEY = os.environ.get("OPENAI_API_KEY")
HF_TOKEN = os.environ.get("HF_TOKEN")


def process_audio(audio_file: str, num_speakers: int) -> tuple:
    """Handle audio upload and run transcription agent."""
    if not audio_file:
        return "Please upload an audio file.", None, ""
    if not OPENAI_API_KEY:
        return "Error: OPENAI_API_KEY not configured.", None, ""
    if not HF_TOKEN:
        return "Error: HF_TOKEN not configured.", None, ""

    agent = create_agent(OPENAI_API_KEY, HF_TOKEN)

    # Convert num_speakers: 0 means auto-detect
    speakers = num_speakers if num_speakers > 0 else None

    result = agent.process(audio_file, num_speakers=speakers)

    return result.message, result.download_path, result.transcript


def create_ui():
    """Create the Gradio interface."""
    with gr.Blocks(title="AI Call Transcriber", theme=gr.themes.Soft()) as app:
        gr.Markdown("""
        # AI Call Transcriber
        Upload an audio file to get a transcription with speaker identification.
        """)

        with gr.Row():
            with gr.Column(scale=1):
                audio_input = gr.Audio(
                    label="Upload Audio File",
                    type="filepath"
                )
                num_speakers_input = gr.Slider(
                    minimum=0,
                    maximum=10,
                    step=1,
                    value=2,
                    label="Number of Speakers",
                    info="Set to 0 for auto-detect. For phone calls, usually 2."
                )
                transcribe_btn = gr.Button("Transcribe", variant="primary", size="lg")

            with gr.Column(scale=2):
                status_output = gr.Textbox(label="Status", interactive=False)
                transcript_output = gr.Textbox(
                    label="Transcript",
                    lines=15,
                    interactive=False,
                    show_copy_button=True
                )
                download_output = gr.File(label="Download Transcript")

        transcribe_btn.click(
            fn=process_audio,
            inputs=[audio_input, num_speakers_input],
            outputs=[status_output, download_output, transcript_output]
        )

    return app


if __name__ == "__main__":
    app = create_ui()
    app.launch(server_name="0.0.0.0", server_port=7860)
