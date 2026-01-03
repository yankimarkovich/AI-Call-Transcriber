---
title: AI Call Transcriber
emoji: 🎙️
colorFrom: blue
colorTo: purple
sdk: gradio
sdk_version: "4.19.2"
app_file: app.py
pinned: false
---

# AI Call Transcriber

AI-powered audio call transcriber with speaker identification.

## Project Structure

```
AI-Call-Transcriber/
├── app.py                 # Gradio UI (entry point)
├── src/
│   ├── __init__.py
│   ├── agent.py           # Transcription agent
│   └── tools.py           # Audio processing utilities
├── requirements.txt
├── README.md
└── space.yaml             # HuggingFace Space config
```

## Features

- Speaker diarization (pyannote.audio)
- Transcription (OpenAI Whisper API)
- Organized output with participant labels
- Downloadable transcript
- Fallback handling

## Deploy to Hugging Face Space (Terminal)

### 1. Install HuggingFace CLI

```bash
pip install huggingface_hub
```

### 2. Login to HuggingFace

```bash
huggingface-cli login
```

### 3. Upload to Your Space

```bash
cd C:\Users\yanki\Code\GitHub\AI-Call-Transcriber
huggingface-cli upload YOUR_USERNAME/YOUR_SPACE_NAME . . --repo-type space
```

Replace `YOUR_USERNAME/YOUR_SPACE_NAME` with your actual space path (e.g., `johndoe/call-transcriber`).

### Alternative: Git Push

```bash
# Clone your existing space
git clone https://huggingface.co/spaces/YOUR_USERNAME/YOUR_SPACE_NAME
cd YOUR_SPACE_NAME

# Copy files
cp -r /path/to/AI-Call-Transcriber/* .

# Push
git add .
git commit -m "Deploy AI Call Transcriber"
git push
```

## Local Development

```bash
# Create virtual environment
python -m venv venv
venv\Scripts\activate  # Windows

# Install dependencies
pip install -r requirements.txt

# Run
python app.py
```

## Required API Keys

- **OpenAI API Key**: [platform.openai.com](https://platform.openai.com)
- **HuggingFace Token**: [huggingface.co/settings/tokens](https://huggingface.co/settings/tokens)
  - Must accept pyannote terms: [pyannote/speaker-diarization-3.1](https://huggingface.co/pyannote/speaker-diarization-3.1)

## Cost

- pyannote: Free (runs on HF Space)
- OpenAI Whisper: ~$0.006/min
