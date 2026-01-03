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

## Features

- Speaker diarization using pyannote.audio 3.1
- Transcription using Groq Whisper large-v3 (better accuracy than whisper-1)
- Word-level timestamps for precise speaker attribution
- Organized output with participant labels (Participant 1, Participant 2, etc.)
- Downloadable transcript file
- Multi-language support (auto-detection)
- Fallback handling for errors

## Project Structure

```
AI-Call-Transcriber/
├── app.py                 # Gradio UI (entry point)
├── src/
│   ├── __init__.py
│   ├── agent.py           # Transcription agent
│   └── tools.py           # Audio processing utilities
├── deploy.py              # HuggingFace deployment script
├── requirements.txt
└── README.md
```

## Quick Deploy to HuggingFace Space

```bash
python deploy.py --token YOUR_HF_TOKEN --groq-key YOUR_GROQ_KEY
```

## Local Development

```bash
# Create virtual environment
python -m venv venv
venv\Scripts\activate  # Windows
source venv/bin/activate  # Linux/Mac

# Install dependencies
pip install -r requirements.txt

# Set environment variables
export GROQ_API_KEY=your_groq_key
export HF_TOKEN=your_hf_token

# Run
python app.py
```

## Required API Keys

### 1. Groq API Key (Free)
Get your free API key at [console.groq.com](https://console.groq.com/keys)

### 2. HuggingFace Token
Get your token at [huggingface.co/settings/tokens](https://huggingface.co/settings/tokens)

**Important:** You must accept the terms for these gated models:
- [pyannote/speaker-diarization-3.1](https://huggingface.co/pyannote/speaker-diarization-3.1)
- [pyannote/segmentation-3.0](https://huggingface.co/pyannote/segmentation-3.0)

## Cost

- **pyannote:** Free (runs locally on HF Space)
- **Groq Whisper:** Free tier available

## Tech Stack

- **Transcription:** Groq Whisper large-v3
- **Speaker Diarization:** pyannote.audio 3.1
- **UI:** Gradio
- **Hosting:** HuggingFace Spaces
