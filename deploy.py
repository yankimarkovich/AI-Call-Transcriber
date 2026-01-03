"""
Deploy script for Hugging Face Space.

Usage:
  python deploy.py --token YOUR_HF_TOKEN --groq-key YOUR_GROQ_KEY
"""

import argparse
import os
from huggingface_hub import HfApi, create_repo, upload_folder, login

# Configuration
SPACE_NAME = "AI-Call-Transcriber"
USERNAME = "YankiMarkovich"
REPO_ID = f"{USERNAME}/{SPACE_NAME}"


def main():
    parser = argparse.ArgumentParser(description="Deploy to HuggingFace Space")
    parser.add_argument("--token", required=True, help="HuggingFace token")
    parser.add_argument("--groq-key", help="Groq API key (optional, sets as secret)")
    args = parser.parse_args()

    print("=" * 50)
    print("AI Call Transcriber - HuggingFace Deployment")
    print("=" * 50)

    # Step 1: Login
    print("\n[1/4] Login to HuggingFace")
    login(token=args.token)
    api = HfApi()
    print("[OK] Logged in successfully!")

    # Step 2: Create Space
    print(f"\n[2/4] Creating Space: {REPO_ID}")
    try:
        create_repo(
            repo_id=REPO_ID, repo_type="space", space_sdk="gradio", exist_ok=True
        )
        print(f"[OK] Space created/verified: https://huggingface.co/spaces/{REPO_ID}")
    except Exception as e:
        print(f"Error creating space: {e}")
        return

    # Step 3: Set environment variables
    print("\n[3/4] Setting secrets")

    # Always set HF_TOKEN for pyannote
    api.add_space_secret(repo_id=REPO_ID, key="HF_TOKEN", value=args.token)
    print("[OK] HF_TOKEN secret added!")

    if args.groq_key:
        api.add_space_secret(repo_id=REPO_ID, key="GROQ_API_KEY", value=args.groq_key)
        print("[OK] GROQ_API_KEY secret added!")

    # Step 4: Upload code
    print(f"\n[4/4] Uploading code to {REPO_ID}")
    project_dir = os.path.dirname(os.path.abspath(__file__))

    upload_folder(
        folder_path=project_dir,
        repo_id=REPO_ID,
        repo_type="space",
        ignore_patterns=["venv/*", "__pycache__/*", "*.pyc", ".git/*", "deploy.py"],
    )

    print("[OK] Code uploaded successfully!")
    print("\n" + "=" * 50)
    print("Your Space is live at:")
    print(f"   https://huggingface.co/spaces/{REPO_ID}")
    print("=" * 50)


if __name__ == "__main__":
    main()
