# Transcribe long audio files using OpenAI's Transcription API with chunking to handle duration limits.

import json
import math
import os
import sys
import time
import tempfile
from pathlib import Path

from openai import OpenAI
from pydub import AudioSegment

# Initialize OpenAI client using API key from environment
client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

# Max chunk duration (safe buffer under ~22 min API limit)
CHUNK_DURATION_MINUTES = 20
CHUNK_DURATION_MS = CHUNK_DURATION_MINUTES * 60 * 1000


def validate_inputs(audio_path: str) -> Path:
    """Validate API key and audio file existence"""
    path = Path(audio_path)
    if not os.getenv("OPENAI_API_KEY"):
        raise ValueError("OPENAI_API_KEY is not set.")
    if not path.exists():
        raise FileNotFoundError(f"File not found: {audio_path}")
    return path


def load_audio(audio_path: Path) -> AudioSegment:
    """Load audio file into pydub AudioSegment"""
    return AudioSegment.from_file(audio_path)


def split_audio(audio: AudioSegment):
    """Split audio into <=20 min chunks to satisfy API limits"""
    chunks = []
    total_ms = len(audio)
    total_chunks = math.ceil(total_ms / CHUNK_DURATION_MS)

    for i in range(total_chunks):
        start = i * CHUNK_DURATION_MS
        end = min(start + CHUNK_DURATION_MS, total_ms)
        chunk = audio[start:end]
        chunks.append((chunk, i))  # store chunk + index

    return chunks


def export_chunk(chunk: AudioSegment, index: int, temp_dir: str):
    """Export chunk to a temporary mp3 file for API upload"""
    path = os.path.join(temp_dir, f"chunk_{index}.mp3")
    chunk.export(path, format="mp3")
    return path


def transcribe_chunk(chunk_path: str):
    """Call OpenAI transcription API on a single chunk"""
    with open(chunk_path, "rb") as audio_file:
        transcript = client.audio.transcriptions.create(
            model="gpt-4o-transcribe",
            file=audio_file,
        )
    return transcript.text.strip()


def combine_texts(text_chunks):
    """Combine chunk transcripts into one continuous string"""
    return " ".join(text_chunks)


def save_outputs(full_text: str, audio_path: Path):
    """Save combined transcript as JSON + TXT in same directory"""
    base = audio_path.with_suffix("")

    json_path = base.parent / f"{base.name}_transcript.json"
    txt_path = base.parent / f"{base.name}_transcript.txt"

    # Save plain text version
    with open(txt_path, "w", encoding="utf-8") as f:
        f.write(full_text)

    # Save structured JSON version
    with open(json_path, "w", encoding="utf-8") as f:
        json.dump({"text": full_text}, f, indent=2, ensure_ascii=False)

    return json_path, txt_path


def print_preview(text):
    """Print first 500 characters for quick inspection"""
    print("\nPreview:\n")
    preview = text[:500]
    print(preview + ("..." if len(text) > 500 else ""))


def transcribe_audio(audio_path: str):
    """Main pipeline: split → transcribe → combine → save"""
    path = validate_inputs(audio_path)

    print(f"Transcribing: {path.name}")

    audio = load_audio(path)
    chunks = split_audio(audio)

    print(f"Total chunks: {len(chunks)}")

    texts = []
    total_api_time = 0
    overall_start = time.perf_counter()

    # Temporary directory auto-cleans chunk files
    with tempfile.TemporaryDirectory() as temp_dir:
        for chunk, index in chunks:
            print(f"\nProcessing chunk {index + 1}/{len(chunks)}")

            chunk_path = export_chunk(chunk, index, temp_dir)

            # Measure API latency per chunk
            start = time.perf_counter()
            text = transcribe_chunk(chunk_path)
            end = time.perf_counter()

            elapsed = end - start
            total_api_time += elapsed

            print(f"⏱️ Chunk {index + 1}: {elapsed:.2f}s")

            texts.append(text)

    # Combine all chunk outputs into one transcript
    full_text = combine_texts(texts)

    json_path, txt_path = save_outputs(full_text, path)

    overall_end = time.perf_counter()

    print("\n====== SUMMARY ======")
    print(f"Total API time: {total_api_time:.2f}s")
    print(f"Total wall time: {overall_end - overall_start:.2f}s")

    print(f"\nSaved JSON → {json_path}")
    print(f"Saved TXT  → {txt_path}")

    print_preview(full_text)


if __name__ == "__main__":
    # CLI usage: python3 script.py <audio_file>
    if len(sys.argv) < 2:
        print("Usage: python3 script.py <audio_file>")
        sys.exit(1)

    audio_file = sys.argv[1]
    transcribe_audio(audio_file)