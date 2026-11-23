import os
import json
import whisper

TRANSCRIPT_DIR = "data/transcripts"
AUDIO_DIR = "audio"
MODEL_SIZE = "small"  # 'small' handles Hindi/Hinglish much better than 'base'

os.makedirs(TRANSCRIPT_DIR, exist_ok=True)


def transcribe_lectures():
    print(f"Loading Whisper '{MODEL_SIZE}' model...")
    model = whisper.load_model(MODEL_SIZE)

    audio_files = [f for f in os.listdir(AUDIO_DIR) if f.endswith((".mp3", ".wav", ".m4a"))]
    if not audio_files:
        print(f"No audio files found in '{AUDIO_DIR}/'.")
        return

    print(f"Found {len(audio_files)} audio file(s).\n")

    for idx, filename in enumerate(audio_files, start=1):
        base_name = os.path.splitext(filename)[0]
        output_path = os.path.join(TRANSCRIPT_DIR, f"{base_name}.json")
        audio_path = os.path.join(AUDIO_DIR, filename)

        if os.path.exists(output_path):
            print(f"[{idx}/{len(audio_files)}] Skipping {filename} (already exists)")
            continue

        print(f"[{idx}/{len(audio_files)}] Transcribing {filename}...")

        # Forcing language="hi" prevents Whisper from drifting into Urdu script
        result = model.transcribe(
            audio_path,
            language="hi",
            task="transcribe",
            verbose=False
        )

        segments = []
        for seg in result.get("segments", []):
            text = seg["text"].strip()
            if text:
                segments.append({
                    "start": round(seg["start"], 2),
                    "end": round(seg["end"], 2),
                    "text": text
                })

        payload = {
            "lecture_name": base_name,
            "source_file": filename,
            "full_text": result["text"].strip(),
            "segments": segments
        }

        with open(output_path, "w", encoding="utf-8") as f:
            json.dump(payload, f, ensure_ascii=False, indent=2)

        print(f"Saved -> {output_path} ({len(segments)} segments)")

    print("\nTranscription finished.")


if __name__ == "__main__":
    transcribe_lectures()