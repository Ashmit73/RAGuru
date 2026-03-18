import whisper
import json
import os


os.makedirs("data/transcripts", exist_ok=True)


print("Loading Whisper model...")
model = whisper.load_model("base")
print("Whisper model ready!")


audio_files = [f for f in os.listdir("audio") if f.endswith(".mp3")]
print(f"\nTotal audio files found: {len(audio_files)}")
print("=" * 50)


for index, audio_file in enumerate(audio_files):
    audio_path = f"audio/{audio_file}"
    transcript_name = audio_file.replace(".mp3", "")
    transcript_path = f"data/transcripts/{transcript_name}.json"


    if os.path.exists(transcript_path):
        print(f"Already transcribed! Skipping: {audio_file}")
        continue

    print(f"\nTranscribing {index + 1}/{len(audio_files)}: {audio_file}")


    result = model.transcribe(
        audio_path,
        verbose=False,
        word_timestamps=True
    )


    transcript_data = {
        "file": audio_file,
        "text": result["text"],
        "segments": [
            {
                "start": round(seg["start"], 2),
                "end": round(seg["end"], 2),
                "text": seg["text"].strip()
            }
            for seg in result["segments"]
        ]
    }

    with open(transcript_path, "w", encoding="utf-8") as f:
        json.dump(transcript_data, f, ensure_ascii=False, indent=2)

    print(f"Transcribed successfully: {transcript_name}.json")
    print(f"Preview: {result['text'][:150]}...")

print("\n" + "=" * 50)
print("All transcriptions completed!")
print("Check data/transcripts/ folder!")
print("=" * 50)