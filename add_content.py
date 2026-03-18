import os
import json
import requests
import chromadb
from pypdf import PdfReader
from sentence_transformers import SentenceTransformer
import time

print("Loading models...")
embedding_model = SentenceTransformer("all-MiniLM-L6-v2")
client = chromadb.PersistentClient(path="data/embeddings")
collection = client.get_or_create_collection(name="raguru_lectures")

os.makedirs("data/content", exist_ok=True)
os.makedirs("data/pdfs", exist_ok=True)
os.makedirs("audio", exist_ok=True)

print(f"Current chunks in knowledge base: {collection.count()}")


def add_youtube_videos(video_list):
    print("\n" + "=" * 50)
    print("Adding YouTube Videos...")
    print("=" * 50)

    import whisper
    whisper_model = whisper.load_model("base")

    for video in video_list:
        audio_path = f"audio/{video['name']}.mp3"
        save_path = f"data/content/video_{video['name']}.json"

        if os.path.exists(save_path):
            print(f"Already done: {video['name']} — skipping!")
            continue

        print(f"\nDownloading: {video['name']}")
        os.system(f'yt-dlp -x --audio-format mp3 -o "{audio_path}" "{video["url"]}"')

        if not os.path.exists(audio_path):
            print(f"Download failed: {video['name']}")
            continue

        print(f"Transcribing: {video['name']}")
        result = whisper_model.transcribe(audio_path)

        with open(save_path, 'w', encoding='utf-8') as f:
            json.dump({
                "title": video['name'],
                "type": "youtube",
                "text": result['text'],
                "segments": [
                    {
                        "start": seg['start'],
                        "end": seg['end'],
                        "text": seg['text']
                    }
                    for seg in result['segments']
                ]
            }, f, ensure_ascii=False, indent=2)

        print(f"Done: {video['name']}")

    print("\nAll videos processed!")


def add_all_pdfs():
    print("\n" + "=" * 50)
    print("Adding PDFs...")
    print("=" * 50)

    pdf_folder = "data/pdfs"
    pdfs = [f for f in os.listdir(pdf_folder) if f.endswith('.pdf')]

    if not pdfs:
        print("No PDFs found in data/pdfs/ folder!")
        return

    print(f"Found {len(pdfs)} PDFs")

    for pdf_file in pdfs:
        pdf_path = f"{pdf_folder}/{pdf_file}"
        save_path = f"data/content/pdf_{pdf_file}.json"

        if os.path.exists(save_path):
            print(f"Already done: {pdf_file} — skipping!")
            continue

        print(f"\nReading PDF: {pdf_path}")

        try:
            reader = PdfReader(pdf_path)
            full_text = ""

            for page_num, page in enumerate(reader.pages):
                text = page.extract_text()
                if text:
                    full_text += f"\nPage {page_num+1}:\n{text}"

            with open(save_path, 'w', encoding='utf-8') as f:
                json.dump({
                    "title": pdf_file.replace('.pdf', ''),
                    "type": "pdf",
                    "text": full_text
                }, f, ensure_ascii=False, indent=2)

            print(f"PDF added: {pdf_file} — {len(full_text)} characters")

        except Exception as e:
            print(f"Error reading {pdf_file}: {e}")

    print("\nAll PDFs processed!")


def add_wikipedia_topics(topics):
    print("\n" + "=" * 50)
    print("Adding Wikipedia Articles...")
    print("=" * 50)

    for topic in topics:
        save_path = f"data/content/wiki_{topic.replace(' ', '_')}.json"

        if os.path.exists(save_path):
            print(f"Already done: {topic} — skipping!")
            continue

        try:
            url = f"https://en.wikipedia.org/w/api.php?action=query&titles={topic.replace(' ', '+')}&prop=extracts&exintro&explaintext&format=json"
            headers = {"User-Agent": "RAGuru/1.0"}
            response = requests.get(url, headers=headers, timeout=15)
            data = response.json()

            pages = data['query']['pages']
            page = list(pages.values())[0]
            text = page.get('extract', '')
            title = page.get('title', topic)

            if text:
                with open(save_path, 'w', encoding='utf-8') as f:
                    json.dump({
                        "title": f"Wikipedia: {title}",
                        "type": "wikipedia",
                        "text": text
                    }, f, ensure_ascii=False, indent=2)
                print(f"Added: {title} — {len(text)} characters")
            else:
                print(f"Empty article: {topic}")

        except Exception as e:
            print(f"Error: {topic} — {e}")

        time.sleep(1)

    print("\nAll Wikipedia articles processed!")
def embed_all_content():
    print("\n" + "=" * 50)
    print("Creating Embeddings...")
    print("=" * 50)

    content_folder = "data/content"
    files = [f for f in os.listdir(content_folder) if f.endswith('.json')]

    print(f"Total content files: {len(files)}")

    all_chunks = []
    all_ids = []
    all_metadata = []

    for file in files:
        try:
            with open(f"{content_folder}/{file}", 'r', encoding='utf-8') as f:
                data = json.load(f)

            text = data.get('text', '')
            title = data.get('title', file)
            content_type = data.get('type', 'unknown')

            chunk_size = 500
            for i in range(0, len(text), chunk_size):
                chunk = text[i:i+chunk_size]
                if len(chunk) < 50:
                    continue

                chunk_id = f"{file}_{i}"

                try:
                    existing = collection.get(ids=[chunk_id])
                    if existing['ids']:
                        continue
                except:
                    pass

                all_chunks.append(chunk)
                all_ids.append(chunk_id)
                all_metadata.append({
                    "title": title,
                    "type": content_type,
                    "source": file,
                    "lecture": title,
                    "start_time": 0,
                    "end_time": 0
                })

        except Exception as e:
            print(f"Error processing {file}: {e}")

    if not all_chunks:
        print("No new content to embed — everything already done!")
        return

    print(f"New chunks to embed: {len(all_chunks)}")

    batch_size = 100
    for i in range(0, len(all_chunks), batch_size):
        batch = all_chunks[i:i+batch_size]
        batch_ids = all_ids[i:i+batch_size]
        batch_meta = all_metadata[i:i+batch_size]

        embeddings = embedding_model.encode(batch).tolist()
        collection.add(
            documents=batch,
            embeddings=embeddings,
            ids=batch_ids,
            metadatas=batch_meta
        )

        print(f"Progress: {min(i+batch_size, len(all_chunks))}/{len(all_chunks)}")

    print(f"\nTotal chunks now: {collection.count()}")
    print("Knowledge base updated successfully!")


if __name__ == "__main__":

    print("=" * 50)
    print("RAGuru — Content Manager")
    print("=" * 50)

    new_videos = [,
    ]
    if new_videos:
        add_youtube_videos(new_videos)
    add_all_pdfs()
    wiki_topics = [
        "Python (programming language)",  # Yeh change karo
        "Object-oriented programming",
        "Data structure",
        "Algorithm",
        "Machine learning",
        "Artificial intelligence",
        "Computer programming",
        "Software development",
    ]
    add_wikipedia_topics(wiki_topics)


    embed_all_content()

    print("\n" + "=" * 50)
    print("All done! RAGuru is now smarter!")
    print("Run: python -m streamlit run app.py")
    print("=" * 50)