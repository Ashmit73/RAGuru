import os
video_list = [
    {
        "url": "https://youtu.be/7wnove7K-ZQ?si=Onw2pX1FrlmWaqzG",
        "name": "lecture1"
    },
    {
        "url": "https://youtu.be/Tto8TS-fJQU?si=SyFRz4815bRPT_5j",
        "name": "lecture2"
    },
    {
        "url": "https://youtu.be/xwKO_y2gHxQ?si=g_BgmEyaupfmKLPk",
        "name": "lecture3"
    },
    {
        "url": "https://youtu.be/7IWOYhfAcVg?si=L6PgEC3KEykNjT6n",
        "name": "lecture4"
    },
    {
        "url": "https://youtu.be/qxPMmW93eDs?si=rMRwxs6hPNbfsKok",
        "name": "lecture5"
    },

]

# Create audio folder if not exists
os.makedirs("audio", exist_ok=True)

print("=" * 50)
print("RAGuru — Video Download Started!")
print(f"Total videos to download: {len(video_list)}")
print("=" * 50)

# Download all videos one by one
for index, video in enumerate(video_list):
    print(f"\nDownloading {index + 1}/{len(video_list)}: {video['name']}")

    output_path = f"audio/{video['name']}.mp3"

    # Skip if already downloaded
    if os.path.exists(output_path):
        print(f"Already exists! Skipping: {video['name']}")
        continue

    # Download using yt-dlp
    command = f'yt-dlp -x --audio-format mp3 -o "{output_path}" "{video["url"]}"'
    os.system(command)

    print(f"Downloaded: {video['name']}.mp3")

print("\n" + "=" * 50)
print("All videos downloaded successfully!")
print("Check your audio/ folder!")
print("=" * 50)
