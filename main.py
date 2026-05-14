import requests
import sys
import os

URL = "https://stuff.mit.edu/afs/sipb/contrib/pi/pi-billion.txt"
CACHE_FILE = "pi.txt"
CHUNK_SIZE = 1024 * 256  # 256KB per chunk
BAR_LEN = 30


def download_pi() -> str:
    """Download pi.txt with a progress bar and return the full text."""
    print("📡 Connecting to file...")

    # Disable automatic decompression so Content-Length matches raw bytes received
    with requests.get(URL, stream=True, headers={"Accept-Encoding": "identity"}) as response:
        response.raise_for_status()

        total_size = int(response.headers.get("Content-Length", 0))
        print(f"📦 File size: {total_size / (1024 * 1024):.1f} MB\n")

        downloaded = 0
        chunks = []

        for chunk in response.iter_content(chunk_size=CHUNK_SIZE):
            chunks.append(chunk)
            downloaded += len(chunk)

            if total_size:
                percent = min(downloaded / total_size * 100, 100.0)
                filled = min(int(BAR_LEN * downloaded / total_size), BAR_LEN)
                bar = "█" * filled + "░" * (BAR_LEN - filled)
                print(f"\r[{bar}] {percent:5.1f}%  |  Downloaded: {downloaded / (1024*1024):.1f} MB", end="", flush=True)
            else:
                print(f"\r⬇️  Downloaded: {downloaded / (1024 * 1024):.1f} MB", end="", flush=True)

    print(f"\n✅ Download complete!\n")
    text = b"".join(chunks).decode("utf-8", errors="ignore")

    # Save to cache
    with open(CACHE_FILE, "w", encoding="utf-8") as f:
        f.write(text)

    return text


def search_in_text(text: str, query: str) -> int:
    """Search for query in text with a progress bar. Returns count of occurrences."""
    print(f"\n🔍 Searching for: '{query}'")

    total = len(text)
    found = 0
    overlap = len(query) - 1
    pos = 0

    while pos < total:
        end = min(pos + CHUNK_SIZE, total)
        # Include overlap from previous chunk to catch boundary matches
        start = max(pos - overlap, 0)
        slice_text = text[start:end]

        found += slice_text.count(query)

        percent = min(end / total * 100, 100.0)
        filled = min(int(BAR_LEN * end / total), BAR_LEN)
        bar = "█" * filled + "░" * (BAR_LEN - filled)
        print(f"\r[{bar}] {percent:5.1f}%  |  Found so far: {found:,}", end="", flush=True)

        pos = end

    print(f"\n\n✅ Done! Found {found:,} occurrence(s) of '{query}' in the file.\n")
    return found


def main():
    # Load or download the file
    if os.path.exists(CACHE_FILE):
        print(f"📂 Loading cached file: {CACHE_FILE}")
        with open(CACHE_FILE, "r", encoding="utf-8") as f:
            pi_text = f.read()
        print(f"✅ Loaded {len(pi_text) / (1024*1024):.1f} MB from cache.\n")
    else:
        pi_text = download_pi()

    # Search loop
    while True:
        query = input("🔢 Enter the sequence to search for (or 'q' to quit): ").strip()

        if query.lower() == "q":
            print("👋 Goodbye!")
            break

        if not query:
            print("❌ No sequence entered. Try again.\n")
            continue

        search_in_text(pi_text, query)


if __name__ == "__main__":
    main()