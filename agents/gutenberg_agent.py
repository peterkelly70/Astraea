# gutenberg_agent.py
import requests
from bs4 import BeautifulSoup
import os

def fetch_gutenberg_books(output_dir="texts"):
    books = {
        "Meditations": "http://gutenberg.net.au/ebooks03/0300081.txt",
        "Discourses": "http://gutenberg.net.au/ebooks14/1400721.txt",  # Verify pre-1957 trans.
        "Enchiridion": "http://gutenberg.net.au/ebooks04/0400141.txt"
    }
    texts = {}
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    for title, url in books.items():
        file_path = os.path.join(output_dir, f"{title.lower().replace(' ', '_')}.txt")
        if not os.path.exists(file_path):
            response = requests.get(url)
            soup = BeautifulSoup(response.content, "html.parser")
            text = soup.get_text(separator="\n").strip()
            with open(file_path, "w", encoding="utf-8") as f:
                f.write(text)
        with open(file_path, "r", encoding="utf-8") as f:
            texts[title] = f.read().splitlines()
    return texts

if __name__ == "__main__":
    texts = fetch_gutenberg_books()
    print(f"Fetched {len(texts)} books: {list(texts.keys())}")
