# web_scraper_agent.py
import requests
from bs4 import BeautifulSoup

def scrape_web_stoicism(output_file="web_stoic.txt", urls=None):
    if urls is None:
        urls = [
            "https://modernstoicism.com/blog",
            "https://dailystoic.com/stoicism-101/"
        ]
    web_texts = []
    for url in urls:
        try:
            response = requests.get(url, timeout=10)
            soup = BeautifulSoup(response.content, "html.parser")
            paragraphs = [p.get_text().strip() for p in soup.find_all("p") if len(p.get_text()) > 50]
            web_texts.extend(paragraphs)
        except Exception as e:
            print(f"Failed to scrape {url}: {e}")
    with open(output_file, "w", encoding="utf-8") as f:
        f.write("\n".join(web_texts))
    return web_texts

if __name__ == "__main__":
    texts = scrape_web_stoicism()
    print(f"Scraped {len(texts)} snippets!")
