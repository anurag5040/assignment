

import os
import time
from urllib.parse import quote
from typing import List

from playwright.sync_api import sync_playwright
from html2docx import html2docx
from docx import Document
from bs4 import BeautifulSoup


def get_top_5_links(query: str) -> List[str]:
    query_encoded = quote(query)
    search_url = f"https://duckduckgo.com/?q={query_encoded}&t=h_&ia=web"
    hrefs = []

    with sync_playwright() as p:
        browser = p.chromium.launch(headless=False)     
        page = browser.new_page()

        try:
            page.goto(search_url, timeout=20000)
            page.wait_for_timeout(3000)

            links = page.locator("a[data-testid='result-title-a']")

            for i in range(min(10, links.count())):
                try:
                    href = links.nth(i).get_attribute("href")
                    if href and href.startswith("http"):
                        hrefs.append(href)
                    if len(hrefs) >= 5:
                        break
                except Exception as e:
                    print(f"⚠️ Skipping index {i}: {e}")
                    continue

            print(f"✅ Found links:\n" + "\n".join(hrefs))

        except Exception as e:
            print(f"❌ Failed to fetch links: {e}")
        finally:
            browser.close()

    return hrefs

import os
import time
import requests
from bs4 import BeautifulSoup
from docx import Document
from data import *

def save_links_to_docx(links, folder_name=None):
    if folder_name is None:
        folder_name = f"output_docs"
    os.makedirs(folder_name, exist_ok=True)

    headers = {
        "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64)"
    }

    for idx, link in enumerate(links):
        try:
            response = requests.get(link, headers=headers, timeout=10)
            response.raise_for_status()

            soup = BeautifulSoup(response.text, "html.parser")
            text = soup.get_text(separator="\n", strip=True)
            process_and_embed_text(text)

        except Exception as e:
            print(f"⚠️ Failed to process {link}: {e}")
