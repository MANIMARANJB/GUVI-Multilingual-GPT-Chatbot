"""
GUVI Multilingual GPT Chatbot

Advanced Web Scraper

Collects:
- Homepage information
- Course information
- Blog article content

Output:
data/raw/guvi_raw.json
"""


import json
import os
import re

import requests
from bs4 import BeautifulSoup



# --------------------------------------
# Configuration
# --------------------------------------

OUTPUT_FILE = "data/raw/guvi_raw.json"


HEADERS = {
    "User-Agent":
        "Mozilla/5.0"
}


BASE_URL = "https://www.guvi.in"


STATIC_PAGES = [
    {
        "url": "https://www.guvi.in/",
        "category": "about"
    },
    {
        "url": "https://www.guvi.in/courses/",
        "category": "courses"
    }
]


BLOG_PAGE = "https://www.guvi.in/blog/"

MAX_BLOGS = 30



# --------------------------------------
# Fetch webpage
# --------------------------------------

def fetch_page(url):

    try:

        response = requests.get(
            url,
            headers=HEADERS,
            timeout=15
        )

        response.raise_for_status()

        return response.text


    except Exception as error:

        print(
            f"Failed: {url}",
            error
        )

        return None



# --------------------------------------
# Clean text
# --------------------------------------

def clean_text(text):

    text = re.sub(
        r"\s+",
        " ",
        text
    )

    return text.strip()



# --------------------------------------
# Extract article links
# --------------------------------------

def extract_blog_links(html):

    soup = BeautifulSoup(
        html,
        "html.parser"
    )

    links = []


    blocked_paths = [
        "/category/",
        "/author/",
        "/tag/",
        "/page/"
    ]


    for a in soup.find_all("a", href=True):

        href = a["href"]


        if "/blog/" not in href:
            continue


        if href.startswith("/"):

            href = BASE_URL + href


        # remove category/author pages

        if any(
            path in href
            for path in blocked_paths
        ):
            continue


        # remove main blog page

        if href.rstrip("/") == BLOG_PAGE.rstrip("/"):
            continue


        if href not in links:

            links.append(
                href
            )


    return links[:MAX_BLOGS]

# --------------------------------------
# Extract page content
# --------------------------------------

def extract_content(html):

    soup = BeautifulSoup(
        html,
        "html.parser"
    )


    # remove noise

    for tag in soup(
        [
            "script",
            "style",
            "nav",
            "footer",
            "header"
        ]
    ):

        tag.decompose()



    title = "GUVI"


    if soup.title:

        title = soup.title.text.strip()



    paragraphs = []


    for p in soup.find_all(
        "p"
    ):

        text = p.get_text(
            strip=True
        )


        if len(text) > 40:

            paragraphs.append(
                text
            )



    content = " ".join(
        paragraphs
    )


    return {
        "title": title,
        "content": clean_text(content)
    }



# --------------------------------------
# Scrape static pages
# --------------------------------------

def scrape_static_pages():

    data = []


    for page in STATIC_PAGES:

        print(
            "Scraping:",
            page["url"]
        )


        html = fetch_page(
            page["url"]
        )


        if html:

            result = extract_content(
                html
            )


            result["category"] = page["category"]

            result["source"] = page["url"]


            data.append(
                result
            )


    return data



# --------------------------------------
# Scrape blogs
# --------------------------------------

def scrape_blogs():

    data = []


    print(
        "Finding blog articles..."
    )


    html = fetch_page(
        BLOG_PAGE
    )


    if not html:

        return data



    blog_links = extract_blog_links(
        html
    )


    print(
        "Blogs found:",
        len(blog_links)
    )



    for url in blog_links:


        print(
            "Scraping blog:",
            url
        )


        html = fetch_page(
            url
        )


        if html:


            result = extract_content(
                html
            )


            result["category"] = "blog"

            result["source"] = url


            if len(result["content"]) > 100:

                data.append(
                    result
                )


    return data



# --------------------------------------
# Save dataset
# --------------------------------------

def save_data(data):

    os.makedirs(
        "data/raw",
        exist_ok=True
    )


    with open(
        OUTPUT_FILE,
        "w",
        encoding="utf-8"
    ) as file:


        json.dump(
            data,
            file,
            indent=4,
            ensure_ascii=False
        )



# --------------------------------------
# Main
# --------------------------------------

def main():

    dataset = []


    dataset.extend(
        scrape_static_pages()
    )


    dataset.extend(
        scrape_blogs()
    )


    save_data(
        dataset
    )


    print(
        "\nScraping completed"
    )


    print(
        "Total documents:",
        len(dataset)
    )



if __name__ == "__main__":

    main()