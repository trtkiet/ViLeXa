import os
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from urllib.parse import urljoin

import requests
from bs4 import BeautifulSoup
from requests.adapters import HTTPAdapter
from urllib3.util.retry import Retry

from config import BASE_URL, START_URL, DOWNLOAD_FOLDER, ID_LISTS

MAX_WORKERS = int(os.environ.get("CRAWLER_MAX_WORKERS", "10"))
REQUEST_TIMEOUT = float(os.environ.get("CRAWLER_REQUEST_TIMEOUT", "30"))


# Create download folder if it doesn't exist
if not os.path.exists(DOWNLOAD_FOLDER):
    os.makedirs(DOWNLOAD_FOLDER)


def build_session():
    retry_strategy = Retry(
        total=3,
        backoff_factor=0.5,
        status_forcelist=[429, 500, 502, 503, 504],
        allowed_methods=["HEAD", "GET", "OPTIONS"],
    )
    adapter = HTTPAdapter(
        pool_connections=MAX_WORKERS * 2,
        pool_maxsize=MAX_WORKERS * 2,
        max_retries=retry_strategy,
    )
    session = requests.Session()
    session.mount("http://", adapter)
    session.mount("https://", adapter)
    session.headers.update({"User-Agent": "vbpl-crawler/1.0"})
    return session

def crawl_document(session, url, doc_name, expired, output_folder):
    Id = url.split("ID=")[-1]

    filename = f"{Id}_{doc_name}.json"
    filepath = os.path.join(output_folder, filename)
    if os.path.exists(filepath):
        # print(f"Document {filename} already exists. Skipping download.")
        return

    if expired:
        print(f"Document at {url} is expired. Skipping download.")
        return

    try:
        response = session.get(url, timeout=REQUEST_TIMEOUT)
        response.raise_for_status()
    except requests.RequestException as exc:
        print(f"Failed to fetch {url}: {exc}")
        return
    
    soup = BeautifulSoup(response.content, "html.parser")

    if soup.find("div", class_="toanvancontent") is None:
        print(f"No content found for URL: {url}")
        return

    content_div = soup.find("div", class_="toanvancontent")
    paragraphs = content_div.find_all("p")
    content = "\n".join(p.text.strip() for p in paragraphs)

    json_data = {"Id": Id, "Content": content}
    with open(filepath, "w", encoding="utf-8") as f:
        import json

        json.dump(json_data, f, ensure_ascii=False, indent=4)
    # print(f"Downloaded: {filename}")

def get_page_number(soup):
    """
    Get the last page number from the pagination section of the page.
    """
    paging = soup.find("div", class_="paging")
    
    if not paging:
        return 1  # Assume only one page if no pagination is found
    
    last_page_link = paging.find_all("a")[-1]['href']
    last_page_number = int(last_page_link.split("Page=")[-1])
    return last_page_number

def process_page(session, page_url, output_folder, doc_executor):
    try:
        page_response = session.get(page_url, timeout=REQUEST_TIMEOUT)
        page_response.raise_for_status()
    except requests.RequestException as exc:
        print(f"Failed to fetch {page_url}: {exc}")
        return

    page_soup = BeautifulSoup(page_response.content, "html.parser")
    
    listLaw = page_soup.find("ul", class_="listLaw")
    if not listLaw:
        print(f"No document list found on {page_url}")
        return
    
    doc_titles = listLaw.find_all("p", class_="title")
    document_links = []
    for doc_title in doc_titles:
        document_links.extend(doc_title.find_all("a"))
    
    right_sides =  listLaw.find_all("div", class_="right")
    labels = [(div.find("p", class_="red") is not None and div.find("p", class_="red").text == "Trạng thái:Hết hiệu lực toàn bộ") for div in right_sides]
    doc_names = [link.text.strip().replace("/", "_") for link in document_links if link.get('href')]
    doc_urls = [urljoin(BASE_URL, link['href']) for link in document_links if link.get('href')]
    
    for doc_url, doc_name, expired in zip(doc_urls, doc_names, labels):
        doc_executor.submit(
            crawl_document, session, doc_url, doc_name, expired, output_folder
        )

def crawl_vbpl(session, url):
    try:
        response = session.get(url, timeout=REQUEST_TIMEOUT)
        response.raise_for_status()
    except requests.RequestException as exc:
        print(f"Failed to fetch {url}: {exc}")
        return

    soup = BeautifulSoup(response.content, "html.parser")
    max_page = get_page_number(soup)
    law_type = soup.find("a", class_="selected").text.strip()
    print(f"Crawling documents of type: {law_type} (Total pages: {max_page})")
    law_type_str = law_type.split(":")[1].strip().split(".")[0].strip().replace(" ", "_")
    print(f"Saving documents to folder: {law_type_str}")
    
    output_folder = os.path.join(DOWNLOAD_FOLDER, law_type_str)
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)
    
    # Executor for processing pages (fetching list of docs)
    page_executor = ThreadPoolExecutor(max_workers=5)
    
    # Executor for downloading documents
    doc_executor = ThreadPoolExecutor(max_workers=MAX_WORKERS)
    
    page_futures = []
    print(f"Processing {max_page} pages in parallel...")
    for current_page in range(1, max_page + 1):
        page_url = f"{url}&Page={current_page}"
        page_futures.append(
            page_executor.submit(process_page, session, page_url, output_folder, doc_executor)
        )
    
    # Wait for all pages to be processed
    for future in as_completed(page_futures):
        future.result()
        
    page_executor.shutdown()
    
    # Wait for all documents to be downloaded
    print("Waiting for document downloads to complete...")
    doc_executor.shutdown(wait=True)
    print(f"Finished crawling {law_type}")

def main():
    session = build_session()
    for id_loai_van_ban in ID_LISTS:
        url = START_URL.replace("idInput", str(id_loai_van_ban))
        print(f"Crawling documents for idLoaiVanBan={id_loai_van_ban} from {url}")
        crawl_vbpl(session, url)
        time.sleep(2)  # Be polite and avoid overwhelming the server

if __name__ == "__main__":
    main()
