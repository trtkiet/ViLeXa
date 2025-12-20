import os
import requests
from bs4 import BeautifulSoup
from urllib.parse import urljoin
import time
from config import BASE_URL, START_URL, DOWNLOAD_FOLDER, ID_LISTS


# Create download folder if it doesn't exist
if not os.path.exists(DOWNLOAD_FOLDER):
    os.makedirs(DOWNLOAD_FOLDER)
    
def check_expired(soup):
    doc_info = soup.find("div", class_="vbInfo")
    # Check if "Còn hiệu lực" is in the validity_div text
    if doc_info and ("Còn hiệu lực" in doc_info.text or "Hết hiệu lực một phần" in doc_info.text):
        return False
    return True

def crawl_document(url, output_folder):
    soup = BeautifulSoup(requests.get(url).content, "html.parser")
    
    if soup.find("div", class_="toanvancontent") is None:
        print(f"No content found for URL: {url}")
        return 
    
    if check_expired(soup):
        print(f"Document at {url} is expired. Skipping download.")
        return
    
    Id = url.split("ID=")[-1]
    
    filename = f"{Id}.json"
    filepath = os.path.join(output_folder, filename)
    if os.path.exists(filepath):
        print(f"Document {filename} already exists. Skipping download.")
        return
    
    content_div = soup.find("div", class_="toanvancontent")
    # From paragraphs inside content_div, extract text
    paragraphs = content_div.find_all("p")
    content = "\n".join(p.text.strip() for p in paragraphs)

    json_data = {
        "Id": Id,
        "Content": content
    }
    with open(filepath, "w", encoding="utf-8") as f:
        import json
        json.dump(json_data, f, ensure_ascii=False, indent=4)
    
    

def get_page_number(soup):
    """
    Get the last page number from the pagination section of the page.
    """
    paging = soup.find("div", class_="paging")
    
    if not paging:
        # print("No pagination found.")
        return 1  # Assume only one page if no pagination is found
    
    last_page_link = paging.find_all("a")[-1]['href']
    last_page_number = int(last_page_link.split("Page=")[-1])
    # print(f"Last page number: {last_page_number}")
    return last_page_number

def crawl_vbpl(url):
    soup = BeautifulSoup(requests.get(url).content, "html.parser")
    max_page = get_page_number(soup)
    law_type = soup.find("a", class_="selected").text.strip()
    print(f"Crawling documents of type: {law_type}")
    law_type_str = law_type.split(":")[1].strip().split(".")[0].strip().replace(" ", "_")
    print(f"Saving documents to folder: {law_type_str}")
    
    if not os.path.exists(os.path.join(DOWNLOAD_FOLDER, law_type_str)):
        os.makedirs(os.path.join(DOWNLOAD_FOLDER, law_type_str))
    
    current_page = 1
    
    while current_page <= max_page:
        print(f"Crawling page {current_page} of {max_page}")
        page_url = f"{url}&Page={current_page}"
        page_soup = BeautifulSoup(requests.get(page_url).content, "html.parser")
        
        listLaw = page_soup.find("ul", class_="listLaw")
        doc_titles = listLaw.find_all("p", class_="title")
        document_links = []
        for doc_title in doc_titles:
            document_links.extend(doc_title.find_all("a"))
        
        print()
        
        # document_links = page_soup.find_all("a", class_="document-link")
        
        for link in document_links:
            print(link["href"])
            doc_url = urljoin(BASE_URL, link['href'])
            print(f"Downloading document from {doc_url}")
            crawl_document(doc_url, os.path.join(DOWNLOAD_FOLDER, law_type_str))
        
        current_page += 1
        time.sleep(1)  # Be polite and avoid overwhelming the server

def main():
    for id_loai_van_ban in ID_LISTS:
        url = START_URL.replace("idInput", str(id_loai_van_ban))
        print(f"Crawling documents for idLoaiVanBan={id_loai_van_ban} from {url}")
        crawl_vbpl(url)
        time.sleep(2)  # Be polite and avoid overwhelming the server

if __name__ == "__main__":
    main()
    
    # get_page_number(BeautifulSoup(requests.get("https://vbpl.vn/TW/Pages/vanban.aspx?idLoaiVanBan=15&dvid=13").content, "html.parser"))