import requests
from bs4 import BeautifulSoup
import os
import time

# --- Constants ---
BASE_URL = "https://www.pexels.com/videos/"
SEARCH_QUERY = "people-walking"
DOWNLOAD_DIR = "downloaded_videos"
MAX_VIDEOS = 10
HEADERS = {
    'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36'
}

# --- Functions ---

def setup_directory(dir_name):
    """Create the download directory if it doesn't exist."""
    if not os.path.exists(dir_name):
        os.makedirs(dir_name)
        print(f"Created directory: {dir_name}")

def download_video(video_url, file_path):
    """Downloads a single video from a URL to a given path."""
    try:
        r = requests.get(video_url, stream=True, headers=HEADERS, timeout=30)
        r.raise_for_status()
        with open(file_path, 'wb') as f:
            for chunk in r.iter_content(chunk_size=8192):
                f.write(chunk)
        print(f"Successfully downloaded {file_path}")
        return True
    except requests.exceptions.RequestException as e:
        print(f"Error downloading {video_url}: {e}")
        return False

def scrape_pexels():
    """Main function to scrape video links and download them."""
    setup_directory(DOWNLOAD_DIR)
    
    # In a real scenario, we would need to handle pagination
    search_url = f"{BASE_URL}search/{SEARCH_QUERY}/"
    
    print(f"Fetching video list from: {search_url}")
    response = requests.get(search_url, headers=HEADERS)
    
    if response.status_code != 200:
        print(f"Failed to fetch page. Status code: {response.status_code}")
        return

    soup = BeautifulSoup(response.content, 'html.parser')
    
    # This selector is a GUESS and will likely need to be updated.
    # Pexels uses dynamic class names. A more robust solution might need Selenium.
    video_elements = soup.select('a[href*="/video/"]')
    
    downloaded_count = 0
    for video_el in video_elements:
        if downloaded_count >= MAX_VIDEOS:
            break
            
        # This is a simplified extraction logic.
        # Pexels video pages have complex structures to find the actual .mp4 file.
        # This initial version just grabs the page link. A second request would be needed.
        video_page_url = "https://www.pexels.com" + video_el['href']
        print(f"Found video page: {video_page_url}")

        # TODO: Add logic to visit video_page_url and find the actual download link.
        # This is a placeholder for the next development step.
        
        time.sleep(1) # Be respectful to the server

    print(f"\nScraping finished. Total videos to process: {len(video_elements)}")
    print("NOTE: This is a placeholder script. It finds video pages but doesn't download the actual MP4s yet.")


# --- Main Execution ---

if __name__ == "__main__":
    scrape_pexels()
