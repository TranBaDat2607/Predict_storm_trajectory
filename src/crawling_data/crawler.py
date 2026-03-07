import requests
from bs4 import BeautifulSoup
import pandas as pd
import os
import threading
from concurrent.futures import ThreadPoolExecutor, as_completed
from requests.adapters import HTTPAdapter
from urllib3.util.retry import Retry

_local = threading.local()

def get_session() -> requests.Session:
    if not hasattr(_local, 'session'):
        session = requests.Session()
        retry = Retry(total=5, backoff_factor=1, status_forcelist=[500, 502, 503, 504])
        session.mount('https://', HTTPAdapter(max_retries=retry))
        _local.session = session
    return _local.session

def take_hrefs(url: str):
    print(f"[1/4] Fetching storm list from: {url}")
    page = get_session().get(url)
    soup = BeautifulSoup(page.content, 'lxml')
    table = soup.find_all('table')[-1]
    hrefs = []

    for a in table.find_all('a', href=True):
        hrefs.append(a['href'])

    print(f"[1/4] Found {len(hrefs)} storms to crawl.")
    return hrefs

def fetch_one(href: str, base_url: str, index: int, total: int) -> pd.DataFrame:
    url = base_url + href
    print(f"[2/4] ({index}/{total}) Fetching: {href}")
    page = get_session().get(url)
    soup = BeautifulSoup(page.content, 'lxml')
    table = soup.find_all('table')[-1]

    header = [th.get_text(strip=True) for th in table.find_all('th')]

    data = []
    for tr in table.find_all('tr')[1:]:
        row = [td.get_text(strip=True) for td in tr.find_all('td')]
        data.append(row)

    return pd.DataFrame(data, columns=header)

def crawl_data(hrefs: list, max_workers: int = 10):
    base_url = 'https://ncics.org/ibtracs/'
    csv_file = os.path.join(os.path.dirname(__file__), '..', '..', 'data', 'raw', 'storm_data.csv')
    total = len(hrefs)

    frames = [None] * total

    print(f"[2/4] Starting parallel crawl with {max_workers} workers...")
    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        future_to_index = {
            executor.submit(fetch_one, href, base_url, i + 1, total): i
            for i, href in enumerate(hrefs)
        }
        done = 0
        for future in as_completed(future_to_index):
            i = future_to_index[future]
            frames[i] = future.result()
            done += 1
            print(f"[2/4] Progress: {done}/{total} storms fetched.")

    print("[3/4] Merging all data...")
    df_new = pd.concat(frames, ignore_index=True, sort=False)
    print(f"[3/4] Merged {len(df_new)} rows total.")

    print("[4/4] Saving to CSV...")
    if os.path.exists(csv_file):
        df_existing = pd.read_csv(csv_file)
        df_merged = pd.concat([df_existing, df_new], ignore_index=True, sort=False)
        df_merged.to_csv(csv_file, index=False)
        print(f"[4/4] Appended to existing '{csv_file}'. Total rows: {len(df_merged)}.")
    else:
        df_new.to_csv(csv_file, index=False)
        print(f"[4/4] Created new '{csv_file}' with {len(df_new)} rows.")

    print("Done! Data has been saved successfully.")


url = 'https://ncics.org/ibtracs/index.php?name=ATCF-WP'
crawl_data(take_hrefs(url))
