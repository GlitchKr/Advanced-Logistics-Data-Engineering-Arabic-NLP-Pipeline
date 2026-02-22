# -*- coding: utf-8 -*-
"""
New_code_v3.py  —  Data Pipeline v6.0 (Template)
================================================
Changelog from v2:
  - AdvancedLocationCleanerV6 (cleaner_v6.py) replaces inline V5 class
  - External locations.json loaded once (cached)
  - RapidFuzz fuzzy matching fallback (install: pip install rapidfuzz)
  - Enhanced Arabic normalisation (ligatures, Tatweel, EN stopwords)
  - FastAPI microservice available in api.py
  - Full mypy-strict type hints in cleaner_v6.py
"""
import requests
import pandas as pd
import numpy as np
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime
from requests.adapters import HTTPAdapter
from urllib3.util.retry import Retry
import time
import os
import re
import pickle
import json
from pathlib import Path
from itertools import combinations
from dataclasses import dataclass
from functools import lru_cache
import logging
from logging.handlers import RotatingFileHandler
try:
    from dotenv import load_dotenv
    load_dotenv()
except ImportError:
    pass  # install with: pip install python-dotenv

# ================= Configuration (Dynamic Paths) =================
BASE_DIR = Path(__file__).parent
DATA_DIR = BASE_DIR / "data"
os.makedirs(DATA_DIR, exist_ok=True)

HISTORICAL_ARCHIVE_PATH = DATA_DIR / 'historical_archive.csv'
FINAL_OUTPUT_PATH = DATA_DIR / 'master_data.csv'
ARCHIVE_LOCK_FILE = DATA_DIR / '.archive_created'

LOCATION_PAIRS_PATH = DATA_DIR / 'location_pairs_analysis.csv'
LOCATION_STATS_PATH = DATA_DIR / 'location_statistics.csv'
MULTI_LOCATION_DETAILS_PATH = DATA_DIR / 'multi_location_details.csv'

CHECKPOINT_DIR          = BASE_DIR / 'checkpoints'
PROGRESS_FILE           = CHECKPOINT_DIR / 'progress.json'
BATCH_CACHE_DIR         = CHECKPOINT_DIR / 'batches'

# ================= Logging Setup =================
def _setup_logger() -> logging.Logger:
    os.makedirs(CHECKPOINT_DIR, exist_ok=True)
    log_path = CHECKPOINT_DIR / 'pipeline.log'
    fmt = logging.Formatter(
        '%(asctime)s | %(levelname)-8s | %(name)s | %(message)s',
        datefmt='%Y-%m-%d %H:%M:%S'
    )
    fh = RotatingFileHandler(
        log_path, maxBytes=10 * 1024 * 1024, backupCount=5, encoding='utf-8'
    )
    fh.setFormatter(fmt)
    ch = logging.StreamHandler()
    ch.setFormatter(fmt)
    root = logging.getLogger('pipeline')
    root.setLevel(logging.INFO)
    if not root.handlers:
        root.addHandler(fh)
        root.addHandler(ch)
    return root

logger = _setup_logger()

# ================= API Settings =================
token = os.environ.get('API_TOKEN', '')
if not token:
    logger.warning('API_TOKEN not set. Make sure to add it to .env before running against a real API.')

# Sanitized generic API URL for showcase purposes
base_url   = "https://api.example-company.com/v1/trips/pagination"
order      = "desc"
sort_field = "serialId"
limit      = 30

headers = {
    "Authorization": f"bearer {token}",
    "Accept":        "application/json",
    "Content-Type":  "application/json",
}

# ================= Pipeline Configuration =================
@dataclass
class PipelineConfig:
    """Centralised config."""
    base_url:   str = base_url
    order:      str = 'desc'
    sort_field: str = 'serialId'
    limit:      int = 30
    batch_size: int = 5
    sleep_hours: int = 6
    max_workers: int = 5

CONFIG = PipelineConfig()

# ================= Checkpoint Manager =================
class CheckpointManager:
    def __init__(self):
        os.makedirs(CHECKPOINT_DIR, exist_ok=True)
        os.makedirs(BATCH_CACHE_DIR, exist_ok=True)

    def save_progress(self, year, current_page, total_records):
        progress = {
            'year': year, 'current_page': current_page,
            'total_records': total_records,
            'timestamp': datetime.now().isoformat(), 'status': 'in_progress',
        }
        with open(PROGRESS_FILE, 'w') as f:
            json.dump(progress, f, indent=2)

    def load_progress(self):
        if os.path.exists(PROGRESS_FILE):
            try:
                with open(PROGRESS_FILE, 'r') as f:
                    return json.load(f)
            except Exception:
                return None
        return None

    def mark_complete(self, year):
        progress = {'year': year, 'status': 'completed',
                    'timestamp': datetime.now().isoformat()}
        with open(PROGRESS_FILE, 'w') as f:
            json.dump(progress, f, indent=2)

    def save_batch(self, year, batch_num, data):
        batch_file = Path(BATCH_CACHE_DIR) / f'{year}_batch_{batch_num}.pkl'
        with open(batch_file, 'wb') as f:
            pickle.dump(data, f)

    def load_all_batches(self, year):
        all_data = []
        for batch_file in sorted(Path(BATCH_CACHE_DIR).glob(f'{year}_batch_*.pkl')):
            try:
                with open(batch_file, 'rb') as f:
                    all_data.extend(pickle.load(f))
            except Exception:
                print(f"[!] Failed to load {batch_file}")
        return all_data

    def clear_year_checkpoints(self, year):
        for batch_file in Path(BATCH_CACHE_DIR).glob(f'{year}_batch_*.pkl'):
            try:
                batch_file.unlink()
            except Exception:
                pass
        if os.path.exists(PROGRESS_FILE):
            try:
                os.remove(PROGRESS_FILE)
            except Exception:
                pass


# ================= Network Utilities =================
def wait_for_internet(max_retries=None, retry_delay=60):
    retry_count = 0
    while True:
        try:
            requests.get("https://www.google.com", timeout=5)
            if retry_count > 0:
                logger.info("Internet connection restored!")
            return True
        except Exception:
            retry_count += 1
            current_time = datetime.now().strftime('%H:%M:%S')
            if max_retries and retry_count >= max_retries:
                logger.warning("Max retries (%d) reached. Giving up.", max_retries)
                return False
            logger.warning("No Internet [%s] - Retry #%d in %ds...", current_time, retry_count, retry_delay)
            time.sleep(retry_delay)


def robust_api_call(func, *args, max_attempts=5, **kwargs):
    for attempt in range(1, max_attempts + 1):
        try:
            wait_for_internet(max_retries=3, retry_delay=30)
            return func(*args, **kwargs)
        except requests.exceptions.ConnectionError:
            logger.warning("Connection lost (Attempt %d/%d)", attempt, max_attempts)
            if attempt < max_attempts:
                time.sleep(min(60 * attempt, 300))
            else:
                return None
        except requests.exceptions.Timeout:
            logger.warning("Timeout (Attempt %d/%d)", attempt, max_attempts)
            if attempt < max_attempts:
                time.sleep(30)
            else:
                return None
        except Exception as e:
            logger.error("Unexpected error: %s", e)
            if attempt < max_attempts:
                time.sleep(30)
            else:
                return None
    return None


# =============================================================================
#  Location Cleaner — V6
# =============================================================================
from cleaner_v6 import AdvancedLocationCleanerV6

class LocationAnalytics:
    def __init__(self, df: pd.DataFrame, cleaner: AdvancedLocationCleanerV6 = None):
        self.df      = df
        self.cleaner = cleaner if cleaner is not None else AdvancedLocationCleanerV6()

    def generate_multi_location_columns(self, max_locations: int = 5) -> pd.DataFrame:
        print(" -> Generating multi-location columns...")
        self.df['all_locations_list'] = self.df['end_location_original'].apply(
            self.cleaner.extract_all_locations
        )
        self.df['num_locations'] = self.df['all_locations_list'].apply(len)
        for i in range(1, max_locations + 1):
            self.df[f'Location_{i}'] = self.df['all_locations_list'].apply(
                lambda x: x[i - 1] if len(x) >= i else None
            )
        self.df = self.df.drop(columns=['all_locations_list'])
        print(f"    OK — Created {max_locations} location columns")
        return self.df

    def analyze_location_pairs(self) -> pd.DataFrame:
        print(" -> Analyzing location pairs...")
        loc_cols = [f'Location_{i}' for i in range(1, 6)]
        avail    = [c for c in loc_cols if c in self.df.columns]
        multi    = self.df[self.df.get('num_locations', 0) >= 2] if 'num_locations' in self.df.columns else self.df
        if multi.empty:
            return pd.DataFrame()

        all_pairs = []
        for _, row in multi[avail + ['#', 'date', 'sale_price']].iterrows():
            locations = [
                row[c] for c in avail
                if pd.notna(row[c]) and str(row[c]).strip() not in ('', 'Undefined')
            ]
            if len(locations) >= 2:
                for pair in combinations(sorted(locations), 2):
                    all_pairs.append({
                        'Location_A': pair[0], 'Location_B': pair[1],
                        'Trip_ID': row.get('#', ''), 'Date': row.get('date', ''),
                        'Revenue': row.get('sale_price', 0),
                    })
        if not all_pairs:
            return pd.DataFrame()
        pairs_df      = pd.DataFrame(all_pairs)
        pairs_summary = pairs_df.groupby(['Location_A', 'Location_B']).agg(
            Frequency=('Trip_ID', 'count'), Total_Revenue=('Revenue', 'sum')
        ).reset_index().sort_values('Frequency', ascending=False)
        total_multi   = len(multi)
        pairs_summary['Percentage'] = (pairs_summary['Frequency'] / total_multi * 100).round(2)
        print(f"    OK — {len(pairs_summary)} unique location pairs")
        return pairs_summary

    def generate_location_statistics(self) -> pd.DataFrame:
        print(" -> Generating location statistics...")
        loc_cols = [f'Location_{i}' for i in range(1, 6)]
        avail = [c for c in loc_cols if c in self.df.columns]
        if not avail:
            return pd.DataFrame()
        melted = self.df.melt(
            id_vars=['#', 'sale_price', 'Trip_Type', 'date'],
            value_vars=avail,
            var_name='_loc_rank', value_name='Location'
        ).dropna(subset=['Location'])
        melted = melted[melted['Location'].astype(str).str.strip().ne('') &
                        melted['Location'].astype(str).ne('Undefined')]
        if melted.empty:
            return pd.DataFrame()
        melted['sale_price'] = pd.to_numeric(melted['sale_price'], errors='coerce').fillna(0)
        melted['Is_Primary'] = melted['_loc_rank'] == 'Location_1'
        stats = melted.groupby('Location').agg(
            Total_Visits=('#', 'count'),
            Total_Revenue=('sale_price', 'sum'),
            Primary_Destination_Count=('Is_Primary', 'sum'),
        ).reset_index().sort_values('Total_Visits', ascending=False)
        stats['Avg_Revenue_Per_Visit'] = (stats['Total_Revenue'] / stats['Total_Visits']).round(2)
        stats['Visit_Percentage']      = (stats['Total_Visits'] / len(self.df) * 100).round(2)
        return stats

    def create_multi_location_details(self) -> pd.DataFrame:
        multi_trips = self.df[self.df['num_locations'] >= 2].copy()
        if multi_trips.empty:
            return pd.DataFrame()

        def combine_locations(row):
            return ' → '.join(
                row[f'Location_{i}']
                for i in range(1, 6)
                if row.get(f'Location_{i}')
            )

        multi_trips['Trip_Route'] = multi_trips.apply(combine_locations, axis=1)
        cols = ['#', 'customer_name', 'date', 'sale_price', 'num_locations',
                'Trip_Route', 'Trip_Type', 'Location_1', 'Location_2',
                'Location_3', 'Location_4', 'Location_5']
        available = [c for c in cols if c in multi_trips.columns]
        return multi_trips[available]


# ================= Helper Functions =================

def create_session():
    session = requests.Session()
    retry   = Retry(
        total=5, backoff_factor=2,
        status_forcelist=[429, 500, 502, 503, 504],
        allowed_methods=["HEAD", "GET", "OPTIONS", "POST"],
    )
    adapter = HTTPAdapter(max_retries=retry, pool_connections=10, pool_maxsize=10)
    session.mount("https://", adapter)
    session.mount("http://",  adapter)
    return session

_location_cleaner_singleton = AdvancedLocationCleanerV6()
_SHARED_SESSION = create_session()

def fetch_page(page_num, date_from, date_to):
    def _fetch():
        url = f"{base_url}/{order}/{sort_field}/{page_num}/{limit}"
        payload = {
            "date_from": date_from,
            "date_to": date_to,
            "page": page_num,
            "limit": limit
        }
        
        session = _SHARED_SESSION
        try:
            response = session.post(url, headers=headers, json=payload, timeout=60)
            if response.status_code == 200:
                return response.json().get("data", [])
            elif response.status_code == 401:
                logger.critical("Token expired or invalid (401)! Update API_TOKEN in .env.")
                raise SystemExit("Token expired")
            else:
                response.raise_for_status()
        except Exception as e:
            logger.error("Page %d failed: %s", page_num, e)
            raise e

    result = robust_api_call(_fetch, max_attempts=3)
    if result is None:
        raise Exception(f"Failed to fetch page {page_num} after 3 attempts.")
    return result


# ================= Data Transformation =================

def apply_power_query_transformations(df: pd.DataFrame) -> pd.DataFrame:
    if df.empty:
        return df
    print(" -> Applying transformations...")

    for col in ['#', 'delegate_commission', 'driver_commission', 'entry_number',
                'trip_num', 'km_start', 'km_return', 'serial_number']:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors='coerce').fillna(0).astype('int64')

    if 'sale_price' in df.columns:
        df['sale_price'] = pd.to_numeric(df['sale_price'], errors='coerce').fillna(0)

    if 'date' in df.columns:
        df['date'] = pd.to_datetime(df['date'], errors='coerce')

    for col in ['customer_name', 'Guest_name', 'currency', 'delegate_name',
                'driver_name', 'start_location', 'end_location', 'reference_id',
                'station', 'receiver_name', 'payment_type', 'car_number']:
        if col in df.columns:
            df[col] = df[col].fillna('').astype(str).replace('nan', '')

    drop_cols = ['Guest_name', 'delegate_name', 'delegate_commission',
                 'driver_commission', 'trip_num', 'payment_type', 'serial_number']
    df = df.drop(columns=[c for c in drop_cols if c in df.columns])

    s = pd.to_numeric(df.get('km_start', 0),  errors='coerce').fillna(0)
    r = pd.to_numeric(df.get('km_return', 0), errors='coerce').fillna(0)
    df['Total_KM'] = np.where((s > 0) & (r > s), (r - s).astype(int), 0)

    def extract_plate(t: str) -> str:
        digits = re.sub(r'\D', '', str(t))
        return digits.lstrip('0') or 'No Plate'

    if 'car_number' in df.columns:
        df['Car_Num'] = df['car_number'].apply(extract_plate)

    if 'date' in df.columns:
        df['date'] = df['date'].apply(
            lambda x: f"{x.month}/{x.day}/{x.year}" if pd.notna(x) else ''
        )

    cleaner = _location_cleaner_singleton

    if 'end_location' in df.columns:
        df['end_location_original'] = df['end_location']
        df['_detected_locs'] = df['end_location'].apply(cleaner.extract_all_locations)
        df['End_Location_Clean'] = df['_detected_locs'].apply(
            lambda locs: locs[0] if locs else 'Undefined'
        )

        mask_undef = df['End_Location_Clean'] == 'Undefined'
        df.loc[mask_undef, 'End_Location_Clean'] = (
            df.loc[mask_undef, 'end_location'].apply(cleaner.extract_main_location)
        )

        df['Trip_Type'] = df.apply(
            lambda row: cleaner.categorize_trip_type(
                row['end_location'], row['_detected_locs']
            ),
            axis=1,
        )
        df['end_location'] = df['End_Location_Clean']
        df = df.drop(columns=['End_Location_Clean', '_detected_locs'])

    analytics = LocationAnalytics(df, cleaner=cleaner)
    df        = analytics.generate_multi_location_columns(max_locations=5)

    desired = [
        '#', 'customer_name', 'date', 'sale_price', 'currency',
        'driver_name', 'entry_number', 'start_location', 'end_location',
        'reference_id', 'km_start', 'km_return', 'Total_KM', 'station',
        'receiver_name', 'car_number', 'Car_Num', 'Trip_Type',
        'num_locations', 'Location_1', 'Location_2', 'Location_3',
        'Location_4', 'Location_5', 'end_location_original',
    ]
    df = df[[c for c in desired if c in df.columns]]
    return df


def structure_raw_data(all_raw_data: list) -> pd.DataFrame:
    rows = []
    for item in all_raw_data:
        if not item or item.get('confirm_status') is not True:
            continue
        entry   = item.get('entry') or {}
        station = item.get('station') or {}
        rows.append({
            '#':                  item.get('serialId'),
            'customer_name':      item.get('customer_name'),
            'date':               item.get('date'),
            'sale_price':         item.get('sale_price'),
            'currency':           item.get('currency_name'),
            'driver_name':        item.get('driver_name'),
            'entry_number':       entry.get('number'),
            'start_location':     item.get('start_location'),
            'end_location':       item.get('end_location'),
            'km_start':           item.get('km_start'),
            'km_return':          item.get('km_return'),
            'station':            station.get('name'),
        })
    return pd.DataFrame(rows)


def fetch_year_data_with_checkpoints(year, date_from, date_to, year_label):
    print(f"\n{'='*70}")
    print(f" Fetching {year_label}: {date_from} to {date_to}")
    print(f"{'='*70}")

    checkpoint = CheckpointManager()
    progress   = checkpoint.load_progress()
    resume_from, cached_data = 0, []

    if (progress and progress.get('year') == year and progress.get('status') == 'in_progress'):
        resume_from  = progress.get('current_page', 0)
        cached_data  = checkpoint.load_all_batches(year)

    wait_for_internet()

    all_raw_data   = cached_data
    current_page   = resume_from
    batch_size     = 5
    keep_fetching  = True
    batch_counter  = current_page // batch_size

    # Only fetch if a valid token is provided
    if token:
        while keep_fetching:
            wait_for_internet()
            pages_batch = range(current_page, current_page + batch_size)

            with ThreadPoolExecutor(max_workers=batch_size) as ex:
                results = [f.result() for f in [ex.submit(fetch_page, p, date_from, date_to) for p in pages_batch]]

            batch_data, batch_has_data = [], False
            for page_data in results:
                if page_data:
                    all_raw_data.extend(page_data)
                    batch_data.extend(page_data)
                    batch_has_data = True

            if batch_data:
                checkpoint.save_batch(year, batch_counter, batch_data)
                checkpoint.save_progress(year, current_page + batch_size, len(all_raw_data))
                batch_counter += 1

            keep_fetching = batch_has_data and bool(results[-1])
            if keep_fetching:
                current_page += batch_size
                time.sleep(0.2)

    df = structure_raw_data(all_raw_data)
    df = apply_power_query_transformations(df)

    checkpoint.mark_complete(year)
    checkpoint.clear_year_checkpoints(year)
    return df


def fetch_current_year_data() -> pd.DataFrame:
    current_year = datetime.now().year
    return fetch_year_data_with_checkpoints(
        year=current_year,
        date_from=f"01-01-{current_year}",
        date_to=datetime.now().strftime("%d-%m-%Y"),
        year_label=f"{current_year} (Current)",
    )


def generate_analytics_files(df: pd.DataFrame):
    print("\n" + "=" * 70)
    print(" GENERATING ANALYTICS FILES")
    analytics = LocationAnalytics(df)

    for path, fn, label in [
        (LOCATION_PAIRS_PATH,       analytics.analyze_location_pairs,       "Location Pairs"),
        (LOCATION_STATS_PATH,       analytics.generate_location_statistics, "Location Stats"),
        (MULTI_LOCATION_DETAILS_PATH, analytics.create_multi_location_details, "Multi-Location Details"),
    ]:
        try:
            result = fn()
            if not result.empty:
                result.to_csv(path, index=False, encoding='utf-8-sig')
        except Exception as e:
            logger.error(f"Failed to generate {label}: {e}")

# ================= MAIN =================
if __name__ == "__main__":
    print("\n" + "="*80)
    print(">> DATA PIPELINE v6.0 - SHOWCASE MODE")
    print("="*80)
    
    # In a real environment, this loop runs continuously
    # For showcase purposes, we execute one pass.
    try:
        logger.info("Pipeline triggered...")
        df_current = fetch_current_year_data()
        
        if not df_current.empty:
            df_current['#'] = df_current['#'].astype(str)
            df_current.to_csv(FINAL_OUTPUT_PATH, index=False, encoding='utf-8-sig')
            generate_analytics_files(df_current)
            logger.info("Pipeline execution complete. Output saved.")
        else:
            logger.warning("No data retrieved. Make sure API token is valid.")
            
    except Exception as e:
        logger.error("Execution failed", exc_info=True)