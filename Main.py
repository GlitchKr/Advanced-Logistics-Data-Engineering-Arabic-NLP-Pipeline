# -*- coding: utf-8 -*-
"""
New_code_v3.py  —  Limousine Pipeline v6.0
==========================================
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
from collections import Counter
from itertools import combinations
from typing import Optional
from dataclasses import dataclass, field
from functools import lru_cache
import logging
from logging.handlers import RotatingFileHandler
try:
    from dotenv import load_dotenv
    load_dotenv()
except ImportError:
    pass  # install with: pip install python-dotenv

# ================= Configuration =================
HISTORICAL_ARCHIVE_PATH = r'D:\Work Projects\code\Projects\1\historical_archive_2024_2025.csv'
FINAL_OUTPUT_PATH = r'D:\Work Projects\code\Projects\1\limousine_data.csv'
ARCHIVE_LOCK_FILE = r'D:\Work Projects\code\Projects\1\.archive_created'

LOCATION_PAIRS_PATH = r'D:\Work Projects\code\Projects\1\location_pairs_analysis.csv'
LOCATION_STATS_PATH = r'D:\Work Projects\code\Projects\1\location_statistics.csv'
MULTI_LOCATION_DETAILS_PATH = r'D:\Work Projects\code\Projects\1\multi_location_details.csv'

CHECKPOINT_DIR          = r'D:\Work Projects\code\Projects\1\checkpoints'
PROGRESS_FILE           = f'{CHECKPOINT_DIR}/progress.json'
BATCH_CACHE_DIR         = f'{CHECKPOINT_DIR}/batches'

# ================= Logging Setup =================
def _setup_logger() -> logging.Logger:
    os.makedirs(CHECKPOINT_DIR, exist_ok=True)
    log_path = os.path.join(CHECKPOINT_DIR, 'pipeline.log')
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
    root = logging.getLogger('limousine')
    root.setLevel(logging.INFO)
    if not root.handlers:
        root.addHandler(fh)
        root.addHandler(ch)
    return root

logger = _setup_logger()


# API Settings
# Load token from environment / .env file
# Create .env in the same folder: API_TOKEN=eyJ...
token = os.environ.get('API_TOKEN', '')
if not token:
    raise EnvironmentError(
        'API_TOKEN not set. Add it to .env or set as environment variable.'
    )
base_url   = "https://portal-api.etolv.net/public/api/airline_delivery/pagination"
order      = "desc"
sort_field = "serialId"
limit      = 50

headers = {
    "Authorization": f"bearer {token}",
    "Accept":        "application/json",
    "Content-Type":  "application/json",
}




# ================= Pipeline Configuration =================
@dataclass
class PipelineConfig:
    """Centralised config - edit here, not in scattered constants."""
    base_url:   str = 'https://portal-api.etolv.net/public/api/airline_delivery/pagination'
    order:      str = 'desc'
    sort_field: str = 'serialId'
    limit:      int = 50
    batch_size: int = 5
    sleep_hours: int = 6
    max_workers: int = 5

    @property
    def headers(self) -> dict:
        return {
            'Authorization': f'bearer {token}',
            'Accept':        'application/json',
            'Content-Type':  'application/json',
        }

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
        batch_file = f'{BATCH_CACHE_DIR}/{year}_batch_{batch_num}.pkl'
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
#  Imported from cleaner_v6.py (external JSON + RapidFuzz + strict typing)
# =============================================================================
from cleaner_v6 import AdvancedLocationCleanerV6
# Alias for backward compatibility with any code that references V5
AdvancedLocationCleanerV5 = AdvancedLocationCleanerV6


# =============================================================================
#  LocationAnalytics  (لا تغييرات منطقية، فقط يستخدم الـ V5 cleaner)
# =============================================================================

class LocationAnalytics:
    def __init__(self, df: pd.DataFrame, cleaner: AdvancedLocationCleanerV6 = None):
        self.df      = df
        # Accept a shared cleaner (V6) to avoid re-instantiating
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
            print("    No multi-location trips found")
            return pd.DataFrame()

        all_pairs = []
        # Only iterate multi-destination rows (much smaller subset)
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
            print("    No multi-location trips found")
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
        print(f"    OK — {len(stats)} unique locations")
        return stats

    def create_multi_location_details(self) -> pd.DataFrame:
        print(" -> Creating multi-location trip details...")
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
        print(f"    OK — {len(multi_trips)} multi-location trips")
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


# Singleton cleaner — built once, shared everywhere (avoids double __init__)
_location_cleaner_singleton = AdvancedLocationCleanerV5()


# ─── Location lookup cache (avoids re-processing repeated strings) ──────────
@lru_cache(maxsize=8192)
def _cached_extract_main(text: str) -> str:
    """Cached wrapper - identical strings are looked up only once."""
    return _location_cleaner_singleton.extract_main_location(text)


# Module-level shared session — reused across all threads (HTTP keep-alive)
_SHARED_SESSION = create_session()


def fetch_page(page_num, date_from, date_to):
    """سحب صفحة واحدة مع معالجة الأخطاء - النسخة الآمنة"""
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
            # بنحاول نكلم السيرفر
            response = session.post(url, headers=headers, json=payload, timeout=60)
            
            # لو الرد 200 (تمام)
            if response.status_code == 200:
                return response.json().get("data", [])
            elif response.status_code == 401:
                logger.critical(
                    "Token expired or invalid (401)! Update API_TOKEN in .env and restart."
                )
                raise SystemExit("Token expired")
            else:
                # لو أي كود تاني (500, 404, etc) ارفع Error
                response.raise_for_status()
                
        except Exception as e:
            # ===> (هنا الجزء اللي أنت سألت عليه) <===
            # بنطبع المشكلة ونرفعها عشان robust_api_call يعيد المحاولة
            logger.error("Page %d failed: %s", page_num, e)
            raise e

    # حاول 3 مرات، لو فشلوا كلهم robust_api_call هيرجع None
    result = robust_api_call(_fetch, max_attempts=3)
    
    # اللحظة الحاسمة: لو رجع None معناها فشل تماماً بعد 3 محاولات
    if result is None:
        # ارفع Exception عشان نوقف الباتش ده ومنحفظش Checkpoint غلط
        raise Exception(f"Failed to fetch page {page_num} after 3 attempts.")
        
    return result


# ================= Data Transformation =================

def apply_power_query_transformations(df: pd.DataFrame) -> pd.DataFrame:
    if df.empty:
        return df
    print(" -> Applying transformations...")

    # أعمدة رقمية
    for col in ['#', 'delegate_commission', 'driver_commission', 'entry_number',
                'trip_num', 'km_start', 'km_return', 'serial_number']:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors='coerce').fillna(0).astype('int64')

    if 'sale_price' in df.columns:
        df['sale_price'] = pd.to_numeric(df['sale_price'], errors='coerce').fillna(0)

    if 'date' in df.columns:
        df['date'] = pd.to_datetime(df['date'], errors='coerce')

    # نصوص
    for col in ['customer_name', 'Guest_name', 'currency', 'delegate_name',
                'driver_name', 'start_location', 'end_location', 'reference_id',
                'station', 'receiver_name', 'payment_type', 'car_number']:
        if col in df.columns:
            df[col] = df[col].fillna('').astype(str).replace('nan', '')

    # حذف أعمدة غير مطلوبة
    drop_cols = ['Guest_name', 'delegate_name', 'delegate_commission',
                 'driver_commission', 'trip_num', 'payment_type', 'serial_number']
    df = df.drop(columns=[c for c in drop_cols if c in df.columns])

    # الكيلومترات — vectorised with np.where (no Python-level row loop)
    s = pd.to_numeric(df['km_start'],  errors='coerce').fillna(0)
    r = pd.to_numeric(df['km_return'], errors='coerce').fillna(0)
    df['Total_KM'] = np.where((s > 0) & (r > s), (r - s).astype(int), 0)

    # رقم السيارة - يحذف اصفار البداية
    def extract_plate(t: str) -> str:
        digits = re.sub(r'\D', '', str(t))
        return digits.lstrip('0') or 'No Plate'

    df['Car_Num'] = df['car_number'].apply(extract_plate)

    # تنسيق التاريخ
    if 'date' in df.columns:
        df['date'] = df['date'].apply(
            lambda x: f"{x.month}/{x.day}/{x.year}" if pd.notna(x) else ''
        )

    # ── استخدام المنظّف V5 ──────────────────────────────────────────────────
    cleaner = AdvancedLocationCleanerV6()

    # start_location
    if 'start_location' in df.columns:
        start_map = {
            'راديسون بلو':  'Radisson Blu',
            'لوبساج':       'Le Passage',
            'سفير  الدقي':  'Safir Dokki',
            'فندق سفير':    'Safir',
        }
        df['start_location'] = df['start_location'].replace(start_map)

    # end_location  ─────────────────────────────────────────────────────────
    if 'end_location' in df.columns:
        df['end_location_original'] = df['end_location']

        # استخرج المواقع أولاً (المصدر الحقيقي للحقيقة)
        df['_detected_locs'] = df['end_location'].apply(cleaner.extract_all_locations)

        # الموقع الرئيسي مشتق من المواقع المكتشفة
        df['End_Location_Clean'] = df['_detected_locs'].apply(
            lambda locs: locs[0] if locs else 'Undefined'
        )

        # للصفوف التي لم يُكتشف فيها موقع → جرّب extract_main_location
        mask_undef = df['End_Location_Clean'] == 'Undefined'
        df.loc[mask_undef, 'End_Location_Clean'] = (
            df.loc[mask_undef, 'end_location'].apply(cleaner.extract_main_location)
        )

        # Trip_Type مشتق من المواقع المكتشفة + النص الأصلي
        df['Trip_Type'] = df.apply(
            lambda row: cleaner.categorize_trip_type(
                row['end_location'], row['_detected_locs']
            ),
            axis=1,
        )

        df['end_location'] = df['End_Location_Clean']
        df = df.drop(columns=['End_Location_Clean', '_detected_locs'])

    # ── أعمدة المواقع المتعددة ──────────────────────────────────────────────
    analytics = LocationAnalytics(df, cleaner=cleaner)  # reuse the same cleaner
    df        = analytics.generate_multi_location_columns(max_locations=5)

    # ترتيب الأعمدة
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
            'Guest_name':         None,
            'date':               item.get('date'),
            'sale_price':         item.get('sale_price'),
            'currency':           item.get('currency_name'),
            'delegate_name':      item.get('delegate_name'),
            'delegate_commission':item.get('delegate_commission'),
            'driver_name':        item.get('driver_name'),
            'driver_commission':  0,
            'entry_number':       entry.get('number'),
            'start_location':     item.get('start_location'),
            'end_location':       item.get('end_location'),
            'trip_num':           item.get('trip_num'),
            'reference_id':       item.get('reference_id'),
            'km_start':           item.get('km_start'),
            'km_return':          item.get('km_return'),
            'station':            station.get('name'),
            'receiver_name':      item.get('receiver_name'),
            'payment_type':       item.get('payment_type_name'),
            'serial_number':      item.get('serial_number'),
            'car_number':         item.get('platesNo'),
        })
    return pd.DataFrame(rows)


# ================= Data Fetching with Checkpoints =================

def fetch_year_data_with_checkpoints(year, date_from, date_to, year_label):
    print(f"\n{'='*70}")
    print(f" Fetching {year_label}: {date_from} to {date_to}")
    print(f"{'='*70}")

    checkpoint = CheckpointManager()
    progress   = checkpoint.load_progress()
    resume_from, cached_data = 0, []

    if (progress and progress.get('year') == year
            and progress.get('status') == 'in_progress'):
        resume_from  = progress.get('current_page', 0)
        cached_data  = checkpoint.load_all_batches(year)
        print(f"[!] Resuming from page {resume_from} — {len(cached_data):,} cached records")

    wait_for_internet()

    all_raw_data   = cached_data
    current_page   = resume_from
    batch_size     = 5
    keep_fetching  = True
    batch_counter  = current_page // batch_size

    while keep_fetching:
        wait_for_internet(max_retries=10, retry_delay=30)
        pages_batch = range(current_page, current_page + batch_size)

        with ThreadPoolExecutor(max_workers=batch_size) as ex:
            results = [f.result() for f in
                       [ex.submit(fetch_page, p, date_from, date_to) for p in pages_batch]]

        batch_data, batch_has_data = [], False
        for page_data in results:
            if page_data:
                all_raw_data.extend(page_data)
                batch_data.extend(page_data)
                batch_has_data = True

        if batch_data:
            checkpoint.save_batch(year, batch_counter, batch_data)
            checkpoint.save_progress(year, current_page + batch_size, len(all_raw_data))
            print(f" -> Pages {current_page}-{current_page+batch_size}: "
                  f"{len(all_raw_data):,} total | checkpoint saved")
            batch_counter += 1

        keep_fetching = batch_has_data and bool(results[-1])
        if keep_fetching:
            current_page += batch_size
            time.sleep(0.2)

    print(f" [OK] Finished — {len(all_raw_data):,} raw records")
    df = structure_raw_data(all_raw_data)
    print(f" [OK] Confirmed: {len(df):,} records")
    df = apply_power_query_transformations(df)

    checkpoint.mark_complete(year)
    checkpoint.clear_year_checkpoints(year)
    return df


def create_historical_archive() -> pd.DataFrame:
    print("\n" + "=" * 70)
    print(" CREATING HISTORICAL ARCHIVE (2024 + 2025)")
    print("=" * 70)
    wait_for_internet()
    df_2024 = fetch_year_data_with_checkpoints(2024, "01-01-2024", "31-12-2024", "2024")
    wait_for_internet()
    df_2025 = fetch_year_data_with_checkpoints(2025, "01-01-2025", "31-12-2025", "2025")

    print("\n Merging 2024 + 2025 ...")
    df_archive = pd.concat([df_2024, df_2025], ignore_index=True)
    df_archive = df_archive.drop_duplicates(subset=['#'], keep='last')
    print(f"[OK] Archive: {len(df_2024):,} + {len(df_2025):,} = {len(df_archive):,}")

    os.makedirs(os.path.dirname(HISTORICAL_ARCHIVE_PATH), exist_ok=True)
    df_archive.to_csv(HISTORICAL_ARCHIVE_PATH, index=False, encoding='utf-8-sig')
    with open(ARCHIVE_LOCK_FILE, 'w') as f:
        f.write(f"Archive created: {datetime.now().isoformat()}\n"
                f"Records: {len(df_archive):,}\n")
    return df_archive


def fetch_current_year_data() -> pd.DataFrame:
    current_year = datetime.now().year
    return fetch_year_data_with_checkpoints(
        year=current_year,
        date_from=f"01-01-{current_year}",
        date_to=datetime.now().strftime("%d-%m-%Y"),
        year_label=f"{current_year} (Current)",
    )


# ================= Analytics Generation =================

def generate_analytics_files(df: pd.DataFrame):
    print("\n" + "=" * 70)
    print(" GENERATING ANALYTICS FILES")
    print("=" * 70)
    analytics = LocationAnalytics(df)

    for path, fn, label in [
        (LOCATION_PAIRS_PATH,       analytics.analyze_location_pairs,       "Location Pairs"),
        (LOCATION_STATS_PATH,       analytics.generate_location_statistics, "Location Stats"),
        (MULTI_LOCATION_DETAILS_PATH, analytics.create_multi_location_details, "Multi-Location Details"),
    ]:
        try:
            result = fn()
            if not result.empty:
                os.makedirs(os.path.dirname(path), exist_ok=True)
                result.to_csv(path, index=False, encoding='utf-8-sig')
                print(f" OK — {label}: {path}")
        except Exception as e:
            print(f" FAILED — {label}: {e}")

    print("=" * 70)


# ================= MAIN =================

# =============================================================================
#  ROBUST MAIN LOOP 
# =============================================================================

if __name__ == "__main__":
    print("\n" + "="*80)
    print(">> LIMOUSINE PIPELINE v5.0 - PRODUCTION MODE (SAFE)")
    print("   Word-Boundary Cleaning | Auto-Retry | No Data Loss")
    print("="*80)
    
    wait_for_internet()
    
    # Historical Archive
    archive_exists = os.path.exists(HISTORICAL_ARCHIVE_PATH) and os.path.exists(ARCHIVE_LOCK_FILE)
    
    if not archive_exists:
        print("\n[!] Creating historical archive (ONE TIME)")
        try:
            df_history = create_historical_archive()
            print("[OK] Archive created successfully!")
        except Exception as e:
            print(f"[!!!] Failed to create archive: {e}")
            exit()
    else:
        print("\n[OK] Loading historical archive...")
        df_history = pd.read_csv(HISTORICAL_ARCHIVE_PATH, dtype={'#': str})
        print(f"    {len(df_history):,} records (2024-2025)")
    
    # === الحلقة اللانهائية (The Infinite Loop) ===
    while True:
        try:
            wait_for_internet()
            
            logger.info("=" * 60)
            logger.info(">> Job Started: %s", datetime.now().strftime("%Y-%m-%d %H:%M:%S"))
            logger.info("=" * 60)
            
            # محاولة سحب الداتا (لو فشلت هنا، هتروح للـ except ومش هتحفظ داتا ناقصة)
            df_current = fetch_current_year_data()
            
            if not df_current.empty:
                df_current['#'] = df_current['#'].astype(str)
                
                print(f"\n Current Year Summary:")
                print(f"   Records:            {len(df_current):,}")
                print(f"   Multi-Location:     {(df_current['num_locations'] >= 2).sum():,}")
                print(f"   Round Trips:        {(df_current['Trip_Type'] == 'Round Trip').sum():,}")
                print(f"   Airport Transfers:  {(df_current['Trip_Type'] == 'Airport Transfer').sum():,}")
                
                print("\n Merging...")
                df_master = pd.concat([df_history, df_current], ignore_index=True)
                df_master = df_master.drop_duplicates(subset=['#'], keep='last')
                print(f" [OK] Master: {len(df_master):,} unique records")
                
                # Save & Analyze
                os.makedirs(os.path.dirname(FINAL_OUTPUT_PATH), exist_ok=True)
                df_master.to_csv(FINAL_OUTPUT_PATH, index=False, encoding='utf-8-sig')
                print(f"\n[OK] Main File: {FINAL_OUTPUT_PATH}")
                
                generate_analytics_files(df_master)
                
                # Stats Summary
                print(f"\n{'='*70}")
                print(f" Total Revenue: {df_master['sale_price'].sum():,.2f}")
                print(f"{'='*70}")
                
                # Success -> Sleep 6 Hours
                next_run = datetime.now() + pd.Timedelta(hours=6)
                logger.info("Success! Sleeping 6 hours. Next run: %s", next_run.strftime("%H:%M"))
                time.sleep(21600)
                
            else:
                print("[!] No data found / Empty response. Retry in 5 min...")
                time.sleep(300)

        except KeyboardInterrupt:
            logger.info("Pipeline stopped by user (KeyboardInterrupt).")
            break
            
        except Exception as e:
            # ممتص الصدمات: لو حصل Error في السحب أو النت، السكريبت مش بيموت
            # لكنه برضه مش بيسجل إن الشغل خلص، فبيعيد المحاولة بعدين
            logger.error("CRITICAL ERROR: %s -- retrying in 5 min", e, exc_info=True)
            time.sleep(300)