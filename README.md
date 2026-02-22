# 🚀 End-to-End Data Engineering & NLP Pipeline

[![Python Version](https://img.shields.io/badge/python-3.9%2B-blue.svg)](https://www.python.org/)
[![FastAPI](https://img.shields.io/badge/FastAPI-0.110.0-009688.svg?style=flat&logo=FastAPI&logoColor=white)](https://fastapi.tiangolo.com/)
[![Pandas](https://img.shields.io/badge/pandas-2.0%2B-150458.svg)](https://pandas.pydata.org/)
[![RapidFuzz](https://img.shields.io/badge/RapidFuzz-Fuzzy%20Matching-orange.svg)](https://maxbachmann.github.io/RapidFuzz/)
[![Power BI](https://img.shields.io/badge/Power_BI-F2C811?style=flat&logo=powerbi&logoColor=black)](https://powerbi.microsoft.com/)
[![Google Drive](https://img.shields.io/badge/Google_Drive-4285F4?style=flat&logo=googledrive&logoColor=white)](https://drive.google.com/)

This project demonstrates a production-grade **End-to-End Data Engineering Pipeline** designed to tackle a complex challenge: extracting messy, user-generated Arabic/English textual data from unstable APIs, applying advanced Natural Language Processing (NLP) to clean and categorize it, and feeding the structured data into a live **Power BI Dashboard** via automated server deployment.

## 🏗️ The End-to-End Workflow

```text
[Raw API Data] 
      │ (Fault-Tolerant Fetching & Checkpointing)
      ▼
[Python ETL & NLP Engine] ──(RapidFuzz & Regex)──> [Cleaned Data DataFrames]
      │
      ▼
[Local CSV Files on Server] 
      │ (Automated Cloud Sync)
      ▼
[Google Drive Cloud Storage]
      │ (Direct Live Connection)
      ▼
[📊 Power BI Dashboards] (Actionable Business Insights)

```

## 🎯 The Challenge

When dealing with transportation or logistics data, text inputs (like destination names) are often riddled with:

* **Severe Typos:** E.g., writing "مطاررر" instead of "مطار" (Airport).
* **Inconsistent Formats & Stopwords:** Mixing Arabic and English, or using different Arabic ligatures.
* **Operational Noise:** Prepending operational terms (e.g., "12 hour rental to...") before the actual location.
* **Network Instability:** Extracting tens of thousands of records from an API often fails midway due to timeouts.

## 🛠️ The Solution (Architecture)

To build a resilient, fault-tolerant system, I broke down the monolith into a **Microservices-inspired Architecture**:

### 1. The NLP Engine (`cleaner_v6.py`)

The brain of the operation. It utilizes a 3-Tier lookup strategy for `O(1)` equivalent performance:

* **Tier 1 (Exact Match):** Uses a pre-computed reverse index built from an external `locations.json` dictionary.
* **Tier 2 (Regex Match):** Pre-compiled, length-sorted regex patterns to catch complex strings.
* **Tier 3 (Fuzzy Matching fallback):** Integrates **RapidFuzz** (C++ powered) to intelligently guess highly misspelled words using `token_sort_ratio` without sacrificing speed.
* **Arabic Normalization:** Normalizes ligatures (lam-alef), removes Kashida (Tatweel), standardizes Alef/Ya variants, and strips embedded English stopwords.

### 2. The Extraction & ETL Pipeline (`New_code_v3.py`)

* **Fault-Tolerant Checkpointing:** Tracks API extraction progress. If the network drops at page 500, it resumes exactly from page 500, preventing data loss.
* **Concurrency:** Uses `ThreadPoolExecutor` to fetch API pages in parallel.
* **Pandas Transformations:** Vectorized operations (`np.where`) for high-performance data manipulation, calculating distances, and generating statistical analytics files.

### 3. Server Automation & Power BI Integration

* **24/7 Unattended Execution:** The pipeline is deployed on a dedicated server, running an infinite loop (or via Cron jobs) that wakes up every 6 hours to fetch the latest operational data.
* **Cloud Sync:** Cleaned datasets are automatically synced to **Google Drive**.
* **Business Intelligence:** **Power BI** is connected directly to the Google Drive source, allowing stakeholders to view live, interactive dashboards built on perfectly clean and categorized data.

### 4. The Controller API (`api.py`)

* **FastAPI Microservice:** Exposes the underlying python logic as RESTful endpoints (`/clean`, `/run-cycle`, `/health`).
* **Async & Thread Pools:** Heavy CPU-bound tasks are offloaded to `run_in_executor` so the API remains blazingly fast.
* **Non-Blocking Logging:** Implements `QueueHandler` and `QueueListener` to write logs asynchronously.

## 📁 Repository Structure

```text
├── api.py               # FastAPI server and endpoints
├── cleaner_v6.py        # Core NLP and matching algorithms
├── New_code_v3.py       # API extraction and Pandas ETL pipeline
├── locations.json       # Externalized dictionary mapping (Mock Data)
├── requirements.txt     # Dependencies
└── .env.example         # Template for environment variables

```

## 🚀 How to Run Locally

1. **Install Dependencies:**

```bash
pip install -r requirements.txt

```

2. **Environment Variables:**
Rename `.env.example` to `.env` and insert your dummy API token.
3. **Start the API:**

```bash
uvicorn api:app --reload

```

4. **Test the Magic:**
Navigate to `http://localhost:8000/docs` to use the Swagger UI. Try the `/clean` endpoint with a messy string like `"تشغيل 12 ساعه مطارررر القاهره وعوده"`.

## 💡 Key Takeaways

This project highlights advanced skills in **Data Engineering, ETL, NLP (Fuzzy Matching), REST APIs, Concurrent Programming, and Business Intelligence Integration**. It successfully bridges the gap between raw, messy API data and highly polished executive dashboards.