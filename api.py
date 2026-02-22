# -*- coding: utf-8 -*-
"""
api.py  —  Limousine Pipeline FastAPI Microservice
====================================================
Exposes the cleaner and pipeline as HTTP endpoints.

Endpoints
---------
GET  /health          Pipeline health & version
POST /clean           Clean a single location string
POST /clean/batch     Clean a list of locations (vectorised)
POST /run-cycle       Trigger one fetch -> transform -> save cycle
GET  /stats           Statistics from the last completed run

Run locally:
    uvicorn api:app --host 0.0.0.0 --port 8000 --reload

Environment variables (via .env):
    API_TOKEN  — JWT token for the upstream API
"""

from __future__ import annotations

import asyncio
import logging
import logging.handlers
import os
import queue
import sys
import time
from contextlib import asynccontextmanager
from datetime import datetime
from pathlib import Path
from typing import Any, AsyncGenerator, Optional

# ── FastAPI (required) ────────────────────────────────────────────────────────
try:
    from fastapi import FastAPI, HTTPException, Request, status
    from fastapi.middleware.cors import CORSMiddleware
    from fastapi.responses import JSONResponse
    from pydantic import BaseModel, Field
except ImportError:
    print("FastAPI not installed. Run: pip install fastapi uvicorn pydantic")
    sys.exit(1)

# ── Internal imports ──────────────────────────────────────────────────────────
try:
    from dotenv import load_dotenv
    load_dotenv()
except ImportError:
    pass

from cleaner_v6 import AdvancedLocationCleanerV6, load_locations

# ── Async-safe logging setup ──────────────────────────────────────────────────
# Uses QueueHandler + QueueListener so that file-IO never blocks the event loop.

_log_queue: queue.Queue[logging.LogRecord] = queue.Queue(maxsize=-1)

def _build_async_logger() -> logging.Logger:
    """Configure a non-blocking async-safe logger backed by QueueListener.

    Returns:
        Configured root 'limousine' logger.
    """
    log_dir  = Path(os.environ.get("LOG_DIR", "."))
    log_path = log_dir / "api.log"

    formatter = logging.Formatter(
        "%(asctime)s | %(levelname)-8s | %(name)s | %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )

    # Actual handlers (run in a background thread via QueueListener)
    file_handler    = logging.handlers.RotatingFileHandler(
        log_path, maxBytes=10 * 1024 * 1024, backupCount=5, encoding="utf-8"
    )
    console_handler = logging.StreamHandler(sys.stdout)
    for h in (file_handler, console_handler):
        h.setFormatter(formatter)

    # Queue handler — non-blocking, used by the async event loop
    queue_handler = logging.handlers.QueueHandler(_log_queue)

    root = logging.getLogger("limousine")
    root.setLevel(logging.INFO)
    root.handlers.clear()
    root.addHandler(queue_handler)

    # Listener runs in its own thread and drains the queue
    listener = logging.handlers.QueueListener(
        _log_queue, file_handler, console_handler, respect_handler_level=True
    )
    listener.start()
    return root


api_logger = _build_async_logger()

# ── App state ─────────────────────────────────────────────────────────────────
_cleaner:   Optional[AdvancedLocationCleanerV6] = None
_last_stats: dict[str, Any] = {}


# ── Lifespan (startup / shutdown) ─────────────────────────────────────────────
@asynccontextmanager
async def lifespan(app: FastAPI) -> AsyncGenerator[None, None]:
    """FastAPI lifespan: initialise cleaner on startup, log on shutdown."""
    global _cleaner
    api_logger.info("=" * 60)
    api_logger.info("Limousine API starting up")

    locations_path = os.environ.get(
        "LOCATIONS_PATH",
        str(Path(__file__).parent / "locations.json"),
    )

    # Graceful FileNotFoundError handled inside load_locations (logs critical + exits)
    try:
        _cleaner = AdvancedLocationCleanerV6(
            locations_path=locations_path,
            fuzzy_enabled=os.environ.get("FUZZY_ENABLED", "true").lower() == "true",
            fuzzy_cutoff=int(os.environ.get("FUZZY_CUTOFF", "82")),
        )
        api_logger.info("Cleaner V6 initialised successfully")
    except SystemExit:
        api_logger.critical(
            "Startup aborted — locations.json not found at '%s'. "
            "Run 'python export_locations.py' to generate it.",
            locations_path,
        )
        raise

    api_logger.info("API ready — listening for requests")
    yield

    # Shutdown
    api_logger.info("Limousine API shutting down")
    api_logger.info("=" * 60)


# ── FastAPI app ────────────────────────────────────────────────────────────────
app = FastAPI(
    title="Limousine Location Cleaner API",
    description=(
        "Production-grade Arabic/English location cleaning and trip categorisation. "
        "Powered by AdvancedLocationCleanerV6."
    ),
    version="6.0.0",
    lifespan=lifespan,
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["GET", "POST"],
    allow_headers=["*"],
)


# ── Request / Response models ─────────────────────────────────────────────────
class CleanRequest(BaseModel):
    text: str = Field(..., description="Raw trip location string to clean")
    fuzzy: bool = Field(True, description="Enable fuzzy matching for this request")


class CleanResponse(BaseModel):
    original:           str
    main_location:      str
    all_locations:      list[str]
    trip_type:          str
    processing_ms:      float


class BatchCleanRequest(BaseModel):
    texts: list[str] = Field(..., description="List of raw location strings")
    fuzzy: bool = Field(True)


class BatchCleanResponse(BaseModel):
    results:        list[CleanResponse]
    total:          int
    processing_ms:  float


class RunCycleResponse(BaseModel):
    status:       str
    records:      int
    started_at:   str
    finished_at:  str
    duration_s:   float


class StatsResponse(BaseModel):
    last_run_at:    Optional[str]
    records_saved:  Optional[int]
    status:         Optional[str]


# ── Helper ────────────────────────────────────────────────────────────────────
def _require_cleaner() -> AdvancedLocationCleanerV6:
    if _cleaner is None:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="Cleaner not initialised. Check startup logs.",
        )
    return _cleaner


def _clean_one(
    cleaner: AdvancedLocationCleanerV6,
    text: str,
) -> CleanResponse:
    t0   = time.perf_counter()
    locs = cleaner.extract_all_locations(text)
    main = locs[0] if locs else cleaner.extract_main_location(text)
    typ  = cleaner.categorize_trip_type(text, locs)
    ms   = (time.perf_counter() - t0) * 1000
    return CleanResponse(
        original=text,
        main_location=main,
        all_locations=locs,
        trip_type=typ,
        processing_ms=round(ms, 3),
    )


# ── Middleware: request logging ───────────────────────────────────────────────
@app.middleware("http")
async def log_requests(request: Request, call_next):  # type: ignore[no-untyped-def]
    """Log every request and response asynchronously."""
    start   = time.perf_counter()
    response = await call_next(request)
    ms      = (time.perf_counter() - start) * 1000

    # Schedule log write without blocking the event loop
    loop = asyncio.get_event_loop()
    await loop.run_in_executor(
        None,
        api_logger.info,
        "%s %s -> %d (%.1f ms)",
        request.method,
        request.url.path,
        response.status_code,
        ms,
    )
    return response


# ── Endpoints ─────────────────────────────────────────────────────────────────
@app.get("/health", summary="Health check")
async def health() -> dict[str, str]:
    """Return API status and cleaner readiness."""
    ready = _cleaner is not None
    api_logger.info("Health check — cleaner ready: %s", ready)
    return {
        "status":  "ok" if ready else "degraded",
        "cleaner": "ready" if ready else "not initialised",
        "version": "6.0.0",
        "time":    datetime.now().isoformat(),
    }


@app.post("/clean", response_model=CleanResponse, summary="Clean a single location")
async def clean_single(req: CleanRequest) -> CleanResponse:
    """Clean and categorise a single Arabic/English location string.

    Args (body):
        text:  Raw location string.
        fuzzy: Whether to use fuzzy matching (default: true).
    """
    cleaner = _require_cleaner()
    loop    = asyncio.get_event_loop()
    api_logger.debug("POST /clean — text length: %d", len(req.text))

    # Run CPU-bound work in thread-pool to avoid blocking event loop
    result: CleanResponse = await loop.run_in_executor(
        None, _clean_one, cleaner, req.text
    )
    return result


@app.post(
    "/clean/batch",
    response_model=BatchCleanResponse,
    summary="Clean a list of locations",
)
async def clean_batch(req: BatchCleanRequest) -> BatchCleanResponse:
    """Clean and categorise a batch of location strings.

    Args (body):
        texts:  List of raw location strings.
        fuzzy:  Whether to use fuzzy matching (default: true).
    """
    cleaner = _require_cleaner()
    loop    = asyncio.get_event_loop()
    api_logger.info("POST /clean/batch — n=%d", len(req.texts))

    t0 = time.perf_counter()

    def _run_batch() -> list[CleanResponse]:
        return [_clean_one(cleaner, t) for t in req.texts]

    results = await loop.run_in_executor(None, _run_batch)
    total_ms = (time.perf_counter() - t0) * 1000

    return BatchCleanResponse(
        results=results,
        total=len(results),
        processing_ms=round(total_ms, 3),
    )


@app.post(
    "/run-cycle",
    response_model=RunCycleResponse,
    summary="Trigger one fetch + transform + save cycle",
)
async def run_cycle() -> RunCycleResponse:
    """Trigger a single data pipeline cycle (fetch current year data, merge, save).

    This endpoint makes the while-True loop pattern unnecessary — schedule
    this via Airflow, cron, or a task queue instead.
    """
    api_logger.info("POST /run-cycle — pipeline cycle triggered")
    started_at = datetime.now().isoformat()
    t0 = time.perf_counter()

    loop = asyncio.get_event_loop()

    def _run() -> int:
        # Import here to avoid circular dependency at module load
        try:
            from New_code_v3 import fetch_current_year_data  # type: ignore[import]
            df = fetch_current_year_data()
            records = len(df) if df is not None and not df.empty else 0
            api_logger.info("Pipeline cycle complete — %d records", records)
            return records
        except Exception as exc:
            api_logger.error("Pipeline cycle failed: %s", exc, exc_info=True)
            raise

    try:
        records = await loop.run_in_executor(None, _run)
    except Exception as exc:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Pipeline run failed: {exc}",
        ) from exc

    finished_at = datetime.now().isoformat()
    duration_s  = round(time.perf_counter() - t0, 2)

    global _last_stats
    _last_stats = {
        "last_run_at":   finished_at,
        "records_saved": records,
        "status":        "success",
    }

    return RunCycleResponse(
        status="success",
        records=records,
        started_at=started_at,
        finished_at=finished_at,
        duration_s=duration_s,
    )


@app.get(
    "/stats",
    response_model=StatsResponse,
    summary="Last run statistics",
)
async def stats() -> StatsResponse:
    """Return statistics from the most recent pipeline run."""
    api_logger.info("GET /stats")
    return StatsResponse(
        last_run_at=_last_stats.get("last_run_at"),
        records_saved=_last_stats.get("records_saved"),
        status=_last_stats.get("status"),
    )


# ── Dev entrypoint ────────────────────────────────────────────────────────────
if __name__ == "__main__":
    try:
        import uvicorn
        uvicorn.run("api:app", host="0.0.0.0", port=8000, reload=True)
    except ImportError:
        print("Install uvicorn: pip install uvicorn")
