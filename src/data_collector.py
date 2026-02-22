"""
data_collector.py
=================
Phase 1 — Data Collection Layer

Fetches all required data for the Indian Gap Prediction Bot:
  1. Sensex OHLC (via yfinance)
  2. India VIX (via NSE session request)
  3. S&P 500 close (via yfinance)
  4. Gift Nifty proxy (via yfinance — ^NSEI futures)
  5. FII/DII net flows (via NSE API + NSDL fallback)
  6. BSE Sensex Option Chain snapshot (via NSE session)
  7. News sentiment headlines (via RSS feeds + VADER)

All data is saved as CSV into data/raw/ and inserted into SQLite DB.
"""

import os
import time
import logging
import sqlite3
import json
from datetime import datetime, date
from pathlib import Path

import yaml
import pandas as pd
import numpy as np
import yfinance as yf
import requests
import feedparser
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer

# ---------------------------------------------------------------------------
# Load config
# ---------------------------------------------------------------------------
CONFIG_PATH = Path(__file__).resolve().parent.parent / "config" / "settings.yaml"

with open(CONFIG_PATH, "r") as f:
    CFG = yaml.safe_load(f)

# ---------------------------------------------------------------------------
# Logging setup
# ---------------------------------------------------------------------------
LOG_DIR = Path(__file__).resolve().parent.parent / CFG["storage"]["logs_dir"]
LOG_DIR.mkdir(parents=True, exist_ok=True)

logging.basicConfig(
    level=getattr(logging, CFG["logging"]["level"]),
    format=CFG["logging"]["format"],
    datefmt=CFG["logging"]["date_format"],
    handlers=[
        logging.FileHandler(LOG_DIR / "data_collector.log"),
        logging.StreamHandler(),
    ],
)
logger = logging.getLogger("data_collector")

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------
BASE_DIR = Path(__file__).resolve().parent.parent
RAW_DIR = BASE_DIR / CFG["storage"]["raw_data_dir"]
PROC_DIR = BASE_DIR / CFG["storage"]["processed_data_dir"]
DB_PATH = BASE_DIR / CFG["storage"]["db_path"]

RAW_DIR.mkdir(parents=True, exist_ok=True)
PROC_DIR.mkdir(parents=True, exist_ok=True)
DB_PATH.parent.mkdir(parents=True, exist_ok=True)

# ---------------------------------------------------------------------------
# NSE Session Helper
# ---------------------------------------------------------------------------

NSE_HEADERS = {
    "User-Agent": (
        "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
        "AppleWebKit/537.36 (KHTML, like Gecko) "
        "Chrome/120.0.0.0 Safari/537.36"
    ),
    "Accept": "application/json, text/plain, */*",
    "Accept-Language": "en-US,en;q=0.9",
    "Accept-Encoding": "gzip, deflate, br",
    "Referer": "https://www.nseindia.com/",
    "Connection": "keep-alive",
}


def _get_nse_session() -> requests.Session:
    """
    Creates a requests Session with valid NSE cookies.
    NSE requires a cookie obtained by first visiting the main page.
    """
    session = requests.Session()
    session.headers.update(NSE_HEADERS)
    try:
        # Prime the session — get cookies
        resp = session.get("https://www.nseindia.com", timeout=10)
        resp.raise_for_status()
        time.sleep(1)  # Polite delay
        logger.info("NSE session initialised successfully.")
    except requests.RequestException as e:
        logger.error(f"Failed to initialise NSE session: {e}")
    return session


# ---------------------------------------------------------------------------
# 1. Sensex OHLC via yfinance
# ---------------------------------------------------------------------------

def fetch_sensex_ohlc(period: str = "5y") -> pd.DataFrame:
    """
    Downloads historical Sensex OHLC data via yfinance.

    Args:
        period: yfinance period string e.g. '5y', '1y', '6mo'

    Returns:
        DataFrame with columns: Date, Open, High, Low, Close, Volume
    """
    ticker = CFG["index"]["yfinance_ticker"]
    logger.info(f"Fetching Sensex OHLC | ticker={ticker} | period={period}")

    try:
        df = yf.download(ticker, period=period, progress=False, auto_adjust=True)
        if df.empty:
            logger.error("yfinance returned empty DataFrame for Sensex.")
            return pd.DataFrame()

        df.reset_index(inplace=True)
        df.columns = [c[0] if isinstance(c, tuple) else c for c in df.columns]
        df.rename(columns={"index": "Date"}, inplace=True)
        df["Date"] = pd.to_datetime(df["Date"]).dt.date

        save_path = RAW_DIR / "sensex_ohlc.csv"
        df.to_csv(save_path, index=False)
        logger.info(f"Sensex OHLC saved → {save_path} | rows={len(df)}")
        return df

    except Exception as e:
        logger.error(f"Sensex OHLC fetch failed: {e}")
        return pd.DataFrame()


# ---------------------------------------------------------------------------
# 2. India VIX via NSE
# ---------------------------------------------------------------------------

def fetch_india_vix(session: requests.Session) -> pd.DataFrame:
    """
    Fetches India VIX current value from NSE API.
    For historical VIX, falls back to yfinance (^VIX is US VIX;
    NSE historical VIX needs separate endpoint).

    Returns:
        DataFrame with columns: Date, VIX_Close
    """
    logger.info("Fetching India VIX from NSE...")

    # --- Live VIX value ---
    vix_url = "https://www.nseindia.com/api/allIndices"
    try:
        resp = session.get(vix_url, timeout=10)
        resp.raise_for_status()
        data = resp.json()

        vix_value = None
        for item in data.get("data", []):
            if item.get("index") == "INDIA VIX":
                vix_value = float(item.get("last", 0))
                break

        if vix_value is None:
            logger.warning("India VIX not found in NSE allIndices response.")
            return pd.DataFrame()

        today = date.today()
        df = pd.DataFrame([{"Date": today, "VIX_Close": vix_value}])
        save_path = RAW_DIR / "india_vix_live.csv"
        df.to_csv(save_path, index=False)
        logger.info(f"India VIX live value: {vix_value} | saved → {save_path}")
        return df

    except Exception as e:
        logger.error(f"India VIX live fetch failed: {e}")
        return pd.DataFrame()


def fetch_india_vix_historical(session: requests.Session,
                               from_date: str = None,
                               to_date: str = None) -> pd.DataFrame:
    """
    Fetches historical India VIX from NSE historical API.

    Args:
        from_date: 'DD-MM-YYYY' format. Defaults to 5 years ago.
        to_date:   'DD-MM-YYYY' format. Defaults to today.

    Returns:
        DataFrame with columns: Date, VIX_Open, VIX_High, VIX_Low, VIX_Close
    """
    from datetime import timedelta

    if to_date is None:
        to_dt = datetime.today()
    else:
        to_dt = datetime.strptime(to_date, "%d-%m-%Y")

    if from_date is None:
        from_dt = to_dt - timedelta(days=365 * 5)
    else:
        from_dt = datetime.strptime(from_date, "%d-%m-%Y")

    from_str = from_dt.strftime("%d-%m-%Y")
    to_str = to_dt.strftime("%d-%m-%Y")

    url = (
        f"https://www.nseindia.com/api/historical/vixhistory"
        f"?from={from_str}&to={to_str}"
    )
    logger.info(f"Fetching historical India VIX | {from_str} → {to_str}")

    try:
        resp = session.get(url, timeout=15)
        resp.raise_for_status()
        raw = resp.json()

        records = raw.get("data", [])
        if not records:
            logger.warning("No historical VIX data returned from NSE.")
            return pd.DataFrame()

        rows = []
        for rec in records:
            rows.append({
                "Date": pd.to_datetime(rec.get("EOD_TIMESTAMP"), dayfirst=True).date(),
                "VIX_Open": float(rec.get("EOD_OPEN_INDEX_VAL", 0)),
                "VIX_High": float(rec.get("EOD_HIGH_INDEX_VAL", 0)),
                "VIX_Low": float(rec.get("EOD_LOW_INDEX_VAL", 0)),
                "VIX_Close": float(rec.get("EOD_CLOSING_INDEX_VAL", 0)),
            })

        df = pd.DataFrame(rows).sort_values("Date").reset_index(drop=True)
        save_path = RAW_DIR / "india_vix_historical.csv"
        df.to_csv(save_path, index=False)
        logger.info(f"Historical VIX saved → {save_path} | rows={len(df)}")
        return df

    except Exception as e:
        logger.error(f"Historical India VIX fetch failed: {e}")
        return pd.DataFrame()


# ---------------------------------------------------------------------------
# 3. S&P 500 Close via yfinance
# ---------------------------------------------------------------------------

def fetch_sp500(period: str = "5y") -> pd.DataFrame:
    """
    Downloads S&P 500 historical OHLC via yfinance.

    Returns:
        DataFrame with columns: Date, SP500_Open, SP500_High,
                                 SP500_Low, SP500_Close
    """
    ticker = CFG["index"]["sp500_ticker"]
    logger.info(f"Fetching S&P 500 | ticker={ticker} | period={period}")

    try:
        df = yf.download(ticker, period=period, progress=False, auto_adjust=True)
        if df.empty:
            logger.error("yfinance returned empty DataFrame for S&P 500.")
            return pd.DataFrame()

        df.reset_index(inplace=True)
        df.columns = [c[0] if isinstance(c, tuple) else c for c in df.columns]
        df.rename(columns={
            "Date": "Date",
            "Open": "SP500_Open",
            "High": "SP500_High",
            "Low": "SP500_Low",
            "Close": "SP500_Close",
            "Volume": "SP500_Volume",
        }, inplace=True)
        df["Date"] = pd.to_datetime(df["Date"]).dt.date

        save_path = RAW_DIR / "sp500_ohlc.csv"
        df.to_csv(save_path, index=False)
        logger.info(f"S&P 500 saved → {save_path} | rows={len(df)}")
        return df

    except Exception as e:
        logger.error(f"S&P 500 fetch failed: {e}")
        return pd.DataFrame()


# ---------------------------------------------------------------------------
# 4. Gift Nifty Proxy via yfinance
# ---------------------------------------------------------------------------

def fetch_gift_nifty_proxy(period: str = "5y") -> pd.DataFrame:
    """
    Gift Nifty (SGX Nifty) is not directly available for free.
    We use ^NSEI (Nifty 50 index) as a proxy for overnight direction.
    In live trading, Gift Nifty pre-open move can be manually observed.

    Returns:
        DataFrame with columns: Date, GiftNifty_Close
    """
    ticker = CFG["index"]["gift_nifty_ticker"]
    logger.info(f"Fetching Gift Nifty proxy | ticker={ticker}")

    try:
        df = yf.download(ticker, period=period, progress=False, auto_adjust=True)
        if df.empty:
            logger.warning("Gift Nifty proxy: empty DataFrame.")
            return pd.DataFrame()

        df.reset_index(inplace=True)
        df.columns = [c[0] if isinstance(c, tuple) else c for c in df.columns]
        df = df[["Date", "Close"]].rename(columns={"Close": "GiftNifty_Close"})
        df["Date"] = pd.to_datetime(df["Date"]).dt.date

        save_path = RAW_DIR / "gift_nifty_proxy.csv"
        df.to_csv(save_path, index=False)
        logger.info(f"Gift Nifty proxy saved → {save_path} | rows={len(df)}")
        return df

    except Exception as e:
        logger.error(f"Gift Nifty proxy fetch failed: {e}")
        return pd.DataFrame()


# ---------------------------------------------------------------------------
# 5. FII/DII Net Flows
# ---------------------------------------------------------------------------

def fetch_fii_dii_nse(session: requests.Session) -> pd.DataFrame:
    """
    Fetches FII/DII trading activity from NSE API.
    This is the primary source — provisional intraday data.

    Returns:
        DataFrame with columns: Date, FII_Net_Buy, DII_Net_Buy
    """
    url = CFG["data_sources"]["fii_dii_url"]
    logger.info("Fetching FII/DII from NSE API...")

    try:
        resp = session.get(url, timeout=10)
        resp.raise_for_status()
        data = resp.json()

        rows = []
        for item in data:
            try:
                rows.append({
                    "Date": pd.to_datetime(item.get("date"), dayfirst=True).date(),
                    "FII_Net_Buy": float(str(item.get("fiiNetBuy", "0")).replace(",", "")),
                    "DII_Net_Buy": float(str(item.get("diiNetBuy", "0")).replace(",", "")),
                })
            except (ValueError, TypeError) as e:
                logger.warning(f"FII/DII row parse error: {e} | item={item}")

        if not rows:
            logger.warning("NSE FII/DII API returned no usable rows.")
            return pd.DataFrame()

        df = pd.DataFrame(rows).sort_values("Date").reset_index(drop=True)
        save_path = RAW_DIR / "fii_dii_nse.csv"
        df.to_csv(save_path, index=False)
        logger.info(f"FII/DII (NSE) saved → {save_path} | rows={len(df)}")
        return df

    except Exception as e:
        logger.error(f"NSE FII/DII fetch failed: {e}")
        return pd.DataFrame()


def fetch_fii_dii_nsdl() -> pd.DataFrame:
    """
    Fallback: Scrapes FII/DII data from NSDL website using pandas read_html.
    NSDL data is more reliable but updated less frequently.

    Returns:
        DataFrame with columns: Date, FII_Net_Buy (equity segment)
    """
    url = CFG["data_sources"]["nsdl_fii_url"]
    logger.info("Fetching FII data from NSDL (fallback)...")

    try:
        tables = pd.read_html(url, flavor="lxml")
        if not tables:
            logger.warning("NSDL page returned no HTML tables.")
            return pd.DataFrame()

        # NSDL typically has the data in the first or second table
        df = tables[0]
        logger.info(f"NSDL raw table columns: {list(df.columns)}")

        # Save raw for inspection
        save_path = RAW_DIR / "fii_dii_nsdl_raw.csv"
        df.to_csv(save_path, index=False)
        logger.info(f"NSDL raw FII data saved → {save_path}")
        return df

    except Exception as e:
        logger.error(f"NSDL FII/DII fetch failed: {e}")
        return pd.DataFrame()


# ---------------------------------------------------------------------------
# 6. BSE Sensex Option Chain Snapshot
# ---------------------------------------------------------------------------

def fetch_sensex_option_chain(session: requests.Session) -> pd.DataFrame:
    """
    Fetches the Sensex option chain snapshot from NSE API.

    BSE Sensex options are listed on NSE under symbol 'SENSEX'.
    Endpoint: https://www.nseindia.com/api/option-chain-indices?symbol=SENSEX

    Returns:
        DataFrame with option chain data (strikes, OI, IV, LTP etc.)
    """
    symbol = CFG["option_chain"]["symbol"]
    url = f"https://www.nseindia.com/api/option-chain-indices?symbol={symbol}"
    logger.info(f"Fetching Sensex option chain | symbol={symbol}")

    try:
        # Must re-prime session before option chain call
        session.get("https://www.nseindia.com/option-chain", timeout=10)
        time.sleep(1)

        resp = session.get(url, timeout=15)
        resp.raise_for_status()
        data = resp.json()

        records = data.get("records", {})
        chain_data = records.get("data", [])
        expiry_dates = records.get("expiryDates", [])
        underlying_value = records.get("underlyingValue", None)

        logger.info(
            f"Option chain: {len(chain_data)} records | "
            f"expiries={expiry_dates[:3]} | "
            f"underlying={underlying_value}"
        )

        rows = []
        snapshot_time = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

        for item in chain_data:
            strike = item.get("strikePrice")

            # CE side
            ce = item.get("CE", {})
            if ce:
                rows.append({
                    "Snapshot_Time": snapshot_time,
                    "Underlying_Value": underlying_value,
                    "Option_Type": "CE",
                    "Strike": strike,
                    "Expiry": ce.get("expiryDate"),
                    "LTP": ce.get("lastPrice"),
                    "OI": ce.get("openInterest"),
                    "OI_Change": ce.get("changeinOpenInterest"),
                    "IV": ce.get("impliedVolatility"),
                    "Volume": ce.get("totalTradedVolume"),
                    "Bid": ce.get("bidprice"),
                    "Ask": ce.get("askPrice"),
                })

            # PE side
            pe = item.get("PE", {})
            if pe:
                rows.append({
                    "Snapshot_Time": snapshot_time,
                    "Underlying_Value": underlying_value,
                    "Option_Type": "PE",
                    "Strike": strike,
                    "Expiry": pe.get("expiryDate"),
                    "LTP": pe.get("lastPrice"),
                    "OI": pe.get("openInterest"),
                    "OI_Change": pe.get("changeinOpenInterest"),
                    "IV": pe.get("impliedVolatility"),
                    "Volume": pe.get("totalTradedVolume"),
                    "Bid": pe.get("bidprice"),
                    "Ask": pe.get("askPrice"),
                })

        if not rows:
            logger.warning("Option chain: no rows parsed.")
            return pd.DataFrame()

        df = pd.DataFrame(rows)

        # Save with date-stamped filename
        today_str = date.today().strftime("%Y%m%d")
        save_path = RAW_DIR / f"sensex_option_chain_{today_str}.csv"
        df.to_csv(save_path, index=False)
        logger.info(f"Option chain saved → {save_path} | rows={len(df)}")
        return df

    except Exception as e:
        logger.error(f"Sensex option chain fetch failed: {e}")
        return pd.DataFrame()


# ---------------------------------------------------------------------------
# 7. News Sentiment via RSS + VADER
# ---------------------------------------------------------------------------

def fetch_news_sentiment() -> pd.DataFrame:
    """
    Fetches market news headlines from free RSS feeds.
    Runs VADER sentiment analysis on each headline.
    Returns aggregate daily sentiment score.

    Sources:
        - Moneycontrol RSS
        - Economic Times Markets RSS

    Returns:
        DataFrame with columns: Date, Headline, Source,
                                 Sentiment_Compound, Sentiment_Label
    """
    analyzer = SentimentIntensityAnalyzer()
    rss_sources = {
        "Moneycontrol": CFG["data_sources"]["moneycontrol_rss"],
        "EconomicTimes": CFG["data_sources"]["economic_times_rss"],
    }

    all_rows = []
    today = date.today()

    for source_name, rss_url in rss_sources.items():
        logger.info(f"Fetching RSS: {source_name} | {rss_url}")
        try:
            feed = feedparser.parse(rss_url)
            entries = feed.get("entries", [])
            logger.info(f"  → {len(entries)} entries from {source_name}")

            for entry in entries:
                title = entry.get("title", "")
                if not title:
                    continue

                scores = analyzer.polarity_scores(title)
                compound = scores["compound"]

                if compound >= 0.05:
                    label = "POSITIVE"
                elif compound <= -0.05:
                    label = "NEGATIVE"
                else:
                    label = "NEUTRAL"

                all_rows.append({
                    "Date": today,
                    "Source": source_name,
                    "Headline": title,
                    "Sentiment_Compound": compound,
                    "Sentiment_Label": label,
                })

        except Exception as e:
            logger.error(f"RSS fetch failed for {source_name}: {e}")

    if not all_rows:
        logger.warning("No sentiment data collected.")
        return pd.DataFrame()

    df = pd.DataFrame(all_rows)

    save_path = RAW_DIR / f"news_sentiment_{today.strftime('%Y%m%d')}.csv"
    df.to_csv(save_path, index=False)
    logger.info(f"Sentiment data saved → {save_path} | rows={len(df)}")

    # Compute daily aggregate score
    agg = df.groupby("Date")["Sentiment_Compound"].mean().reset_index()
    agg.rename(columns={"Sentiment_Compound": "Daily_Sentiment_Score"}, inplace=True)
    logger.info(f"Daily sentiment score: {agg['Daily_Sentiment_Score'].values[0]:.4f}")

    return df


# ---------------------------------------------------------------------------
# SQLite Database
# ---------------------------------------------------------------------------

def init_database() -> sqlite3.Connection:
    """
    Initialises SQLite database with required tables.

    Tables:
        sensex_ohlc          - Daily OHLC
        india_vix            - Daily VIX
        sp500_ohlc           - Daily S&P 500
        fii_dii              - Daily FII/DII flows
        option_chain         - Intraday option chain snapshots
        sentiment            - Daily sentiment scores
        predictions          - Model output predictions (Phase 3+)
    """
    conn = sqlite3.connect(DB_PATH)
    cursor = conn.cursor()

    cursor.executescript("""
        CREATE TABLE IF NOT EXISTS sensex_ohlc (
            Date        TEXT PRIMARY KEY,
            Open        REAL,
            High        REAL,
            Low         REAL,
            Close       REAL,
            Volume      REAL
        );

        CREATE TABLE IF NOT EXISTS india_vix (
            Date        TEXT PRIMARY KEY,
            VIX_Open    REAL,
            VIX_High    REAL,
            VIX_Low     REAL,
            VIX_Close   REAL
        );

        CREATE TABLE IF NOT EXISTS sp500_ohlc (
            Date            TEXT PRIMARY KEY,
            SP500_Open      REAL,
            SP500_High      REAL,
            SP500_Low       REAL,
            SP500_Close     REAL,
            SP500_Volume    REAL
        );

        CREATE TABLE IF NOT EXISTS fii_dii (
            Date            TEXT PRIMARY KEY,
            FII_Net_Buy     REAL,
            DII_Net_Buy     REAL
        );

        CREATE TABLE IF NOT EXISTS option_chain (
            id              INTEGER PRIMARY KEY AUTOINCREMENT,
            Snapshot_Time   TEXT,
            Underlying_Value REAL,
            Option_Type     TEXT,
            Strike          REAL,
            Expiry          TEXT,
            LTP             REAL,
            OI              REAL,
            OI_Change       REAL,
            IV              REAL,
            Volume          REAL,
            Bid             REAL,
            Ask             REAL
        );

        CREATE TABLE IF NOT EXISTS sentiment (
            id                      INTEGER PRIMARY KEY AUTOINCREMENT,
            Date                    TEXT,
            Source                  TEXT,
            Headline                TEXT,
            Sentiment_Compound      REAL,
            Sentiment_Label         TEXT
        );

        CREATE TABLE IF NOT EXISTS predictions (
            id                      INTEGER PRIMARY KEY AUTOINCREMENT,
            Prediction_Date         TEXT,
            Target_Date             TEXT,
            Regime                  TEXT,
            P_Gap_Up                REAL,
            P_Gap_Down              REAL,
            P_Flat                  REAL,
            Expected_Move_Pct       REAL,
            Option_Structure        TEXT,
            EV                      REAL,
            Signal_Sent             INTEGER DEFAULT 0
        );
    """)

    conn.commit()
    logger.info(f"SQLite database initialised → {DB_PATH}")
    return conn


def _upsert_df(conn: sqlite3.Connection, df: pd.DataFrame,
               table: str, pk_col: str) -> None:
    """
    Upserts DataFrame rows into SQLite table.
    Uses INSERT OR REPLACE to handle duplicates.
    """
    if df.empty:
        logger.warning(f"Skipping upsert — empty DataFrame for table: {table}")
        return

    df = df.copy()
    # Ensure date columns are strings for SQLite
    for col in df.columns:
        if "Date" in col or "date" in col:
            df[col] = df[col].astype(str)

    df.to_sql(table, conn, if_exists="append", index=False,
              method="multi")
    logger.info(f"Upserted {len(df)} rows → {table}")


def store_all_to_db(conn: sqlite3.Connection,
                    sensex_df: pd.DataFrame,
                    vix_hist_df: pd.DataFrame,
                    sp500_df: pd.DataFrame,
                    fii_dii_df: pd.DataFrame,
                    option_chain_df: pd.DataFrame,
                    sentiment_df: pd.DataFrame) -> None:
    """
    Stores all fetched DataFrames to the SQLite database.
    Uses INSERT OR REPLACE to avoid duplicates.
    """
    cursor = conn.cursor()

    # --- sensex_ohlc ---
    if not sensex_df.empty:
        cols = ["Date", "Open", "High", "Low", "Close", "Volume"]
        df_s = sensex_df[[c for c in cols if c in sensex_df.columns]].copy()
        df_s["Date"] = df_s["Date"].astype(str)
        cursor.executemany(
            "INSERT OR REPLACE INTO sensex_ohlc VALUES (?,?,?,?,?,?)",
            df_s.itertuples(index=False, name=None)
        )

    # --- india_vix ---
    if not vix_hist_df.empty:
        cols = ["Date", "VIX_Open", "VIX_High", "VIX_Low", "VIX_Close"]
        df_v = vix_hist_df[[c for c in cols if c in vix_hist_df.columns]].copy()
        df_v["Date"] = df_v["Date"].astype(str)
        cursor.executemany(
            "INSERT OR REPLACE INTO india_vix VALUES (?,?,?,?,?)",
            df_v.itertuples(index=False, name=None)
        )

    # --- sp500_ohlc ---
    if not sp500_df.empty:
        cols = ["Date", "SP500_Open", "SP500_High", "SP500_Low",
                "SP500_Close", "SP500_Volume"]
        df_sp = sp500_df[[c for c in cols if c in sp500_df.columns]].copy()
        df_sp["Date"] = df_sp["Date"].astype(str)
        cursor.executemany(
            "INSERT OR REPLACE INTO sp500_ohlc VALUES (?,?,?,?,?,?)",
            df_sp.itertuples(index=False, name=None)
        )

    # --- fii_dii ---
    if not fii_dii_df.empty and "FII_Net_Buy" in fii_dii_df.columns:
        cols = ["Date", "FII_Net_Buy", "DII_Net_Buy"]
        df_fii = fii_dii_df[[c for c in cols if c in fii_dii_df.columns]].copy()
        df_fii["Date"] = df_fii["Date"].astype(str)
        for _, row in df_fii.iterrows():
            cursor.execute(
                "INSERT OR REPLACE INTO fii_dii VALUES (?,?,?)",
                (str(row.get("Date")),
                 row.get("FII_Net_Buy", None),
                 row.get("DII_Net_Buy", None))
            )

    # --- option_chain ---
    if not option_chain_df.empty:
        cols = ["Snapshot_Time", "Underlying_Value", "Option_Type", "Strike",
                "Expiry", "LTP", "OI", "OI_Change", "IV", "Volume", "Bid", "Ask"]
        df_oc = option_chain_df[[c for c in cols if c in option_chain_df.columns]].copy()
        df_oc.to_sql("option_chain", conn, if_exists="append", index=False)

    # --- sentiment ---
    if not sentiment_df.empty:
        cols = ["Date", "Source", "Headline",
                "Sentiment_Compound", "Sentiment_Label"]
        df_sent = sentiment_df[[c for c in cols if c in sentiment_df.columns]].copy()
        df_sent["Date"] = df_sent["Date"].astype(str)
        df_sent.to_sql("sentiment", conn, if_exists="append", index=False)

    conn.commit()
    logger.info("All data committed to SQLite database.")


# ---------------------------------------------------------------------------
# Master Run Function
# ---------------------------------------------------------------------------

def run_data_collection(mode: str = "live") -> dict:
    """
    Master function that orchestrates all data collection.

    Args:
        mode: 'live'      — fetch today's data only
              'backtest'  — fetch 5 years of historical data

    Returns:
        dict of all DataFrames collected
    """
    logger.info(f"=== Data Collection START | mode={mode} ===")
    period = "5y" if mode == "backtest" else "5d"

    # Initialise DB
    conn = init_database()

    # Initialise NSE session (shared across all NSE calls)
    nse_session = _get_nse_session()

    # --- Collect all data ---
    sensex_df       = fetch_sensex_ohlc(period=period)
    vix_live_df     = fetch_india_vix(nse_session)
    vix_hist_df     = fetch_india_vix_historical(nse_session)
    sp500_df        = fetch_sp500(period=period)
    gift_df         = fetch_gift_nifty_proxy(period=period)
    fii_dii_df      = fetch_fii_dii_nse(nse_session)

    # NSDL fallback only if NSE FII/DII fails
    if fii_dii_df.empty:
        logger.warning("NSE FII/DII failed — trying NSDL fallback.")
        fii_dii_df = fetch_fii_dii_nsdl()

    option_chain_df = fetch_sensex_option_chain(nse_session)
    sentiment_df    = fetch_news_sentiment()

    # --- Store to DB ---
    store_all_to_db(
        conn=conn,
        sensex_df=sensex_df,
        vix_hist_df=vix_hist_df,
        sp500_df=sp500_df,
        fii_dii_df=fii_dii_df,
        option_chain_df=option_chain_df,
        sentiment_df=sentiment_df,
    )

    conn.close()
    logger.info("=== Data Collection COMPLETE ===")

    return {
        "sensex": sensex_df,
        "vix_live": vix_live_df,
        "vix_historical": vix_hist_df,
        "sp500": sp500_df,
        "gift_nifty": gift_df,
        "fii_dii": fii_dii_df,
        "option_chain": option_chain_df,
        "sentiment": sentiment_df,
    }


if __name__ == "__main__":
    results = run_data_collection(mode="backtest")
    for name, df in results.items():
        print(f"{name}: {len(df)} rows" if not df.empty else f"{name}: EMPTY")
