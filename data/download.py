# @author: Íñigo Martínez Jiménez
# This module handles all communication with the Binance REST API to retrieve
# historical OHLCV kline data and store it as a raw CSV file for later processing.

from pathlib import Path
import requests
import pandas as pd
import yaml

# We resolve the project root one level above this file so that every path
# constructed here remains independent of where the script is invoked from.
ROOT_DIR = Path(__file__).resolve().parents[1]
CONFIG_PATH = ROOT_DIR / "config" / "config.yaml"


def load_config(config_path: Path) -> dict:
    # We open the YAML configuration file and parse it into a plain Python
    # dictionary so that the rest of the pipeline can access settings without
    # needing to import yaml themselves.
    with open(config_path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def date_to_ms(date_str: str, add_one_day: bool = False) -> int:
    # We convert a date string such as "2018-01-01" to a UTC-aware Pandas Timestamp
    # and then express it in milliseconds, since that is the unit the Binance
    # klines endpoint expects for its startTime and endTime parameters.
    # The add_one_day flag lets us shift the end boundary by one day to make
    # the configured date range inclusive on both sides.
    ts = pd.Timestamp(date_str, tz="UTC")
    if add_one_day:
        ts += pd.Timedelta(days=1)
    return int(ts.timestamp() * 1000)


def fetch_klines_batch(
    symbol: str,
    interval: str,
    start_ms: int,
    end_ms: int,
    base_url: str,
    endpoint: str,
    limit: int,
) -> list:
    # We assemble the request URL and pass all pagination parameters as query
    # arguments. The 30-second timeout prevents the script from hanging
    # indefinitely if the API becomes unresponsive.
    url = f"{base_url}{endpoint}"
    params = {
        "symbol": symbol,
        "interval": interval,
        "startTime": start_ms,
        "endTime": end_ms,
        "limit": limit,
    }

    response = requests.get(url, params=params, timeout=30)
    # We raise an exception immediately if the server returns any HTTP error
    # status, which keeps error handling centralized in download_klines.
    response.raise_for_status()
    return response.json()


def download_klines(
    symbol: str,
    interval: str,
    start_ms: int,
    end_ms: int,
    base_url: str,
    endpoint: str,
    limit: int,
) -> list:
    # We iterate over the full date range in pages because Binance caps each
    # request at `limit` candles. After every batch we advance the start cursor
    # to one millisecond past the last received open time, which guarantees
    # consecutive pages share no gaps or overlapping candles.
    all_klines = []
    current_start = start_ms

    while current_start < end_ms:
        batch = fetch_klines_batch(
            symbol=symbol,
            interval=interval,
            start_ms=current_start,
            end_ms=end_ms,
            base_url=base_url,
            endpoint=endpoint,
            limit=limit,
        )

        if not batch:
            break

        all_klines.extend(batch)

        last_open_time = batch[-1][0]
        next_start = last_open_time + 1

        # We guard against an infinite loop: if the API returns the same open
        # time repeatedly without advancing the cursor, we raise immediately.
        if next_start <= current_start:
            raise RuntimeError("Pagination got stuck.")

        current_start = next_start

        # A partial page signals that we have reached the end of available
        # data for the requested symbol and interval.
        if len(batch) < limit:
            break

    return all_klines


def klines_to_dataframe(klines: list) -> pd.DataFrame:
    # We extract only the six fields we need from each raw kline tuple:
    # the open time and the four OHLC prices plus volume. Binance returns
    # twelve fields per candle; the remaining ones (quote volume, trade count,
    # taker volumes, etc.) are discarded.
    rows = [[k[0], k[1], k[2], k[3], k[4], k[5]] for k in klines]

    df = pd.DataFrame(
        rows,
        columns=["date", "open", "high", "low", "close", "volume"],
    )

    # We convert the open time from milliseconds to a tz-naive UTC datetime.
    # Removing the timezone keeps downstream date comparisons consistent
    # across all modules in the pipeline.
    df["date"] = pd.to_datetime(df["date"], unit="ms", utc=True).dt.tz_localize(None)

    # We cast all price and volume columns to float because Binance serializes
    # them as quoted decimal strings inside the JSON response.
    numeric_cols = ["open", "high", "low", "close", "volume"]
    df[numeric_cols] = df[numeric_cols].astype(float)

    # We sort by date and remove any duplicate open times that could arise from
    # overlapping pagination windows, then reset the integer index.
    df = df.sort_values("date").drop_duplicates(subset="date").reset_index(drop=True)

    return df


def filter_date_range(df: pd.DataFrame, start_date: str, end_date: str) -> pd.DataFrame:
    # We apply a closed date filter so that the resulting DataFrame only keeps
    # rows whose open time falls within [start_date, end_date], inclusive on both ends.
    start = pd.Timestamp(start_date)
    end = pd.Timestamp(end_date)
    return df[(df["date"] >= start) & (df["date"] <= end)].copy()


def validate_dataframe(df: pd.DataFrame) -> None:
    # We verify that the DataFrame is not empty, which would indicate an issue
    # with the symbol, interval, or date range specified in the configuration.
    # Sort order, duplicate removal, and numeric casting are already guaranteed
    # by klines_to_dataframe, so we do not repeat those checks here.
    if df.empty:
        raise ValueError("The DataFrame is empty.")


def save_raw_dataframe(df: pd.DataFrame, raw_dir: Path, filename: str) -> Path:
    # We create the output directory if it does not exist yet and write the
    # DataFrame as a CSV without a row index, since the index carries no
    # semantic meaning in the raw download.
    raw_dir.mkdir(parents=True, exist_ok=True)
    output_path = raw_dir / filename
    df.to_csv(output_path, index=False)
    return output_path
