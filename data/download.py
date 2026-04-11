from pathlib import Path
import requests
import pandas as pd
import yaml

ROOT_DIR = Path(__file__).resolve().parents[1]
CONFIG_PATH = ROOT_DIR / "config" / "config.yaml"


def load_config(config_path: Path) -> dict:
    with open(config_path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def date_to_ms(date_str: str, add_one_day: bool = False) -> int:
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
    url = f"{base_url}{endpoint}"
    params = {
        "symbol": symbol,
        "interval": interval,
        "startTime": start_ms,
        "endTime": end_ms,
        "limit": limit,
    }

    response = requests.get(url, params=params, timeout=30)
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

        if next_start <= current_start:
            raise RuntimeError("Pagination got stuck.")

        current_start = next_start

        if len(batch) < limit:
            break

    return all_klines


def klines_to_dataframe(klines: list) -> pd.DataFrame:
    rows = [[k[0], k[1], k[2], k[3], k[4], k[5]] for k in klines]

    df = pd.DataFrame(
        rows,
        columns=["date", "open", "high", "low", "close", "volume"],
    )

    df["date"] = pd.to_datetime(df["date"], unit="ms", utc=True).dt.tz_localize(None)

    numeric_cols = ["open", "high", "low", "close", "volume"]
    df[numeric_cols] = df[numeric_cols].astype(float)

    df = df.sort_values("date").drop_duplicates(subset="date").reset_index(drop=True)

    return df


def filter_date_range(df: pd.DataFrame, start_date: str, end_date: str) -> pd.DataFrame:
    start = pd.Timestamp(start_date)
    end = pd.Timestamp(end_date)
    return df[(df["date"] >= start) & (df["date"] <= end)].copy()


def validate_dataframe(df: pd.DataFrame) -> None:
    if df.empty:
        raise ValueError("The DataFrame is empty.")

    if df["date"].duplicated().any():
        raise ValueError("Duplicate dates were found.")

    if not df["date"].is_monotonic_increasing:
        raise ValueError("The 'date' column is not sorted in ascending order.")

    numeric_cols = ["open", "high", "low", "close", "volume"]
    for col in numeric_cols:
        if not pd.api.types.is_numeric_dtype(df[col]):
            raise TypeError(f"The column {col} is not numeric.")


def save_raw_dataframe(df: pd.DataFrame, raw_dir: Path, filename: str) -> Path:
    raw_dir.mkdir(parents=True, exist_ok=True)
    output_path = raw_dir / filename
    df.to_csv(output_path, index=False)
    return output_path