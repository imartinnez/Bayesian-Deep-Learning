from pathlib import Path
import sys

ROOT_DIR = Path(__file__).resolve().parents[1]
if str(ROOT_DIR) not in sys.path:
    sys.path.append(str(ROOT_DIR))

from data.download import load_config, date_to_ms, download_klines
from data.download import klines_to_dataframe, filter_date_range, save_raw_dataframe

CONFIG_PATH = ROOT_DIR / "config" / "config.yaml"


if __name__ == "__main__":
    cfg = load_config(CONFIG_PATH)

    symbol = cfg["data"]["symbol"]
    interval = cfg["data"]["interval"]
    start_date = cfg["data"]["start_date"]
    end_date = cfg["data"]["end_date"]

    base_url = cfg["api"]["base_url"]
    endpoint = cfg["api"]["klines_endpoint"]
    limit = cfg["api"]["limit"]

    raw_dir = ROOT_DIR / cfg["paths"]["raw"]
    raw_filename = cfg["paths"]["raw_filename"]

    start_ms = date_to_ms(start_date)
    end_ms = date_to_ms(end_date, add_one_day=True)

    klines = download_klines(
        symbol=symbol,
        interval=interval,
        start_ms=start_ms,
        end_ms=end_ms,
        base_url=base_url,
        endpoint=endpoint,
        limit=limit,
    )

    df = klines_to_dataframe(klines)
    df = filter_date_range(df, start_date, end_date)

    output_path = save_raw_dataframe(df, raw_dir, raw_filename)

    print(f"Raw data saved to: {output_path}")
    print(f"Number of rows: {len(df)}")
    print(f"Date range: {df['date'].min().date()} to {df['date'].max().date()}")
    print("First 5 rows:")
    print(df.head())
    print("Last 5 rows:")
    print(df.tail())