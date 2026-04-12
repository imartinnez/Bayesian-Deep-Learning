# @author: Íñigo Martínez Jiménez
# Orchestration script for the first stage of the data pipeline.
# We read all settings from config.yaml, download the full BTCUSDT daily kline
# history from the Binance REST API, and persist it as a raw CSV file.

from pathlib import Path
import sys

# We add the project root to sys.path so that the data package can be imported
# regardless of the working directory from which this script is executed.
ROOT_DIR = Path(__file__).resolve().parents[1]
if str(ROOT_DIR) not in sys.path:
    sys.path.append(str(ROOT_DIR))

from src.data.download import load_config, date_to_ms, download_klines
from src.data.download import klines_to_dataframe, filter_date_range, save_raw_dataframe

CONFIG_PATH = ROOT_DIR / "config" / "config.yaml"


if __name__ == "__main__":
    # We load the configuration and extract all parameters needed for the download.
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

    # We convert the string dates to millisecond timestamps required by Binance.
    # The end date gets shifted by one day so that the last candle of end_date
    # is included in the paginated download.
    start_ms = date_to_ms(start_date)
    end_ms = date_to_ms(end_date, add_one_day=True)

    # We fetch all kline pages, convert them to a clean DataFrame, and apply a
    # final date filter to clip the result to the exact configured range.
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

    # We print a brief summary so we can visually confirm row count and date
    # boundaries without having to open the output file manually.
    print(f"Raw data saved to: {output_path}")
    print(f"Number of rows: {len(df)}")
    print(f"Date range: {df['date'].min().date()} to {df['date'].max().date()}")
    print("First 5 rows:")
    print(df.head())
    print("Last 5 rows:")
    print(df.tail())
