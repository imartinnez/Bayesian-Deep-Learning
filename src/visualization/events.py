# @author: Inigo Martinez Jimenez
# Crypto stress-event registry. Every figure that needs to annotate the
# timeline with real-world context pulls its markers from this single list,
# which keeps names, dates, and spans consistent across the thesis.

from __future__ import annotations

from dataclasses import dataclass, field

import pandas as pd


@dataclass(frozen=True)
class StressEvent:
    """A named stress period in the crypto market."""
    key: str
    name: str
    start: str
    end: str
    label: str
    description: str = ""

    @property
    def start_ts(self) -> pd.Timestamp:
        return pd.Timestamp(self.start)

    @property
    def end_ts(self) -> pd.Timestamp:
        return pd.Timestamp(self.end)

    @property
    def mid_ts(self) -> pd.Timestamp:
        return self.start_ts + (self.end_ts - self.start_ts) / 2

    def touches(self, start: pd.Timestamp, end: pd.Timestamp) -> bool:
        # Returns True if the event overlaps the given inclusive date range.
        return (self.end_ts >= pd.Timestamp(start)) and (self.start_ts <= pd.Timestamp(end))


# Canonical ordered list of stress events covering the dataset range
# 2018-2024. We only include events with clear market impact on Bitcoin so
# every annotation adds real narrative weight to a thesis figure.
STRESS_EVENTS: list[StressEvent] = [
    StressEvent(
        key="covid_crash",
        name="COVID-19 Market Shock",
        start="2020-03-08",
        end="2020-03-20",
        label="COVID-19 Shock",
        description="Global risk-off selloff, BTC prints multi-year low.",
    ),
    StressEvent(
        key="china_ban_2021",
        name="China Crypto Ban / BTC Flash Crash",
        start="2021-05-12",
        end="2021-05-23",
        label="May 2021 Flash Crash",
        description="Mining ban in China triggers a >30% weekly drawdown.",
    ),
    StressEvent(
        key="luna_ust",
        name="LUNA / UST Collapse",
        start="2022-05-08",
        end="2022-05-18",
        label="LUNA / UST Collapse",
        description="Terra stablecoin depeg wipes ~$40B in days.",
    ),
    StressEvent(
        key="three_arrows",
        name="Three Arrows Capital / Celsius",
        start="2022-06-10",
        end="2022-06-22",
        label="3AC / Celsius",
        description="Crypto credit crunch and major lender failures.",
    ),
    StressEvent(
        key="ftx",
        name="FTX / Alameda Collapse",
        start="2022-11-06",
        end="2022-11-16",
        label="FTX Collapse",
        description="Run on FTX exchange and ensuing contagion.",
    ),
    StressEvent(
        key="svb_banking_stress",
        name="SVB / USDC Depeg / Banking Stress",
        start="2023-03-08",
        end="2023-03-17",
        label="SVB / USDC Depeg",
        description="US regional banking stress and USDC temporary depeg.",
    ),
    StressEvent(
        key="etf_launch",
        name="US Spot BTC ETF Launch",
        start="2024-01-08",
        end="2024-01-15",
        label="Spot ETF Launch",
        description="US spot Bitcoin ETFs approved and begin trading.",
    ),
    StressEvent(
        key="aug_2024_unwind",
        name="August 2024 Carry Unwind",
        start="2024-08-02",
        end="2024-08-08",
        label="Aug 2024 Carry Unwind",
        description="Yen carry-trade unwind drives a cross-asset risk-off.",
    ),
]


def events_in_range(start, end) -> list[StressEvent]:
    # Filter the canonical event list to those overlapping the given range,
    # preserving the original order for deterministic plotting.
    start_ts = pd.Timestamp(start)
    end_ts = pd.Timestamp(end)
    return [e for e in STRESS_EVENTS if e.touches(start_ts, end_ts)]


def get_event(key: str) -> StressEvent:
    # Lookup helper for scripts that want to zoom into a single event window.
    for event in STRESS_EVENTS:
        if event.key == key:
            return event
    raise KeyError(f"Unknown stress event '{key}'. Known: {[e.key for e in STRESS_EVENTS]}")
