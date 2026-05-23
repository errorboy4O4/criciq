"""
ingest.py - Load and clean IPL data for CricIQ
"""

import pandas as pd
import os

RAW_DIR = os.path.join(os.path.dirname(__file__), "..", "data", "raw")
PROCESSED_DIR = os.path.join(os.path.dirname(__file__), "..", "data", "processed")


def load_raw_data():
    """Load raw CSVs and return deliveries and matches DataFrames."""
    deliveries = pd.read_csv(os.path.join(RAW_DIR, "deliveries.csv"))
    matches = pd.read_csv(os.path.join(RAW_DIR, "matches.csv"))
    return deliveries, matches


def clean_data(deliveries, matches):
    """Basic cleaning: fix types, handle nulls, standardize names."""

    # --- Matches ---
    matches["date"] = pd.to_datetime(matches["date"])

    # Standardize season to just the starting year (e.g., '2007/08' -> 2007)
    matches["season_year"] = matches["season"].apply(
        lambda x: int(str(x).split("/")[0])
    )

    # Fill missing city from venue (some rows have venue but no city)
    matches["city"] = matches["city"].fillna("Unknown")

    # --- Deliveries ---
    # Fill NaN in extras_type, dismissal columns
    deliveries["extras_type"] = deliveries["extras_type"].fillna("none")
    deliveries["dismissal_kind"] = deliveries["dismissal_kind"].fillna("not_out")
    deliveries["player_dismissed"] = deliveries["player_dismissed"].fillna("none")
    deliveries["fielder"] = deliveries["fielder"].fillna("none")

    # Detect boundaries: 4s and 6s from batsman_runs
    deliveries["is_four"] = (deliveries["batsman_runs"] == 4).astype(int)
    deliveries["is_six"] = (deliveries["batsman_runs"] == 6).astype(int)

    # Merge match info into deliveries for easy access
    merged = deliveries.merge(
        matches[["id", "season_year", "date", "venue", "city"]],
        left_on="match_id",
        right_on="id",
        how="left",
    )
    merged.drop(columns=["id"], inplace=True)

    return merged, matches


def save_processed(merged, matches):
    """Save cleaned data to processed folder."""
    os.makedirs(PROCESSED_DIR, exist_ok=True)
    merged.to_csv(os.path.join(PROCESSED_DIR, "deliveries_cleaned.csv"), index=False)
    matches.to_csv(os.path.join(PROCESSED_DIR, "matches_cleaned.csv"), index=False)
    print(f"Saved: deliveries_cleaned.csv ({merged.shape[0]} rows)")
    print(f"Saved: matches_cleaned.csv ({matches.shape[0]} rows)")


if __name__ == "__main__":
    print("Loading raw data...")
    deliveries, matches = load_raw_data()

    print("Cleaning data...")
    merged, matches = clean_data(deliveries, matches)

    print("Saving processed data...")
    save_processed(merged, matches)

    # Quick sanity checks
    print("\n--- Sanity Checks ---")
    print(f"Seasons: {sorted(matches['season_year'].unique())}")
    print(f"Total matches: {matches.shape[0]}")
    print(f"Total deliveries: {merged.shape[0]}")
    print(f"Fours hit: {merged['is_four'].sum()}")
    print(f"Sixes hit: {merged['is_six'].sum()}")
    print(f"Venues: {merged['venue'].nunique()}")