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

    # Standardize team and venue names
    merged, matches = standardize_names(merged, matches)

    return merged, matches

def standardize_names(merged, matches):
    """Fix inconsistent team and venue names across seasons."""

    # Team name mapping — map old/variant names to current names
    team_map = {
        "Delhi Daredevils": "Delhi Capitals",
        "Kings XI Punjab": "Punjab Kings",
        "Rising Pune Supergiant": "Rising Pune Supergiants",
        "Rising Pune Supergiants": "Rising Pune Supergiants",
        "Royal Challengers Bengaluru": "Royal Challengers Bangalore",
        "Deccan Chargers": "Deccan Chargers",  # defunct, keep as-is
        "Gujarat Lions": "Gujarat Lions",  # defunct, keep as-is
        "Kochi Tuskers Kerala": "Kochi Tuskers Kerala",  # defunct
        "Pune Warriors": "Pune Warriors",  # defunct
    }

    # Apply to deliveries
    for col in ["batting_team", "bowling_team"]:
        merged[col] = merged[col].replace(team_map)

    # Apply to matches
    for col in ["team1", "team2", "toss_winner", "winner"]:
        matches[col] = matches[col].replace(team_map)

    # Venue name standardization
    venue_map = {
        # Delhi
        "Feroz Shah Kotla": "Arun Jaitley Stadium, Delhi",
        "Arun Jaitley Stadium": "Arun Jaitley Stadium, Delhi",
        # Bengaluru
        "M Chinnaswamy Stadium": "M Chinnaswamy Stadium, Bengaluru",
        "M.Chinnaswamy Stadium": "M Chinnaswamy Stadium, Bengaluru",
        # Chennai
        "MA Chidambaram Stadium, Chepauk": "MA Chidambaram Stadium, Chepauk, Chennai",
        "MA Chidambaram Stadium": "MA Chidambaram Stadium, Chepauk, Chennai",
        # Kolkata
        "Eden Gardens": "Eden Gardens, Kolkata",
        # Mumbai
        "Wankhede Stadium": "Wankhede Stadium, Mumbai",
        "Brabourne Stadium": "Brabourne Stadium, Mumbai",
        "Dr DY Patil Sports Academy": "Dr DY Patil Sports Academy, Mumbai",
        # Jaipur
        "Sawai Mansingh Stadium": "Sawai Mansingh Stadium, Jaipur",
        # Mohali
        "Punjab Cricket Association IS Bindra Stadium, Mohali": "PCA Stadium, Mohali",
        "Punjab Cricket Association Stadium, Mohali": "PCA Stadium, Mohali",
        "Punjab Cricket Association IS Bindra Stadium": "PCA Stadium, Mohali",
        "Punjab Cricket Association IS Bindra Stadium, Mohali, Chandigarh": "PCA Stadium, Mohali",
        # Hyderabad
        "Rajiv Gandhi International Stadium, Uppal": "Rajiv Gandhi Stadium, Hyderabad",
        "Rajiv Gandhi International Stadium": "Rajiv Gandhi Stadium, Hyderabad",
        "Rajiv Gandhi International Stadium, Uppal, Hyderabad": "Rajiv Gandhi Stadium, Hyderabad",
        # Ahmedabad (Sardar Patel renamed to Narendra Modi)
        "Sardar Patel Stadium, Motera": "Narendra Modi Stadium, Ahmedabad",
        "Sardar Patel Stadium": "Narendra Modi Stadium, Ahmedabad",
        # Visakhapatnam
        "Dr. Y.S. Rajasekhara Reddy ACA-VDCA Cricket Stadium": "ACA-VDCA Stadium, Visakhapatnam",
        "Dr. Y.S. Rajasekhara Reddy ACA-VDCA Cricket Stadium, Visakhapatnam": "ACA-VDCA Stadium, Visakhapatnam",
        # Dharamsala
        "Himachal Pradesh Cricket Association Stadium": "HPCA Stadium, Dharamsala",
        "Himachal Pradesh Cricket Association Stadium, Dharamsala": "HPCA Stadium, Dharamsala",
        # Pune
        "Maharashtra Cricket Association Stadium": "MCA Stadium, Pune",
        "Maharashtra Cricket Association Stadium, Pune": "MCA Stadium, Pune",
        # Others with missing city
        "Nehru Stadium": "Nehru Stadium, Chennai",
        "Green Park": "Green Park, Kanpur",
        "Holkar Cricket Stadium": "Holkar Cricket Stadium, Indore",
        "Barabati Stadium": "Barabati Stadium, Cuttack",
        "Saurashtra Cricket Association Stadium": "SCA Stadium, Rajkot",
        "Subrata Roy Sahara Stadium": "MCA Stadium, Pune",
        "Vidarbha Cricket Association Stadium, Jamtha": "VCA Stadium, Nagpur",
    }

    merged["venue"] = merged["venue"].replace(venue_map)
    matches["venue"] = matches["venue"].replace(venue_map)

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