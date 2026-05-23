"""
features.py - Calculate Dream11 fantasy points from ball-by-ball data
"""

import pandas as pd
import numpy as np
import os

PROCESSED_DIR = os.path.join(os.path.dirname(__file__), "..", "data", "processed")


def load_cleaned_data():
    """Load cleaned deliveries and matches."""
    deliveries = pd.read_csv(os.path.join(PROCESSED_DIR, "deliveries_cleaned.csv"))
    matches = pd.read_csv(os.path.join(PROCESSED_DIR, "matches_cleaned.csv"))
    return deliveries, matches


# ============================================================
# BATTING POINTS
# ============================================================

def calc_batting_points(deliveries):
    """Calculate batting fantasy points per player per match."""

    # Group by match and batter
    bat = deliveries.groupby(["match_id", "batter"]).agg(
        runs=("batsman_runs", "sum"),
        balls_faced=("batsman_runs", "count"),
        fours=("is_four", "sum"),
        sixes=("is_six", "sum"),
    ).reset_index()

    # --- Dream11 Batting Scoring ---
    # 1 pt per run
    bat["pts_runs"] = bat["runs"]

    # Boundary bonus: +1 per four, +2 per six
    bat["pts_fours"] = bat["fours"] * 1
    bat["pts_sixes"] = bat["sixes"] * 2

    # Milestone bonuses
    bat["pts_30"] = (bat["runs"] >= 30).astype(int) * 4
    bat["pts_50"] = (bat["runs"] >= 50).astype(int) * 8
    bat["pts_100"] = (bat["runs"] >= 100).astype(int) * 16

    # Duck: -2 if dismissed for 0 runs
    # Need to check if this batter was dismissed in this match
    dismissed = deliveries[deliveries["is_wicket"] == 1][
        ["match_id", "player_dismissed"]
    ].drop_duplicates()
    dismissed["was_dismissed"] = 1

    bat = bat.merge(
        dismissed,
        left_on=["match_id", "batter"],
        right_on=["match_id", "player_dismissed"],
        how="left",
    )
    bat["was_dismissed"] = bat["was_dismissed"].fillna(0).astype(int)
    bat["pts_duck"] = ((bat["runs"] == 0) & (bat["was_dismissed"] == 1)).astype(int) * -2

    # Total batting points
    bat["batting_points"] = (
        bat["pts_runs"]
        + bat["pts_fours"]
        + bat["pts_sixes"]
        + bat["pts_30"]
        + bat["pts_50"]
        + bat["pts_100"]
        + bat["pts_duck"]
    )

    # Clean up columns
    bat = bat[["match_id", "batter", "runs", "balls_faced", "fours", "sixes",
               "was_dismissed", "batting_points"]]
    bat.rename(columns={"batter": "player"}, inplace=True)

    return bat


# ============================================================
# BOWLING POINTS
# ============================================================

def calc_bowling_points(deliveries):
    """Calculate bowling fantasy points per player per match."""

    # --- Wickets (exclude run outs) ---
    wicket_types_valid = ["caught", "bowled", "lbw", "caught and bowled",
                          "stumped", "hit wicket"]
    lbw_bowled_types = ["lbw", "bowled"]

    wickets_df = deliveries[
        (deliveries["is_wicket"] == 1)
        & (deliveries["dismissal_kind"].isin(wicket_types_valid))
    ]

    bowl_wickets = wickets_df.groupby(["match_id", "bowler"]).agg(
        wickets=("is_wicket", "sum"),
    ).reset_index()

    # LBW/Bowled bonus
    lbw_bowled_df = wickets_df[wickets_df["dismissal_kind"].isin(lbw_bowled_types)]
    lbw_bowled_count = lbw_bowled_df.groupby(["match_id", "bowler"]).agg(
        lbw_bowled=("is_wicket", "sum"),
    ).reset_index()

    # --- Economy and overs ---
    # Filter out wides and noballs for balls bowled count
    legal_balls = deliveries[
        ~deliveries["extras_type"].isin(["wides", "noballs"])
    ]

    bowl_stats = legal_balls.groupby(["match_id", "bowler"]).agg(
        runs_conceded=("total_runs", "sum"),
        balls_bowled=("total_runs", "count"),
    ).reset_index()

    bowl_stats["overs"] = bowl_stats["balls_bowled"] / 6

    # --- Maiden overs ---
    # A maiden = an over with 0 runs off legal deliveries
    over_runs = legal_balls.groupby(["match_id", "bowler", "over"]).agg(
        over_total=("total_runs", "sum"),
        balls_in_over=("total_runs", "count"),
    ).reset_index()

    maidens = over_runs[
        (over_runs["over_total"] == 0) & (over_runs["balls_in_over"] == 6)
    ]
    maiden_count = maidens.groupby(["match_id", "bowler"]).size().reset_index(
        name="maidens"
    )

    # --- Merge all bowling stats ---
    bowl = bowl_stats.copy()
    bowl = bowl.merge(bowl_wickets, on=["match_id", "bowler"], how="left")
    bowl = bowl.merge(lbw_bowled_count, on=["match_id", "bowler"], how="left")
    bowl = bowl.merge(maiden_count, on=["match_id", "bowler"], how="left")

    bowl["wickets"] = bowl["wickets"].fillna(0).astype(int)
    bowl["lbw_bowled"] = bowl["lbw_bowled"].fillna(0).astype(int)
    bowl["maidens"] = bowl["maidens"].fillna(0).astype(int)

    # --- Dream11 Bowling Scoring ---
    bowl["pts_wickets"] = bowl["wickets"] * 25
    bowl["pts_lbw_bowled"] = bowl["lbw_bowled"] * 8
    bowl["pts_maidens"] = bowl["maidens"] * 12

    # Wicket milestone bonuses
    bowl["pts_2w"] = (bowl["wickets"] >= 2).astype(int) * 4
    bowl["pts_3w"] = (bowl["wickets"] >= 3).astype(int) * 8
    bowl["pts_4w"] = (bowl["wickets"] >= 4).astype(int) * 16
    bowl["pts_5w"] = (bowl["wickets"] >= 5).astype(int) * 25

    # Total bowling points
    bowl["bowling_points"] = (
        bowl["pts_wickets"]
        + bowl["pts_lbw_bowled"]
        + bowl["pts_maidens"]
        + bowl["pts_2w"]
        + bowl["pts_3w"]
        + bowl["pts_4w"]
        + bowl["pts_5w"]
    )

    bowl = bowl[["match_id", "bowler", "wickets", "runs_conceded", "balls_bowled",
                 "overs", "maidens", "lbw_bowled", "bowling_points"]]
    bowl.rename(columns={"bowler": "player"}, inplace=True)

    return bowl


# ============================================================
# FIELDING POINTS
# ============================================================

def calc_fielding_points(deliveries):
    """Calculate fielding fantasy points per player per match."""

    wickets = deliveries[deliveries["is_wicket"] == 1].copy()

    # --- Catches (caught, caught and bowled -> fielder gets catch) ---
    catches = wickets[wickets["dismissal_kind"].isin(["caught"])].copy()
    catch_counts = catches.groupby(["match_id", "fielder"]).size().reset_index(
        name="catches"
    )

    # --- Stumpings ---
    stumpings = wickets[wickets["dismissal_kind"] == "stumped"]
    stump_counts = stumpings.groupby(["match_id", "fielder"]).size().reset_index(
        name="stumpings"
    )

    # --- Run outs ---
    runouts = wickets[wickets["dismissal_kind"] == "run out"]
    runout_counts = runouts.groupby(["match_id", "fielder"]).size().reset_index(
        name="runouts"
    )

    # Merge all fielding stats
    field = catch_counts.copy()
    field.rename(columns={"fielder": "player"}, inplace=True)

    stump_counts.rename(columns={"fielder": "player"}, inplace=True)
    runout_counts.rename(columns={"fielder": "player"}, inplace=True)

    field = field.merge(stump_counts, on=["match_id", "player"], how="outer")
    field = field.merge(runout_counts, on=["match_id", "player"], how="outer")

    field = field.fillna(0)
    field["catches"] = field["catches"].astype(int)
    field["stumpings"] = field["stumpings"].astype(int)
    field["runouts"] = field["runouts"].astype(int)

    # Dream11 fielding scoring
    field["pts_catches"] = field["catches"] * 8
    field["pts_stumpings"] = field["stumpings"] * 12
    field["pts_runouts"] = field["runouts"] * 12  # simplified: all as direct

    field["fielding_points"] = (
        field["pts_catches"] + field["pts_stumpings"] + field["pts_runouts"]
    )

    field = field[["match_id", "player", "catches", "stumpings", "runouts",
                   "fielding_points"]]

    return field


# ============================================================
# COMBINE ALL POINTS
# ============================================================

def calc_total_fantasy_points(deliveries):
    """Combine batting + bowling + fielding into total fantasy points."""

    print("Calculating batting points...")
    bat = calc_batting_points(deliveries)

    print("Calculating bowling points...")
    bowl = calc_bowling_points(deliveries)

    print("Calculating fielding points...")
    field = calc_fielding_points(deliveries)

    # Full outer merge on match_id + player
    total = bat.merge(bowl, on=["match_id", "player"], how="outer")
    total = total.merge(field, on=["match_id", "player"], how="outer")

    # Fill NaN with 0 for players who only batted or only bowled
    total = total.fillna(0)

    # Total fantasy points
    total["fantasy_points"] = (
        total["batting_points"] + total["bowling_points"] + total["fielding_points"]
    )

    # Sort by match and fantasy points
    total = total.sort_values(["match_id", "fantasy_points"], ascending=[True, False])

    return total

# ============================================================
# FEATURE ENGINEERING
# ============================================================

def add_match_info(fantasy, deliveries, matches):
    """Add venue, teams, date, season to each player-match row."""

    # Get match-level info
    match_info = deliveries.groupby("match_id").agg(
        venue=("venue", "first"),
        city=("city", "first"),
        season_year=("season_year", "first"),
        date=("date", "first"),
    ).reset_index()

    # Get the two teams in each match
    match_teams = deliveries.groupby("match_id")["batting_team"].unique().reset_index()
    match_teams.columns = ["match_id", "teams"]

    # Get which team each player batted for
    bat_teams = deliveries.groupby(["match_id", "batter"])["batting_team"].first().reset_index()
    bat_teams.columns = ["match_id", "player", "team"]

    # Get which team each player bowled for (bowler's team = bowling_team)
    bowl_teams = deliveries.groupby(["match_id", "bowler"])["bowling_team"].first().reset_index()
    bowl_teams.columns = ["match_id", "player", "team_from_bowling"]

    # Merge match info
    fantasy = fantasy.merge(match_info, on="match_id", how="left")

    # Merge batting team
    fantasy = fantasy.merge(bat_teams, on=["match_id", "player"], how="left")

    # For players who only bowled, fill team from bowling data
    fantasy = fantasy.merge(bowl_teams, on=["match_id", "player"], how="left")
    fantasy["team"] = fantasy["team"].fillna(fantasy["team_from_bowling"])
    fantasy.drop(columns=["team_from_bowling"], inplace=True)

    # Get opposition: the other team in this match
    fantasy = fantasy.merge(match_teams, on="match_id", how="left")
    fantasy["opposition"] = fantasy.apply(
        lambda row: [t for t in row["teams"] if t != row["team"]][0]
        if len(row["teams"]) == 2 and row["team"] in list(row["teams"])
        else "Unknown",
        axis=1,
    )
    fantasy.drop(columns=["teams"], inplace=True)

    # Sort by player and date for rolling calculations
    fantasy["date"] = pd.to_datetime(fantasy["date"])
    fantasy = fantasy.sort_values(["player", "date", "match_id"]).reset_index(drop=True)

    return fantasy


def add_rolling_features(fantasy):
    """Add rolling window features: last 5 match form."""

    fantasy = fantasy.sort_values(["player", "date"]).reset_index(drop=True)

    # Group by player and calculate rolling stats
    grouped = fantasy.groupby("player")

    # Rolling average runs (last 5 matches)
    fantasy["rolling_avg_runs_5"] = grouped["runs"].transform(
        lambda x: x.shift(1).rolling(5, min_periods=1).mean()
    )

    # Rolling strike rate (last 5 matches)
    fantasy["rolling_balls_5"] = grouped["balls_faced"].transform(
        lambda x: x.shift(1).rolling(5, min_periods=1).sum()
    )
    fantasy["rolling_runs_for_sr_5"] = grouped["runs"].transform(
        lambda x: x.shift(1).rolling(5, min_periods=1).sum()
    )
    fantasy["rolling_sr_5"] = (
        fantasy["rolling_runs_for_sr_5"] / fantasy["rolling_balls_5"].replace(0, 1) * 100
    )

    # Rolling wickets (last 5 matches)
    fantasy["rolling_wickets_5"] = grouped["wickets"].transform(
        lambda x: x.shift(1).rolling(5, min_periods=1).mean()
    )

    # Rolling economy (last 5 matches)
    fantasy["rolling_economy_5"] = grouped["runs_conceded"].transform(
        lambda x: x.shift(1).rolling(5, min_periods=1).mean()
    ) / grouped["overs"].transform(
        lambda x: x.shift(1).rolling(5, min_periods=1).mean()
    ).replace(0, 1)

    # Rolling fantasy points (last 5 matches) — recent form index
    fantasy["rolling_fp_5"] = grouped["fantasy_points"].transform(
        lambda x: x.shift(1).rolling(5, min_periods=1).mean()
    )

    # Weighted recent form: match 1 (most recent) = weight 5, match 5 = weight 1
    def weighted_form(series):
        shifted = series.shift(1)
        result = []
        for i in range(len(shifted)):
            window = shifted.iloc[max(0, i - 4):i + 1].dropna()
            if len(window) == 0:
                result.append(0)
            else:
                weights = list(range(1, len(window) + 1))
                result.append(sum(w * v for w, v in zip(weights, window)) / sum(weights))
        return result

    fantasy["recent_form_index"] = grouped["fantasy_points"].transform(weighted_form)

    # Clean up temp columns
    fantasy.drop(columns=["rolling_balls_5", "rolling_runs_for_sr_5"], inplace=True)

    return fantasy


def add_venue_features(fantasy):
    """Add venue-specific historical stats per player."""

    fantasy = fantasy.sort_values(["player", "date"]).reset_index(drop=True)

    # Career average runs at this venue (using only past data)
    venue_stats = []
    for _, group in fantasy.groupby(["player", "venue"]):
        group = group.sort_values("date")
        group["venue_avg_runs"] = group["runs"].shift(1).expanding().mean()
        group["venue_avg_fp"] = group["fantasy_points"].shift(1).expanding().mean()
        group["venue_matches"] = group["match_id"].shift(1).expanding().count()
        venue_stats.append(group[["match_id", "player", "venue_avg_runs",
                                   "venue_avg_fp", "venue_matches"]])

    venue_df = pd.concat(venue_stats, ignore_index=True)
    fantasy = fantasy.merge(venue_df, on=["match_id", "player"], how="left")

    # Fill NaN (first time at venue) with overall player average
    player_avg = fantasy.groupby("player")["runs"].transform(
        lambda x: x.shift(1).expanding().mean()
    )
    fantasy["venue_avg_runs"] = fantasy["venue_avg_runs"].fillna(player_avg)
    fantasy["venue_avg_fp"] = fantasy["venue_avg_fp"].fillna(
        fantasy.groupby("player")["fantasy_points"].transform(
            lambda x: x.shift(1).expanding().mean()
        )
    )
    fantasy["venue_matches"] = fantasy["venue_matches"].fillna(0)

    return fantasy


def add_h2h_features(fantasy):
    """Add head-to-head stats vs opposition team."""

    fantasy = fantasy.sort_values(["player", "date"]).reset_index(drop=True)

    h2h_stats = []
    for _, group in fantasy.groupby(["player", "opposition"]):
        group = group.sort_values("date")
        group["h2h_avg_runs"] = group["runs"].shift(1).expanding().mean()
        group["h2h_avg_fp"] = group["fantasy_points"].shift(1).expanding().mean()
        group["h2h_matches"] = group["match_id"].shift(1).expanding().count()
        h2h_stats.append(group[["match_id", "player", "h2h_avg_runs",
                                 "h2h_avg_fp", "h2h_matches"]])

    h2h_df = pd.concat(h2h_stats, ignore_index=True)
    fantasy = fantasy.merge(h2h_df, on=["match_id", "player"], how="left")

    # Fill NaN with overall player average
    fantasy["h2h_avg_runs"] = fantasy["h2h_avg_runs"].fillna(
        fantasy.groupby("player")["runs"].transform(
            lambda x: x.shift(1).expanding().mean()
        )
    )
    fantasy["h2h_avg_fp"] = fantasy["h2h_avg_fp"].fillna(
        fantasy.groupby("player")["fantasy_points"].transform(
            lambda x: x.shift(1).expanding().mean()
        )
    )
    fantasy["h2h_matches"] = fantasy["h2h_matches"].fillna(0)

    return fantasy


def add_venue_context(fantasy, matches):
    """Add venue-level context: avg first innings score, pitch type."""

    # Historical average first innings score at each venue
    first_innings = fantasy[fantasy["match_id"].isin(matches["id"])].copy()

    venue_batting_avg = matches.groupby("venue")["target_runs"].mean().reset_index()
    venue_batting_avg.columns = ["venue", "venue_avg_first_score"]

    fantasy = fantasy.merge(venue_batting_avg, on="venue", how="left")
    fantasy["venue_avg_first_score"] = fantasy["venue_avg_first_score"].fillna(
        fantasy["venue_avg_first_score"].median()
    )

    return fantasy


def build_feature_matrix(fantasy, deliveries, matches):
    """Run all feature engineering steps and return final feature matrix."""

    print("  Adding match info...")
    fantasy = add_match_info(fantasy, deliveries, matches)

    print("  Adding rolling features...")
    fantasy = add_rolling_features(fantasy)

    print("  Adding venue features...")
    fantasy = add_venue_features(fantasy)

    print("  Adding head-to-head features...")
    fantasy = add_h2h_features(fantasy)

    print("  Adding venue context...")
    fantasy = add_venue_context(fantasy, matches)

    # Fill any remaining NaN with 0
    feature_cols = [
        "rolling_avg_runs_5", "rolling_sr_5", "rolling_wickets_5",
        "rolling_economy_5", "rolling_fp_5", "recent_form_index",
        "venue_avg_runs", "venue_avg_fp", "venue_matches",
        "h2h_avg_runs", "h2h_avg_fp", "h2h_matches",
        "venue_avg_first_score",
    ]
    fantasy[feature_cols] = fantasy[feature_cols].fillna(0)

    return fantasy


# ============================================================
# MAIN
# ============================================================

if __name__ == "__main__":
    print("Loading cleaned data...")
    deliveries, matches = load_cleaned_data()

    print("Calculating fantasy points...")
    fantasy = calc_total_fantasy_points(deliveries)

    print("\nBuilding feature matrix...")
    fantasy = build_feature_matrix(fantasy, deliveries, matches)

    # Save
    out_path = os.path.join(PROCESSED_DIR, "feature_matrix.csv")
    fantasy.to_csv(out_path, index=False)
    print(f"\nSaved: feature_matrix.csv ({fantasy.shape[0]} rows, {fantasy.shape[1]} columns)")

    # Show sample
    feature_cols = [
        "player", "fantasy_points", "rolling_avg_runs_5", "rolling_sr_5",
        "rolling_fp_5", "recent_form_index", "venue_avg_runs", "h2h_avg_runs",
    ]
    print("\n--- Sample Feature Matrix (Top 5 rows) ---")
    print(fantasy[feature_cols].head(10).to_string(index=False))

    print(f"\n--- Feature Stats ---")
    print(fantasy[feature_cols[1:]].describe().round(1).to_string())