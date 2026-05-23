"""
features.py - Calculate Dream11 fantasy points from ball-by-ball data
"""

import pandas as pd
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
# MAIN
# ============================================================

if __name__ == "__main__":
    print("Loading cleaned data...")
    deliveries, matches = load_cleaned_data()

    print("Calculating fantasy points...")
    fantasy = calc_total_fantasy_points(deliveries)

    # Save
    out_path = os.path.join(PROCESSED_DIR, "fantasy_points.csv")
    fantasy.to_csv(out_path, index=False)
    print(f"\nSaved: fantasy_points.csv ({fantasy.shape[0]} rows)")

    # Sanity checks
    print("\n--- Top 10 Fantasy Scores Ever ---")
    top10 = fantasy.nlargest(10, "fantasy_points")[
        ["match_id", "player", "runs", "wickets", "catches",
         "batting_points", "bowling_points", "fielding_points", "fantasy_points"]
    ]
    print(top10.to_string(index=False))

    print(f"\nAverage fantasy points per player per match: {fantasy['fantasy_points'].mean():.1f}")
    print(f"Median: {fantasy['fantasy_points'].median():.1f}")
    print(f"Max: {fantasy['fantasy_points'].max():.1f}")