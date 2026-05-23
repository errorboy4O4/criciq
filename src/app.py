"""
app.py - CricIQ Streamlit Web App
"""

import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import joblib
import os
import sys

# Add src to path for imports
sys.path.insert(0, os.path.dirname(__file__))
from agent import get_player_recommendation, get_confidence_score

# ============================================================
# CONFIG
# ============================================================

st.set_page_config(
    page_title="CricIQ - AI Fantasy Cricket Assistant",
    page_icon="🏏",
    layout="wide",
)

# Paths
BASE_DIR = os.path.join(os.path.dirname(__file__), "..")
MODEL_PATH = os.path.join(BASE_DIR, "models", "xgb_fantasy.joblib")
DATA_PATH = os.path.join(BASE_DIR, "data", "processed", "feature_matrix.csv")

# Feature columns (must match model.py)
FEATURE_COLS = [
    "rolling_avg_runs_5", "rolling_sr_5", "rolling_wickets_5",
    "rolling_economy_5", "rolling_fp_5", "recent_form_index",
    "venue_avg_runs", "venue_avg_fp", "venue_matches",
    "h2h_avg_runs", "h2h_avg_fp", "h2h_matches",
    "venue_avg_first_score",
]


# ============================================================
# LOAD DATA & MODEL
# ============================================================

@st.cache_resource
def load_model():
    return joblib.load(MODEL_PATH)


@st.cache_data
def load_data():
    df = pd.read_csv(DATA_PATH)
    df["date"] = pd.to_datetime(df["date"])
    return df


# ============================================================
# HELPER FUNCTIONS
# ============================================================

def get_teams(df):
    """Get unique team names."""
    teams = sorted(df["team"].dropna().unique())
    return teams


def get_venues(df):
    """Get unique venues."""
    venues = sorted(df["venue"].dropna().unique())
    return venues


def get_latest_player_features(df, player, venue, opposition):
    """Get the most recent features for a player, adjusted for venue/opposition."""
    player_df = df[df["player"] == player].sort_values("date")

    if player_df.empty:
        return None

    # Start with most recent match stats
    latest = player_df.iloc[-1].copy()

    # Override venue stats if player has played at this venue
    venue_matches = df[(df["player"] == player) & (df["venue"] == venue)]
    if not venue_matches.empty:
        latest["venue_avg_runs"] = venue_matches["runs"].mean()
        latest["venue_avg_fp"] = venue_matches["fantasy_points"].mean()
        latest["venue_matches"] = len(venue_matches)

    # Override h2h stats if player has played against this opposition
    h2h_matches = df[(df["player"] == player) & (df["opposition"] == opposition)]
    if not h2h_matches.empty:
        latest["h2h_avg_runs"] = h2h_matches["runs"].mean()
        latest["h2h_avg_fp"] = h2h_matches["fantasy_points"].mean()
        latest["h2h_matches"] = len(h2h_matches)

    # Override venue avg first score
    venue_scores = df[df["venue"] == venue]["venue_avg_first_score"]
    if not venue_scores.empty:
        latest["venue_avg_first_score"] = venue_scores.mean()

    return latest


def get_team_players(df, team, season=2024):
    """Get players who played for a team in the most recent season."""
    team_df = df[(df["team"] == team) & (df["season_year"] == season)]
    players = team_df["player"].unique()

    # If no players found in 2024, try 2023
    if len(players) == 0:
        team_df = df[(df["team"] == team) & (df["season_year"] == season - 1)]
        players = team_df["player"].unique()

    return sorted(players)


# ============================================================
# SCREEN 1: MATCH SETUP
# ============================================================

def show_match_setup(df):
    """Show match setup screen."""

    st.title("🏏 CricIQ")
    st.subheader("AI-Powered Fantasy Cricket Decision Assistant")
    st.markdown("---")

    teams = get_teams(df)
    venues = get_venues(df)

    col1, col2 = st.columns(2)

    with col1:
        team_a = st.selectbox("🏠 Team A (Home)", teams, index=0)

    with col2:
        # Filter out team A from team B options
        team_b_options = [t for t in teams if t != team_a]
        team_b = st.selectbox("✈️ Team B (Away)", team_b_options, index=0)

    venue = st.selectbox("🏟️ Venue", venues, index=0)

    st.markdown("---")

    if st.button("🚀 Get AI Recommendations", type="primary", use_container_width=True):
        st.session_state["match_setup"] = {
            "team_a": team_a,
            "team_b": team_b,
            "venue": venue,
        }
        st.session_state["screen"] = "recommendations"
        st.rerun()


# ============================================================
# SCREEN 2: AI RECOMMENDATIONS
# ============================================================

def show_recommendations(df, model):
    """Show AI recommendations screen."""

    setup = st.session_state["match_setup"]
    team_a = setup["team_a"]
    team_b = setup["team_b"]
    venue = setup["venue"]

    st.title("🏏 CricIQ Recommendations")
    st.subheader(f"{team_a} vs {team_b}")
    st.caption(f"📍 {venue}")
    st.markdown("---")

    if st.button("← Back to Match Setup"):
        st.session_state["screen"] = "setup"
        st.rerun()

    # Get players from both teams
    players_a = get_team_players(df, team_a)
    players_b = get_team_players(df, team_b)
    all_players = list(players_a) + list(players_b)

    if len(all_players) == 0:
        st.error("No player data found for these teams. Try different teams.")
        return

    # Predict fantasy points for all players
    predictions = []

    with st.spinner("🤖 ML Model predicting fantasy points..."):
        for player in all_players:
            team = team_a if player in players_a else team_b
            opposition = team_b if team == team_a else team_a

            features = get_latest_player_features(df, player, venue, opposition)
            if features is None:
                continue

            # Predict
            X = pd.DataFrame([features[FEATURE_COLS].values], columns=FEATURE_COLS)
            pred = model.predict(X)[0]

            # Get confidence
            confidence = get_confidence_score(
                pred,
                features.get("rolling_fp_5", 0),
                features.get("venue_avg_fp", 0),
            )

            predictions.append({
                "player": player,
                "team": team,
                "opposition": opposition,
                "predicted_fp": round(pred, 1),
                "confidence": confidence,
                "rolling_fp_5": features.get("rolling_fp_5", 0),
                "rolling_avg_runs_5": features.get("rolling_avg_runs_5", 0),
                "rolling_sr_5": features.get("rolling_sr_5", 0),
                "rolling_wickets_5": features.get("rolling_wickets_5", 0),
                "rolling_economy_5": features.get("rolling_economy_5", 0),
                "venue_avg_fp": features.get("venue_avg_fp", 0),
                "venue_avg_runs": features.get("venue_avg_runs", 0),
                "venue_matches": features.get("venue_matches", 0),
                "h2h_avg_fp": features.get("h2h_avg_fp", 0),
                "h2h_avg_runs": features.get("h2h_avg_runs", 0),
                "h2h_matches": features.get("h2h_matches", 0),
                "venue_avg_first_score": features.get("venue_avg_first_score", 0),
            })

    if not predictions:
        st.error("Could not generate predictions. Try different teams.")
        return

    # Sort by predicted points
    predictions.sort(key=lambda x: x["predicted_fp"], reverse=True)
    top_5 = predictions[:5]

    # --- Bar Chart ---
    st.subheader("📊 Top 5 Predicted Fantasy Points")
    chart_df = pd.DataFrame(top_5)
    fig = px.bar(
        chart_df,
        x="player",
        y="predicted_fp",
        color="team",
        text="predicted_fp",
        title="Predicted Fantasy Points",
        labels={"predicted_fp": "Fantasy Points", "player": "Player"},
    )
    fig.update_traces(textposition="outside")
    fig.update_layout(height=400)
    st.plotly_chart(fig, use_container_width=True)

    st.markdown("---")

    # --- AI Recommendations ---
    st.subheader("🤖 AI Analysis")

    for i, player_data in enumerate(top_5):
        with st.container():
            col1, col2 = st.columns([3, 1])

            with col1:
                st.markdown(f"### {i + 1}. {player_data['player']} ({player_data['team']})")

                # AI Recommendation
                try:
                    rec = get_player_recommendation(
                        player_data["player"],
                        player_data,
                        player_data["predicted_fp"],
                    )
                    st.markdown(f"*{rec}*")
                except Exception as e:
                    st.markdown(f"*AI analysis unavailable: {e}*")

            with col2:
                st.metric("Predicted FP", f"{player_data['predicted_fp']}")
                st.progress(player_data["confidence"] / 100)
                st.caption(f"Confidence: {player_data['confidence']}%")

            # Deep dive button
            if st.button(f"📊 Deep Dive → {player_data['player']}", key=f"dive_{i}"):
                st.session_state["deep_dive_player"] = player_data["player"]
                st.session_state["screen"] = "deep_dive"
                st.rerun()

            st.markdown("---")


# ============================================================
# SCREEN 3: PLAYER DEEP DIVE
# ============================================================

def show_deep_dive(df):
    """Show detailed player stats."""

    player = st.session_state.get("deep_dive_player", None)
    if not player:
        st.session_state["screen"] = "setup"
        st.rerun()
        return

    setup = st.session_state.get("match_setup", {})
    venue = setup.get("venue", "")

    st.title(f"🏏 {player} — Deep Dive")
    st.markdown("---")

    if st.button("← Back to Recommendations"):
        st.session_state["screen"] = "recommendations"
        st.rerun()

    player_df = df[df["player"] == player].sort_values("date")

    if player_df.empty:
        st.warning("No data found for this player.")
        return

    # --- Recent Form Chart (last 10 matches) ---
    st.subheader("📈 Recent Form (Last 10 Matches)")
    recent = player_df.tail(10).copy()
    recent["match_num"] = range(1, len(recent) + 1)
    recent["label"] = recent["date"].dt.strftime("%d %b %Y")

    fig1 = px.line(
        recent,
        x="label",
        y="fantasy_points",
        markers=True,
        title=f"{player} — Fantasy Points (Last 10 Matches)",
        labels={"fantasy_points": "Fantasy Points", "label": "Match Date"},
    )
    fig1.add_hline(
        y=recent["fantasy_points"].mean(),
        line_dash="dash",
        line_color="red",
        annotation_text=f"Avg: {recent['fantasy_points'].mean():.1f}",
    )
    fig1.update_layout(height=350)
    st.plotly_chart(fig1, use_container_width=True)

    # --- Venue Performance ---
    st.subheader(f"🏟️ Performance at {venue}")
    venue_df = player_df[player_df["venue"] == venue]

    if venue_df.empty:
        st.info(f"{player} has not played at {venue} in the dataset.")
    else:
        col1, col2, col3 = st.columns(3)
        col1.metric("Matches", len(venue_df))
        col2.metric("Avg Fantasy Pts", f"{venue_df['fantasy_points'].mean():.1f}")
        col3.metric("Avg Runs", f"{venue_df['runs'].mean():.1f}")

        fig2 = px.bar(
            venue_df,
            x=venue_df["date"].dt.strftime("%d %b %Y"),
            y="fantasy_points",
            title=f"{player} at {venue}",
            labels={"x": "Match Date", "fantasy_points": "Fantasy Points"},
        )
        fig2.update_layout(height=300)
        st.plotly_chart(fig2, use_container_width=True)

    # --- Head-to-Head vs Opposition ---
    opposition = setup.get("team_b", "") if player_df["team"].iloc[-1] == setup.get("team_a", "") else setup.get("team_a", "")
    st.subheader(f"⚔️ Head-to-Head vs {opposition}")

    h2h_df = player_df[player_df["opposition"] == opposition]

    if h2h_df.empty:
        st.info(f"No head-to-head data found vs {opposition}.")
    else:
        col1, col2, col3 = st.columns(3)
        col1.metric("Matches", len(h2h_df))
        col2.metric("Avg Fantasy Pts", f"{h2h_df['fantasy_points'].mean():.1f}")
        col3.metric("Avg Runs", f"{h2h_df['runs'].mean():.1f}")

        fig3 = px.bar(
            h2h_df,
            x=h2h_df["date"].dt.strftime("%d %b %Y"),
            y="fantasy_points",
            title=f"{player} vs {opposition}",
            labels={"x": "Match Date", "fantasy_points": "Fantasy Points"},
        )
        fig3.update_layout(height=300)
        st.plotly_chart(fig3, use_container_width=True)

    # --- Career Stats Summary ---
    st.subheader("📋 Career Summary")
    col1, col2, col3, col4 = st.columns(4)
    col1.metric("Total Matches", len(player_df))
    col2.metric("Avg Fantasy Pts", f"{player_df['fantasy_points'].mean():.1f}")
    col3.metric("Best Score", f"{player_df['fantasy_points'].max():.0f}")
    col4.metric("Avg Runs", f"{player_df['runs'].mean():.1f}")


# ============================================================
# MAIN APP
# ============================================================

def main():
    # Load data and model
    model = load_model()
    df = load_data()

    # Initialize screen state
    if "screen" not in st.session_state:
        st.session_state["screen"] = "setup"

    # Route to correct screen
    screen = st.session_state["screen"]

    if screen == "setup":
        show_match_setup(df)
    elif screen == "recommendations":
        show_recommendations(df, model)
    elif screen == "deep_dive":
        show_deep_dive(df)
    else:
        st.session_state["screen"] = "setup"
        st.rerun()


if __name__ == "__main__":
    main()