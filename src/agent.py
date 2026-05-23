"""
agent.py - Claude API integration for player recommendations
"""

import anthropic
import os


def get_player_recommendation(player_name, stats, predicted_points):
    """
    Generate a 3-sentence AI recommendation for a player.
    
    Args:
        player_name: str - player's name
        stats: dict - player's stats (rolling form, venue, h2h, etc.)
        predicted_points: float - ML model's predicted fantasy points
    
    Returns:
        str - 3-sentence recommendation
    """

    # Build a data-rich prompt so Claude cites real numbers
    prompt = f"""You are a fantasy cricket analyst for Dream11 IPL.
    
Given the following REAL stats for {player_name}, write EXACTLY 3 sentences:
- Sentence 1: The data case for picking this player (cite specific numbers)
- Sentence 2: One risk or caveat (cite a specific stat)
- Sentence 3: Final verdict — classify as: STRONG PICK / VALUE PICK / RISKY PICK / AVOID

Player: {player_name}
Predicted Fantasy Points: {predicted_points:.1f}
Recent Form (avg fantasy pts last 5 matches): {stats.get('rolling_fp_5', 0):.1f}
Rolling Avg Runs (last 5): {stats.get('rolling_avg_runs_5', 0):.1f}
Rolling Strike Rate (last 5): {stats.get('rolling_sr_5', 0):.1f}
Rolling Wickets (last 5 avg): {stats.get('rolling_wickets_5', 0):.2f}
Rolling Economy (last 5): {stats.get('rolling_economy_5', 0):.1f}
Venue Avg Fantasy Points: {stats.get('venue_avg_fp', 0):.1f}
Venue Avg Runs: {stats.get('venue_avg_runs', 0):.1f}
Venue Matches Played: {stats.get('venue_matches', 0):.0f}
H2H Avg Fantasy Points vs Opposition: {stats.get('h2h_avg_fp', 0):.1f}
H2H Avg Runs vs Opposition: {stats.get('h2h_avg_runs', 0):.1f}
H2H Matches vs Opposition: {stats.get('h2h_matches', 0):.0f}
Venue Avg First Innings Score: {stats.get('venue_avg_first_score', 0):.0f}

Rules:
- You MUST cite actual numbers from the stats above
- Do NOT use generic advice like "he's a good player"
- Keep it to EXACTLY 3 sentences, no more
- Be specific and concise
"""

    client = anthropic.Anthropic()  # reads ANTHROPIC_API_KEY from env

    message = client.messages.create(
        model="claude-sonnet-4-6",
        max_tokens=300,
        messages=[
            {"role": "user", "content": prompt}
        ],
    )

    return message.content[0].text


def get_match_summary(top_players):
    """
    Generate a brief match prediction summary.
    
    Args:
        top_players: list of dicts with player name, team, predicted points
    
    Returns:
        str - 2-3 sentence match overview
    """

    player_lines = "\n".join(
        f"  {p['player']} ({p['team']}): {p['predicted_fp']:.1f} predicted pts"
        for p in top_players
    )

    prompt = f"""You are a fantasy cricket analyst. Given these top predicted 
performers for an upcoming IPL match, write a 2-sentence match overview 
for fantasy team selection. Be specific about which players to prioritize.

Top Predicted Performers:
{player_lines}

Rules:
- Exactly 2 sentences
- Mention specific player names
- Focus on fantasy team building advice
"""

    client = anthropic.Anthropic()

    message = client.messages.create(
        model="claude-sonnet-4-6",
        max_tokens=200,
        messages=[
            {"role": "user", "content": prompt}
        ],
    )

    return message.content[0].text


def get_confidence_score(predicted_points, rolling_fp_5, venue_avg_fp):
    """
    Calculate a confidence score (0-100) for the prediction.
    Higher when model prediction aligns with historical averages.
    """

    if rolling_fp_5 == 0 and venue_avg_fp == 0:
        return 30  # low confidence — no history

    # How close is prediction to recent form?
    if rolling_fp_5 > 0:
        form_diff = abs(predicted_points - rolling_fp_5) / max(rolling_fp_5, 1)
    else:
        form_diff = 1.0

    # How close is prediction to venue average?
    if venue_avg_fp > 0:
        venue_diff = abs(predicted_points - venue_avg_fp) / max(venue_avg_fp, 1)
    else:
        venue_diff = 1.0

    # Score: lower difference = higher confidence
    avg_diff = (form_diff + venue_diff) / 2
    confidence = max(20, min(95, int(100 - avg_diff * 60)))

    return confidence


# ============================================================
# TEST
# ============================================================

if __name__ == "__main__":
    # Quick test with sample data
    test_stats = {
        "rolling_fp_5": 55.2,
        "rolling_avg_runs_5": 35.4,
        "rolling_sr_5": 145.6,
        "rolling_wickets_5": 0.4,
        "rolling_economy_5": 8.2,
        "venue_avg_fp": 48.0,
        "venue_avg_runs": 30.5,
        "venue_matches": 8,
        "h2h_avg_fp": 42.3,
        "h2h_avg_runs": 28.1,
        "h2h_matches": 12,
        "venue_avg_first_score": 175,
    }

    print("Testing Claude API connection...")
    print("=" * 50)

    try:
        rec = get_player_recommendation("V Kohli", test_stats, 52.3)
        print(f"Recommendation for V Kohli:\n{rec}")
        print("=" * 50)

        confidence = get_confidence_score(52.3, 55.2, 48.0)
        print(f"Confidence Score: {confidence}%")
        print("\nClaude API is working!")

    except anthropic.AuthenticationError:
        print("ERROR: ANTHROPIC_API_KEY not set or invalid!")
        print("Set it with: $env:ANTHROPIC_API_KEY='your-key-here'")

    except Exception as e:
        print(f"ERROR: {e}")