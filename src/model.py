"""
model.py - Train XGBoost model to predict fantasy points
"""

import pandas as pd
import numpy as np
import xgboost as xgb
from sklearn.model_selection import cross_val_score
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
import joblib
import os

PROCESSED_DIR = os.path.join(os.path.dirname(__file__), "..", "data", "processed")
MODEL_DIR = os.path.join(os.path.dirname(__file__), "..", "models")

# Features the model will use
FEATURE_COLS = [
    "rolling_avg_runs_5",
    "rolling_sr_5",
    "rolling_wickets_5",
    "rolling_economy_5",
    "rolling_fp_5",
    "recent_form_index",
    "venue_avg_runs",
    "venue_avg_fp",
    "venue_matches",
    "h2h_avg_runs",
    "h2h_avg_fp",
    "h2h_matches",
    "venue_avg_first_score",
]

TARGET = "fantasy_points"


def load_feature_matrix():
    """Load the feature matrix."""
    df = pd.read_csv(os.path.join(PROCESSED_DIR, "feature_matrix.csv"))
    return df


def prepare_train_test(df):
    """Split into train (2007-2023) and test (2024)."""

    train = df[df["season_year"] < 2024].copy()
    test = df[df["season_year"] == 2024].copy()

    # Drop rows where all features are 0 (first match for a player — no history)
    feature_sum = train[FEATURE_COLS].sum(axis=1)
    train = train[feature_sum > 0]

    feature_sum_test = test[FEATURE_COLS].sum(axis=1)
    test = test[feature_sum_test > 0]

    print(f"Train set: {train.shape[0]} rows (seasons 2007-2023)")
    print(f"Test set:  {test.shape[0]} rows (season 2024)")

    X_train = train[FEATURE_COLS]
    y_train = train[TARGET]
    X_test = test[FEATURE_COLS]
    y_test = test[TARGET]

    return X_train, y_train, X_test, y_test, train, test


def train_model(X_train, y_train):
    """Train XGBoost regressor."""

    model = xgb.XGBRegressor(
        n_estimators=300,
        max_depth=6,
        learning_rate=0.05,
        subsample=0.8,
        colsample_bytree=0.8,
        min_child_weight=5,
        random_state=42,
        n_jobs=-1,
    )

    model.fit(X_train, y_train, verbose=False)
    return model


def evaluate_model(model, X_train, y_train, X_test, y_test):
    """Evaluate model on train and test sets."""

    # Train metrics
    train_pred = model.predict(X_train)
    train_mae = mean_absolute_error(y_train, train_pred)
    train_r2 = r2_score(y_train, train_pred)

    # Test metrics
    test_pred = model.predict(X_test)
    test_mae = mean_absolute_error(y_test, test_pred)
    test_rmse = np.sqrt(mean_squared_error(y_test, test_pred))
    test_r2 = r2_score(y_test, test_pred)

    print("\n============================")
    print("   MODEL EVALUATION")
    print("============================")
    print(f"Train MAE:  {train_mae:.2f}")
    print(f"Train R²:   {train_r2:.3f}")
    print(f"Test MAE:   {test_mae:.2f}")
    print(f"Test RMSE:  {test_rmse:.2f}")
    print(f"Test R²:    {test_r2:.3f}")
    print("============================")

    # Context: how good is this?
    print(f"\nBaseline (predict mean): MAE = {mean_absolute_error(y_test, [y_train.mean()] * len(y_test)):.2f}")

    return test_pred


def print_feature_importance(model):
    """Print and save feature importance chart."""

    import plotly.express as px

    importance = pd.DataFrame({
        "feature": FEATURE_COLS,
        "importance": model.feature_importances_,
    }).sort_values("importance", ascending=True)

    print("\n--- Feature Importance ---")
    for _, row in importance.sort_values("importance", ascending=False).iterrows():
        bar = "█" * int(row["importance"] * 50)
        print(f"  {row['feature']:<25} {row['importance']:.3f}  {bar}")

    # Save as image
    fig = px.bar(
        importance,
        x="importance",
        y="feature",
        orientation="h",
        title="CricIQ — XGBoost Feature Importance",
        labels={"importance": "Importance Score", "feature": "Feature"},
    )
    fig.update_layout(
        height=500,
        width=800,
        font=dict(size=14),
        title_font_size=20,
        margin=dict(l=20, r=20, t=60, b=20),
    )

    os.makedirs(os.path.join(os.path.dirname(__file__), "..", "assets"), exist_ok=True)
    chart_path = os.path.join(os.path.dirname(__file__), "..", "assets", "feature_importance.png")
    fig.write_image(chart_path, scale=2)
    print(f"\nFeature importance chart saved to: {chart_path}")

    return importance


def save_model(model):
    """Save trained model."""
    os.makedirs(MODEL_DIR, exist_ok=True)
    path = os.path.join(MODEL_DIR, "xgb_fantasy.joblib")
    joblib.dump(model, path)
    print(f"\nModel saved to: {path}")


def show_sample_predictions(test, test_pred):
    """Show some actual vs predicted comparisons."""

    test = test.copy()
    test["predicted_fp"] = test_pred

    print("\n--- Sample Predictions (2024 Season) ---")
    sample = test.nlargest(15, "fantasy_points")[
        ["player", "venue", "opposition", "fantasy_points", "predicted_fp"]
    ].copy()
    sample["predicted_fp"] = sample["predicted_fp"].round(1)
    sample.columns = ["Player", "Venue", "Opposition", "Actual", "Predicted"]
    print(sample.to_string(index=False))


if __name__ == "__main__":
    print("Loading feature matrix...")
    df = load_feature_matrix()

    print("\nPreparing train/test split...")
    X_train, y_train, X_test, y_test, train, test = prepare_train_test(df)

    print("\nTraining XGBoost model...")
    model = train_model(X_train, y_train)

    test_pred = evaluate_model(model, X_train, y_train, X_test, y_test)
    importance = print_feature_importance(model)
    show_sample_predictions(test, test_pred)

    save_model(model)

    print("\nDone! Model is ready for CricIQ app.")