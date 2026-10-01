# flask web app that compares our predicted mvp candidates to the real mvp race
# run train_model.py first so the saved models exist

import os

import joblib
from flask import Flask, render_template, request

# reuse the same settings and data cleaning we used for training
from train_model import load_data, FEATURES, MODEL_DIR, TRAIN_END_YEAR

app = Flask(__name__)

# the saved model files for each model name
MODEL_FILES = {
    "Random Forest": "random_forest.joblib",
    "Gradient Boosting": "gradient_boosting.joblib",
}

# how many seasons the multi-year page can show (0 means all seasons)
YEAR_OPTIONS = [5, 10, 15, 20, 0]

# load the data and the trained models once when the app starts
df = load_data()
models = {}
for name, filename in MODEL_FILES.items():
    models[name] = joblib.load(os.path.join(MODEL_DIR, filename))

# add a predicted vote share column for each model
for name, model in models.items():
    df[name] = model.predict(df[FEATURES])

# list of all seasons, newest first
SEASONS = sorted([int(s) for s in df["Season"].unique()], reverse=True)

# seasons that have real mvp voting results, newest first
voted_seasons = df[df["Share"] > 0]["Season"].unique()
RESULT_SEASONS = sorted([int(s) for s in voted_seasons], reverse=True)


# read the model choice from the url (or use random forest)
def get_model_name():
    model_name = request.args.get("model", default="Random Forest")
    if model_name not in models:
        model_name = "Random Forest"
    return model_name


# turn the top 5 rows of a table into simple dictionaries for the html page
def top_five(season_df, sort_column):
    top = season_df.sort_values(sort_column, ascending=False).head(5)
    rows = []
    for _, row in top.iterrows():
        rows.append({
            "player": row["Player"],
            "team": row["Team"],
            "pts": row["PTS"],
            "trb": row["TRB"],
            "ast": row["AST"],
            "share": round(row[sort_column] * 100, 1),
        })
    return rows


# compare one list against the other list of names
# exact = same player in the same spot, close = in the other top 5 but a different spot
# miss = not in the other top 5 at all
def add_status(rows, other_names):
    for i, row in enumerate(rows):
        rank = i + 1
        if row["player"] in other_names:
            row["other_rank"] = other_names.index(row["player"]) + 1
        else:
            row["other_rank"] = None

        if row["other_rank"] == rank:
            row["status"] = "exact"
        elif row["other_rank"] is not None:
            row["status"] = "close"
        else:
            row["status"] = "miss"


# build the predicted vs actual comparison for one season
def compare_season(season, model_name):
    season_df = df[df["Season"] == season]
    predicted = top_five(season_df, model_name)

    # actual top 5 from the real voting (only players who got votes)
    actual = top_five(season_df[season_df["Share"] > 0], "Share")
    has_actual = len(actual) > 0

    result = {
        "season": season,
        "predicted": predicted,
        "actual": actual,
        "has_actual": has_actual,
        "was_trained": season <= TRAIN_END_YEAR,
        "total": len(actual),
        "mvp_correct": False,
        "overlap": 0,
        "exact": 0,
    }

    # without real results there is nothing to compare against
    if not has_actual:
        for row in predicted:
            row["status"] = "none"
            row["other_rank"] = None
        return result

    add_status(predicted, [a["player"] for a in actual])
    add_status(actual, [p["player"] for p in predicted])

    # did the model pick the real mvp?
    result["mvp_correct"] = predicted[0]["player"] == actual[0]["player"]
    # how many predicted players finished in the actual top 5?
    result["overlap"] = sum(1 for p in predicted if p["status"] != "miss")
    # how many predicted players are in the exact right spot?
    result["exact"] = sum(1 for p in predicted if p["status"] == "exact")
    return result


# add up the accuracy numbers over many seasons
# seasons the model trained on are skipped, since that would be an unfair test
def summarize(results):
    tested = [r for r in results if r["has_actual"] and not r["was_trained"]]
    if len(tested) == 0:
        return None

    seasons = len(tested)
    mvp_hits = sum(1 for r in tested if r["mvp_correct"])
    overlap = sum(r["overlap"] for r in tested)
    exact = sum(r["exact"] for r in tested)
    slots = sum(r["total"] for r in tested)

    return {
        "seasons": seasons,
        "mvp_hits": mvp_hits,
        "mvp_pct": round(100 * mvp_hits / seasons),
        "overlap": overlap,
        "overlap_pct": round(100 * overlap / slots),
        "exact": exact,
        "exact_pct": round(100 * exact / slots),
        "slots": slots,
    }


# page 1: pick a season and a model, then see both top 5 lists
@app.route("/")
def index():
    season = request.args.get("season", default=SEASONS[0], type=int)
    if season not in SEASONS:
        season = SEASONS[0]
    model_name = get_model_name()

    return render_template(
        "season.html",
        page="season",
        seasons=SEASONS,
        season=season,
        model_names=list(models.keys()),
        model_name=model_name,
        result=compare_season(season, model_name),
        train_end=TRAIN_END_YEAR,
    )


# page 2: see many seasons at once, like the last 10 years
@app.route("/history")
def history():
    years = request.args.get("years", default=10, type=int)
    if years not in YEAR_OPTIONS:
        years = 10
    model_name = get_model_name()

    # pick the most recent seasons that have real results
    if years == 0:
        window = RESULT_SEASONS
    else:
        window = RESULT_SEASONS[:years]

    results = [compare_season(s, model_name) for s in window]

    # score both models on the same seasons so they can be compared
    comparison = []
    for name in models:
        name_results = [compare_season(s, name) for s in window]
        comparison.append({"model": name, "summary": summarize(name_results)})

    return render_template(
        "history.html",
        page="history",
        year_options=YEAR_OPTIONS,
        years=years,
        model_names=list(models.keys()),
        model_name=model_name,
        results=results,
        summary=summarize(results),
        comparison=comparison,
        train_end=TRAIN_END_YEAR,
    )


if __name__ == "__main__":
    app.run(debug=True)