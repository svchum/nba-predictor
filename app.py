# flask web app that compares predicted mvp candidates to the real mvp race

import os

import pandas as pd
from flask import Flask, render_template, request
from sklearn.metrics import r2_score

app = Flask(__name__)



PREDICTIONS_PATH = os.path.join("data", "cv_predictions.csv")


MODEL_NAMES = ["Random Forest", "Gradient Boosting"]


YEAR_OPTIONS = [5, 10, 15, 20, 0]

# stop if the predictions file doesn't exist yet
if not os.path.exists(PREDICTIONS_PATH):
    raise SystemExit("Could not find " + PREDICTIONS_PATH + ". Run cross_validate.py first.")

df = pd.read_csv(PREDICTIONS_PATH)

SEASONS = sorted([int(s) for s in df["Season"].unique()], reverse=True)


voted_seasons = df[df["Share"] > 0]["Season"].unique()
RESULT_SEASONS = sorted([int(s) for s in voted_seasons], reverse=True)


# send the testing info to every page
@app.context_processor
def add_test_info():
    return {
        "first_season": min(RESULT_SEASONS),
        "last_season": max(RESULT_SEASONS),
        "season_count": len(RESULT_SEASONS),
    }


# read the model choice from the url
def get_model_name():
    model_name = request.args.get("model", default="Random Forest")
    if model_name not in MODEL_NAMES:
        model_name = "Random Forest"
    return model_name


# turn the top 5 rows of a table into simple dictionaries
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


# build the predicted vs actual comparison
def compare_season(season, model_name):
    season_df = df[df["Season"] == season]
    predicted = top_five(season_df, model_name)

    actual = top_five(season_df[season_df["Share"] > 0], "Share")
    has_actual = len(actual) > 0

    result = {
        "season": season,
        "predicted": predicted,
        "actual": actual,
        "has_actual": has_actual,
        "total": len(actual),
        "mvp_correct": False,
        "mvp_in_top3": False,
        "overlap": 0,
        "exact": 0,
    }


    if not has_actual:
        for row in predicted:
            row["status"] = "none"
            row["other_rank"] = None
        return result

    add_status(predicted, [a["player"] for a in actual])
    add_status(actual, [p["player"] for p in predicted])

# checking off the boxes
    result["mvp_correct"] = predicted[0]["player"] == actual[0]["player"]
 
    result["mvp_in_top3"] = actual[0]["player"] in [p["player"] for p in predicted[:3]]

    result["overlap"] = sum(1 for p in predicted if p["status"] != "miss")
  
    result["exact"] = sum(1 for p in predicted if p["status"] == "exact")
    return result


# add up the accuracy numbers
def summarize(results):
    tested = [r for r in results if r["has_actual"]]
    if len(tested) == 0:
        return None

    seasons = len(tested)
    mvp_hits = sum(1 for r in tested if r["mvp_correct"])
    top3_hits = sum(1 for r in tested if r["mvp_in_top3"])
    overlap = sum(r["overlap"] for r in tested)
    exact = sum(r["exact"] for r in tested)
    slots = sum(r["total"] for r in tested)

    return {
        "seasons": seasons,
        "mvp_hits": mvp_hits,
        "mvp_pct": round(100 * mvp_hits / seasons),
        "top3_hits": top3_hits,
        "top3_pct": round(100 * top3_hits / seasons),
        "overlap_pct": round(100 * overlap / slots),
        "exact_pct": round(100 * exact / slots),
    }


#pick a season and a model, then see both top 5 lists
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
        model_names=MODEL_NAMES,
        model_name=model_name,
        result=compare_season(season, model_name),
    )


# see many multiple seasons 
@app.route("/history")
def history():
    years = request.args.get("years", default=10, type=int)
    if years not in YEAR_OPTIONS:
        years = 10
    model_name = get_model_name()

  
    if years == 0:
        window = RESULT_SEASONS
    else:
        window = RESULT_SEASONS[:years]

    results = [compare_season(s, model_name) for s in window]

    # score both models on the same seasons
    comparison = []
    for name in MODEL_NAMES:
        name_results = [compare_season(s, name) for s in window]
        comparison.append({"model": name, "summary": summarize(name_results)})

    return render_template(
        "history.html",
        page="history",
        year_options=YEAR_OPTIONS,
        years=years,
        model_names=MODEL_NAMES,
        model_name=model_name,
        results=results,
        comparison=comparison,
    )


# how well did each model do across every tested season
@app.route("/testing")
def testing():
    
    all_results = {}
    for name in MODEL_NAMES:
        all_results[name] = [compare_season(s, name) for s in RESULT_SEASONS]

    # overall accuracy for each model
    overall = []
    for name in MODEL_NAMES:
        summary = summarize(all_results[name])
        summary["r2"] = round(r2_score(df["Share"], df[name]), 3)
        overall.append({"model": name, "summary": summary})

 
    misses = []
    for i, season in enumerate(RESULT_SEASONS):
        picks = []
        for name in MODEL_NAMES:
            r = all_results[name][i]
            picks.append({"player": r["predicted"][0]["player"], "correct": r["mvp_correct"]})
            actual_mvp = r["actual"][0]["player"]
        if any(not p["correct"] for p in picks):
            misses.append({"season": season, "actual": actual_mvp, "picks": picks})

    return render_template(
        "testing.html",
        page="testing",
        model_names=MODEL_NAMES,
        overall=overall,
        misses=misses,
    )


if __name__ == "__main__":
    app.run(debug=True)