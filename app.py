# flask web app that compares our predicted mvp candidates to the real mvp race
# run train_model.py first so the saved models exist

import os

import joblib
from flask import Flask, render_template, request

# reuse the same settings and data cleaning we used for training
from train_model import load_data, FEATURES, MODEL_DIR, TRAIN_END_YEAR

app = Flask(__name__)


MODEL_FILES = {
    "Random Forest": "random_forest.joblib",
    "Gradient Boosting": "gradient_boosting.joblib",
}

# load the data and the trained models once when the app starts
df = load_data()
models = {}
for name, filename in MODEL_FILES.items():
    models[name] = joblib.load(os.path.join(MODEL_DIR, filename))

# add a predicted vote share column for each model
for name, model in models.items():
    df[name] = model.predict(df[FEATURES])

# list of all seasons
SEASONS = sorted([int(s) for s in df["Season"].unique()], reverse=True)


# turn the top 5 rows of a table into dictionaries for html
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


#  see both top 5 lists
@app.route("/")
def index():
    # read the choices from the url (or use defaults)
    season = request.args.get("season", default=SEASONS[0], type=int)
    model_name = request.args.get("model", default="Random Forest")
    if season not in SEASONS:
        season = SEASONS[0]
    if model_name not in models:
        model_name = "Random Forest"

   
    season_df = df[df["Season"] == season]

   
    predicted = top_five(season_df, model_name)

   
    voted_df = season_df[season_df["Share"] > 0]
    actual = top_five(voted_df, "Share")

    # mark predicted players who really finished in the actual top 5
    actual_names = [p["player"] for p in actual]
    for p in predicted:
        p["in_actual"] = p["player"] in actual_names

  
    has_actual = len(actual) > 0
    correct = has_actual and predicted[0]["player"] == actual[0]["player"]

    return render_template(
        "index.html",
        seasons=SEASONS,
        season=season,
        model_names=list(models.keys()),
        model_name=model_name,
        predicted=predicted,
        actual=actual,
        has_actual=has_actual,
        correct=correct,
        was_trained=season <= TRAIN_END_YEAR,
        train_end=TRAIN_END_YEAR,
    )


if __name__ == "__main__":
    app.run(debug=True)