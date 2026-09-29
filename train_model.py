# trains random forest and gradient boosting models to predict nba mvp vote share
# train on seasons 1982-2010, test on 2011 and later

import os

import joblib
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

DATA_PATH = os.path.join("data", "mvp_dataset.csv")
MODEL_DIR = "models"


TRAIN_END_YEAR = 2010

# players need at least this many games to be mvp considered
MIN_GAMES = 41

FEATURES = ["Age", "G", "GS", "MP", "PTS", "TRB", "AST", "STL", "BLK", "TOV",
            "FG%", "3P%", "FT%", "eFG%", "PER", "TS%", "USG%",
            "WS", "WS/48", "OBPM", "DBPM", "BPM", "VORP", "WinPct"]


TARGET = "Share"


# load the csv and clean
def load_data():
    df = pd.read_csv(DATA_PATH)
    df = df[df["G"] >= MIN_GAMES].copy()
    df[FEATURES] = df[FEATURES].fillna(0)
    df[TARGET] = df[TARGET].fillna(0)
    return df


def split_data(df):
    train = df[df["Season"] <= TRAIN_END_YEAR]
    test = df[df["Season"] > TRAIN_END_YEAR]
    return train, test


# create models
def make_models():
    rf = RandomForestRegressor(
        n_estimators=300, min_samples_leaf=2, random_state=42, n_jobs=-1)
    gb = GradientBoostingRegressor(
        n_estimators=300, learning_rate=0.05, max_depth=3, random_state=42)
    return {"Random Forest": rf, "Gradient Boosting": gb}


# check how close the predicted vote shares are to the real ones
def print_scores(name, y_true, y_pred):
    mae = mean_absolute_error(y_true, y_pred)
    rmse = np.sqrt(mean_squared_error(y_true, y_pred))
    r2 = r2_score(y_true, y_pred)
    print("{}: MAE={:.4f}  RMSE={:.4f}  R2={:.3f}".format(name, mae, rmse, r2))


# check if the model's top pick was the real mvp
def check_mvp_picks(name, test):
    print("\n{} - top pick for each season".format(name))
    correct = 0
    top3_correct = 0
    seasons_checked = 0

    for season, group in test.groupby("Season"):
  
        if group[TARGET].max() == 0:
            continue
        seasons_checked += 1

        real_mvp = group.loc[group[TARGET].idxmax(), "Player"]
        ranked = group.sort_values("Predicted", ascending=False)
        top_pick = ranked.iloc[0]["Player"]
        top3 = ranked.head(3)["Player"].tolist()

        if top_pick == real_mvp:
            correct += 1
        if real_mvp in top3:
            top3_correct += 1
        print("  {}  predicted: {:<22} actual: {}".format(
            season, top_pick, real_mvp))

    print("Correct MVP: {}/{}".format(correct, seasons_checked))
    print("MVP in top 3: {}/{}".format(top3_correct, seasons_checked))


# show which stats the model found most useful
def print_importance(name, model):
    importance = pd.Series(model.feature_importances_, index=FEATURES)
    print("\n{} - top 8 features".format(name))
    print(importance.sort_values(ascending=False).head(8).round(3).to_string())


def main():
    df = load_data()
    train, test = split_data(df)
    print("training rows:", len(train), " test rows:", len(test))

    os.makedirs(MODEL_DIR, exist_ok=True)

    # train each model and score
    for name, model in make_models().items():
        print("\n==============", name, "==============")
        model.fit(train[FEATURES], train[TARGET])


        test_copy = test.copy()
        test_copy["Predicted"] = model.predict(test[FEATURES])

        print_scores(name, test_copy[TARGET], test_copy["Predicted"])
        check_mvp_picks(name, test_copy)
        print_importance(name, model)

   
        filename = name.lower().replace(" ", "_") + ".joblib"
        joblib.dump(model, os.path.join(MODEL_DIR, filename))


    joblib.dump(FEATURES, os.path.join(MODEL_DIR, "features.joblib"))
    print("\nmodels saved to", MODEL_DIR)


if __name__ == "__main__":
    main()