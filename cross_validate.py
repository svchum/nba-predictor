# cross validation: tests the models on every season in the data
# the predictions are saved to a csv file that the web app reads

import os

import pandas as pd
from sklearn.metrics import r2_score


from train_model import load_data, make_models, FEATURES, TARGET


OUT_PATH = os.path.join("data", "cv_predictions.csv")


# train on every season except one, then predict that one season
def predict_season(df, season):
    test = df[df["Season"] == season].copy()
    train = df[df["Season"] != season]


    for name, model in make_models().items():
        model.fit(train[FEATURES], train[TARGET])
        test[name] = model.predict(test[FEATURES])
    return test



def predict_all_seasons(df):
    frames = []

    for season in sorted(df["Season"].unique()):
        if df[df["Season"] == season][TARGET].max() == 0:
            print("skipped", season, "(no mvp results)")
            continue

        frames.append(predict_season(df, season))
        print("tested", season)

    return pd.concat(frames, ignore_index=True)


# print a score for each model
def print_scores(results):
    for name in make_models():
        hits = 0
        seasons = 0
        for season, group in results.groupby("Season"):
            seasons += 1
            actual_mvp = group.loc[group[TARGET].idxmax(), "Player"]
            predicted_mvp = group.loc[group[name].idxmax(), "Player"]
            if actual_mvp == predicted_mvp:
                hits += 1
        r2 = r2_score(results[TARGET], results[name])
        print("{}: correct MVP {}/{} ({:.0f}%), R2 {:.3f}".format(
            name, hits, seasons, 100 * hits / seasons, r2))


def main():
    df = load_data()
    print("testing", df["Season"].nunique(), "seasons")
    print("this trains a new model for every season, so it can take a few minutes\n")

    results = predict_all_seasons(df)

    os.makedirs("data", exist_ok=True)
    results.to_csv(OUT_PATH, index=False)
    print("\nsaved", len(results), "rows to", OUT_PATH)
    print_scores(results)


if __name__ == "__main__":
    main()