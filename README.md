# NBA MVP Predictor

A web app that predicts NBA MVP voting from player stats and compares the predictions to the real results. I scraped Basketball Reference for every season from 1982 to 2026, trained a Random Forest and a Gradient Boosting model on stats like PER, Win Shares, VORP and team win %, and the app shows each season's predicted top 5 next to the real top 5.

## Results

I tested the models with cross validation leave-one-out training on every season in the data. Each season gets it's turn as the test season. The models train on all the other seasons, then predict the one they left out. 

Both models predicted the MVP in their top 3 with 96% accuracy. In terms of picking the exact MVP winner, Random Forest got it right 69% of the time while Gradient Boosting had a 71% accuracy rate.  Win Shares, PER and team win % mattered most to both models. The misses mostly came in years where the voters went with a storyline or a team's record over the best stats. Getting the exact order of the top 5 is much harder than picking the winner. The Testing page in the app shows the full breakdown.

## Earlier test

Before cross validation I did a simple split: train on 1982-2010, test on 2011-2026. Gradient Boosting had the real MVP in its top 3 in 16 of 16 seasons and picked the MVP exactly in 14 of 16. Random Forest got 15 of 16 and picked the MVP 13 of 16. The two models finished close to each other. This test isn't part of the app, but `python train_model.py` still runs it.

## Run it

```
pip install -r requirements.txt
python app.py
```

Then open http://127.0.0.1:5000. The scraped data and predictions are already in `data/`.

### Rebuilding the data (optional)

1. `python scrape_data.py` downloads the stats
2. `python cross_validate.py` makes the predictions for every season

Then run `python app.py` 

Data from [Basketball Reference](https://www.basketball-reference.com/).