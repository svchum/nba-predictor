# downloads nba stats from basketball reference and saves them locally


import os
import time
from io import StringIO

import requests
import pandas as pd
from bs4 import BeautifulSoup

BASE_URL = "https://www.basketball-reference.com"
DATA_DIR = "data"
RAW_DIR = os.path.join(DATA_DIR, "raw")

# season start/end
START_YEAR = 1982
END_YEAR = 2026

REQUEST_DELAY = 3.5

HEADERS = {
    "User-Agent": (
        "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) "
        "AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0 Safari/537.36"
    )
}

PER_GAME_COLS = ["Player", "Team", "Season", "Pos", "Age", "G", "GS", "MP",
                 "PTS", "TRB", "AST", "STL", "BLK", "TOV",
                 "FG%", "3P%", "FT%", "eFG%"]
ADVANCED_COLS = ["Player", "Team", "Season", "PER", "TS%", "USG%",
                 "WS", "WS/48", "OBPM", "DBPM", "BPM", "VORP"]

# player played for more than one team
MULTI_TEAM = {"2TM", "3TM", "4TM", "5TM", "TOT"}

TEXT_COLS = ["Player", "Team", "Pos"]


# fetch page from basketball reference
def get_page(url):
    response = requests.get(url, headers=HEADERS, timeout=30)
    response.raise_for_status()
    response.encoding = "utf-8"
    time.sleep(REQUEST_DELAY)
    return response.text


# pull a table out of the html by its id
def read_table(html, table_id):
    clean_html = html.replace("<!--", "").replace("-->", "")
    soup = BeautifulSoup(clean_html, "lxml")
    table = soup.find("table", id=table_id)
    if table is None:
        return None
    return pd.read_html(StringIO(str(table)))[0]


# get per game or advanced stats for a season
def scrape_stats(year, kind):
    url = "{}/leagues/NBA_{}_{}.html".format(BASE_URL, year, kind)
    html = get_page(url)
    table_id = "per_game_stats" if kind == "per_game" else "advanced"
    df = read_table(html, table_id)
    if df is None:
        return None

    # remove repeat header row every 20 rows
    df = df[df["Rk"] != "Rk"].copy()
    df = df.drop(columns=[c for c in df.columns if c.startswith("Unnamed")],
                 errors="ignore")

    df = df.rename(columns={"Tm": "Team"})
    df["Season"] = year
    return df


# get team win % / season
def scrape_standings(year):
    url = "{}/leagues/NBA_{}_standings.html".format(BASE_URL, year)
    html = get_page(url)
    clean_html = html.replace("<!--", "").replace("-->", "")
    soup = BeautifulSoup(clean_html, "lxml")

    records = []
    table_ids = [
        "confs_standings_E", "confs_standings_W",
        "divs_standings_E", "divs_standings_W",
    ]
    for table_id in table_ids:
        table = soup.find("table", id=table_id)
        if table is None:
            continue
        for row in table.find_all("tr"):
            link = row.find("a", href=True)
            win_cell = row.find("td", {"data-stat": "win_loss_pct"})
            if link is None or win_cell is None:
                continue
            parts = link["href"].split("/")
            if len(parts) < 3:
                continue
            try:
                win_pct = float(win_cell.get_text())
            except ValueError:
                continue
            records.append({"Team": parts[2], "WinPct": win_pct})

    standings = pd.DataFrame(records)
    if standings.empty:
        return standings

    return standings.drop_duplicates(subset=["Team"], keep="first")


# get mvp voting results for a season
def scrape_mvp(year):
    url = "{}/awards/awards_{}.html".format(BASE_URL, year)
    html = get_page(url)
    df = read_table(html, "mvp")
    if df is None:
        return None


    if isinstance(df.columns, pd.MultiIndex):
        df.columns = [c[1] for c in df.columns]
    df = df[["Player", "Share"]].copy()
    df["Season"] = year
    return df


# combine all the tables for one season into one dataframe
def build_season(year):
    per_game = scrape_stats(year, "per_game")
    advanced = scrape_stats(year, "advanced")
    standings = scrape_standings(year)
    mvp = scrape_mvp(year)
    if per_game is None or advanced is None:
        return None


    per_game = per_game[[c for c in PER_GAME_COLS if c in per_game.columns]]
    advanced = advanced[[c for c in ADVANCED_COLS if c in advanced.columns]]

    # join per game and advanced stats together for each player
    df = per_game.merge(advanced, on=["Player", "Team", "Season"], how="inner")

    # if a player was traded, keep the team he played the most games for
    df = df[~df["Team"].isin(MULTI_TEAM)].copy()
    df["G"] = pd.to_numeric(df["G"], errors="coerce")
    df = df.sort_values("G", ascending=False).drop_duplicates(
        subset=["Player", "Season"], keep="first")

    df = df.merge(standings, on="Team", how="left")

    # add the mvp vote share
    if mvp is not None:
        mvp = mvp.drop_duplicates(subset=["Player", "Season"])
        df = df.merge(mvp[["Player", "Season", "Share"]],
                      on=["Player", "Season"], how="left")
    df["Share"] = pd.to_numeric(df.get("Share"), errors="coerce").fillna(0.0)

    # turn all the stat columns into numbers
    for col in df.columns:
        if col not in TEXT_COLS:
            df[col] = pd.to_numeric(df[col], errors="coerce")
    return df


# scrape multiple seasons and save to csv
def build_dataset(start_year, end_year):
    os.makedirs(RAW_DIR, exist_ok=True)
    frames = []
    for year in range(start_year, end_year + 1):
        cache_path = os.path.join(RAW_DIR, "{}.csv".format(year))

     
        if os.path.exists(cache_path):
            print("loading cached", year)
            frames.append(pd.read_csv(cache_path))
            continue

        print("scraping", year)
        try:
            season = build_season(year)
        except Exception as error:
            print("  failed", year, "-", error)
            continue
        if season is None:
            print("  skipped", year)
            continue
        season.to_csv(cache_path, index=False)
        frames.append(season)

    # stack all seasons together and save one big file
    full = pd.concat(frames, ignore_index=True)
    out_path = os.path.join(DATA_DIR, "mvp_dataset.csv")
    full.to_csv(out_path, index=False)
    print("saved", full.shape, "to", out_path)
    return full


def main():
    build_dataset(START_YEAR, END_YEAR)


if __name__ == "__main__":
    main()