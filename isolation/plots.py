import ast
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import pandas as pd


INPUT_CSV = "tournament_results.csv"
OUTPUT_PREFIX = "plots/tournament"


def parse_params(value):
    return ast.literal_eval(value)


def params_label(params):
    t, c = params
    return f"t={t}, c={c}"


def prepare_df(path):
    df = pd.read_csv(path)

    df["red_params_tuple"] = df["red_params"].apply(parse_params)
    df["blue_params_tuple"] = df["blue_params"].apply(parse_params)

    df["red_label"] = df["red_params_tuple"].apply(params_label)
    df["blue_label"] = df["blue_params_tuple"].apply(params_label)

    df["games"] = df["red_wins"] + df["blue_wins"]
    df["red_win_rate"] = df["red_wins"] / df["games"]
    df["blue_win_rate"] = df["blue_wins"] / df["games"]

    return df


def get_players(df):
    players = sorted(
        set(df["red_params_tuple"].unique()) | set(df["blue_params_tuple"].unique())
    )
    return players


def build_fair_matrix(df, players):
    wins = {p: {q: 0 for q in players} for p in players}
    games = {p: {q: 0 for q in players} for p in players}

    for _, row in df.iterrows():
        red = row["red_params_tuple"]
        blue = row["blue_params_tuple"]

        red_wins = row["red_wins"]
        blue_wins = row["blue_wins"]
        total = row["games"]

        wins[red][blue] += red_wins
        games[red][blue] += total

        wins[blue][red] += blue_wins
        games[blue][red] += total

    matrix = np.full((len(players), len(players)), 0.0)

    for i, player in enumerate(players):
        for j, opponent in enumerate(players):
            if player == opponent:
                continue

            if games[player][opponent] > 0:
                matrix[i, j] = wins[player][opponent] / games[player][opponent]

    return matrix, wins, games


def plot_fair_matrix(matrix, players):
    labels = [params_label(p) for p in players]

    plt.figure(figsize=(10, 8))
    ax = sns.heatmap(
        matrix,
        annot=True,
        fmt=".2f",
        xticklabels=labels,
        yticklabels=labels,
        cmap="viridis",
        vmin=0,
        vmax=1,
        linewidths=0.5,
        linecolor="white",
        cbar_kws={"label": "Win rate"},
    )

    ax.set_title("Fair win rate: agent z wiersza vs agent z kolumny")
    ax.set_xlabel("Opponent")
    ax.set_ylabel("Agent")

    plt.xticks(rotation=45, ha="right")
    plt.yticks(rotation=0)
    plt.tight_layout()
    plt.savefig(f"{OUTPUT_PREFIX}_fair_matrix.png", dpi=200)
    plt.close()


def build_rankings(df, players):
    rows = []

    for player in players:
        as_red = df[df["red_params_tuple"] == player]
        as_blue = df[df["blue_params_tuple"] == player]

        red_wins = as_red["red_wins"].sum()
        red_games = as_red["games"].sum()

        blue_wins = as_blue["blue_wins"].sum()
        blue_games = as_blue["games"].sum()

        total_wins = red_wins + blue_wins
        total_games = red_games + blue_games

        rows.append(
            {
                "params": player,
                "label": params_label(player),
                "time_limit": player[0],
                "c_coefficient": player[1],
                "wins": total_wins,
                "games": total_games,
                "losses": total_games - total_wins,
                "win_rate": total_wins / total_games if total_games else np.nan,
                "red_wins": red_wins,
                "red_games": red_games,
                "red_win_rate": red_wins / red_games if red_games else np.nan,
                "blue_wins": blue_wins,
                "blue_games": blue_games,
                "blue_win_rate": blue_wins / blue_games if blue_games else np.nan,
            }
        )

    ranking = pd.DataFrame(rows)
    ranking = ranking.sort_values("win_rate", ascending=False).reset_index(drop=True)
    ranking["rank"] = ranking.index + 1

    return ranking


def plot_overall_ranking(ranking):
    plt.figure(figsize=(10, 6))

    sns.barplot(
        data=ranking,
        x="win_rate",
        y="label",
        orient="h",
    )

    plt.xlim(0, 1)
    plt.xlabel("Overall win rate")
    plt.ylabel("Agent")
    plt.title("Ranking agentów")
    plt.tight_layout()
    plt.savefig(f"{OUTPUT_PREFIX}_ranking_overall.png", dpi=200)
    plt.close()


def plot_red_blue_ranking(ranking):
    melted = ranking.melt(
        id_vars=["label"],
        value_vars=["red_win_rate", "blue_win_rate"],
        var_name="side",
        value_name="side_win_rate",
    )

    melted["side"] = melted["side"].map(
        {
            "red_win_rate": "RED",
            "blue_win_rate": "BLUE",
        }
    )

    plt.figure(figsize=(10, 6))

    sns.barplot(
        data=melted,
        x="side_win_rate",
        y="label",
        hue="side",
        orient="h",
    )

    plt.xlim(0, 1)
    plt.xlabel("Win rate")
    plt.ylabel("Agent")
    plt.title("Win rate jako RED vs jako BLUE")
    plt.tight_layout()
    plt.savefig(f"{OUTPUT_PREFIX}_ranking_red_blue.png", dpi=200)
    plt.close()


def plot_time_limit_effect(ranking):
    grouped = (
        ranking.groupby("time_limit", as_index=False)
        .agg(mean_win_rate=("win_rate", "mean"))
        .sort_values("time_limit")
    )

    plt.figure(figsize=(8, 5))

    sns.lineplot(
        data=grouped,
        x="time_limit",
        y="mean_win_rate",
        marker="o",
    )

    plt.ylim(0, 1)
    plt.xlabel("time_limit")
    plt.ylabel("Mean win rate")
    plt.title("Wpływ czasu namysłu na skuteczność")
    plt.tight_layout()
    plt.savefig(f"{OUTPUT_PREFIX}_time_limit_effect.png", dpi=200)
    plt.close()


def plot_c_effect(ranking):
    grouped = (
        ranking.groupby("c_coefficient", as_index=False)
        .agg(mean_win_rate=("win_rate", "mean"))
        .sort_values("c_coefficient")
    )

    plt.figure(figsize=(8, 5))

    sns.barplot(
        data=grouped,
        x="c_coefficient",
        y="mean_win_rate",
    )

    plt.ylim(0, 1)
    plt.xlabel("c_coefficient")
    plt.ylabel("Mean win rate")
    plt.title("Wpływ współczynnika eksploracji c")
    plt.tight_layout()
    plt.savefig(f"{OUTPUT_PREFIX}_c_effect.png", dpi=200)
    plt.close()


def plot_param_heatmap(ranking):
    pivot = ranking.pivot_table(
        index="time_limit",
        columns="c_coefficient",
        values="win_rate",
        aggfunc="mean",
    )

    plt.figure(figsize=(8, 6))

    sns.heatmap(
        pivot,
        annot=True,
        fmt=".2f",
        cmap="viridis",
        vmin=0,
        vmax=1,
        linewidths=0.5,
        linecolor="white",
        cbar_kws={"label": "Overall win rate"},
    )

    plt.xlabel("c_coefficient")
    plt.ylabel("time_limit")
    plt.title("Skuteczność parametrów MCTS")
    plt.tight_layout()
    plt.savefig(f"{OUTPUT_PREFIX}_param_heatmap.png", dpi=200)
    plt.close()


def plot_game_lengths(df):
    plt.figure(figsize=(8, 5))

    sns.histplot(
        data=df,
        x="game_length",
        bins=12,
    )

    plt.xlabel("Game length")
    plt.ylabel("Count")
    plt.title("Rozkład długości partii")
    plt.tight_layout()
    plt.savefig(f"{OUTPUT_PREFIX}_game_lengths_hist.png", dpi=200)
    plt.close()


def plot_game_length_by_winner(df):
    plt.figure(figsize=(8, 5))

    sns.boxplot(
        data=df,
        x="winner",
        y="game_length",
    )

    plt.xlabel("Winner")
    plt.ylabel("Game length")
    plt.title("Długość partii względem zwycięzcy")
    plt.tight_layout()
    plt.savefig(f"{OUTPUT_PREFIX}_game_length_by_winner.png", dpi=200)
    plt.close()


def main():
    df = prepare_df(INPUT_CSV)
    players = get_players(df)

    fair_matrix, wins, games = build_fair_matrix(df, players)
    ranking = build_rankings(df, players)

    plot_fair_matrix(fair_matrix, players)
    plot_overall_ranking(ranking)
    plot_red_blue_ranking(ranking)
    plot_time_limit_effect(ranking)
    plot_c_effect(ranking)
    plot_param_heatmap(ranking)
    plot_game_lengths(df)
    plot_game_length_by_winner(df)


if __name__ == "__main__":
    main()
