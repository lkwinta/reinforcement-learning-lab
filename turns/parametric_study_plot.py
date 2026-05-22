import parse
import os

import numpy as np
from matplotlib import pyplot as plt


def main() -> None:
    root_dir = "parametric_study_results"
    results = {}

    old_results = False

    if old_results:
        for dir_name in os.listdir(root_dir):
            step_no, alpha = parse.parse("step_no_{}_alpha_{:f}", dir_name)
            if os.path.exists(f"{root_dir}/{dir_name}/results.txt"):
                with open(f"{root_dir}/{dir_name}/results.txt", "r") as f:
                    lines = f.readlines()
                    avg_eval_penalty = float(lines[2].split(",")[1])

                if step_no not in results:
                    results[step_no] = []

                results[step_no].append((alpha, -avg_eval_penalty))

    else:
        for step_no_dir in os.listdir(root_dir):
            step_no = parse.parse("step_no_{}", step_no_dir)[0]

            for alpha_dir in os.listdir(f"{root_dir}/{step_no_dir}"):
                alpha = parse.parse("alpha_{:f}", alpha_dir)[0]

                dir = f"step_no_{step_no}/alpha_{alpha:.2f}"

                if os.path.exists(f"{root_dir}/{dir}/eval_penalties.txt"):
                    with open(f"{root_dir}/{dir}/eval_penalties.txt", "r") as f:
                        lines = np.array(f.readlines(), dtype=float)
                        avg_eval_penalty = np.mean(lines)

                    if step_no not in results:
                        results[step_no] = []

                    results[step_no].append((alpha, -avg_eval_penalty))

    plt.figure(figsize=(10, 8))
    plt.yscale("log")
    plt.grid(True, which="both", ls="--", linewidth=0.5)

    min_result = min(
        [
            (alpha, penalty, step_no)
            for step_no in results.keys()
            for alpha, penalty in results[step_no]
        ],
        key=lambda x: x[1],
    )

    for step_no in sorted(results.keys()):
        alphas, penalties = zip(*sorted(results[step_no]))

        plt.plot(
            alphas, penalties, label=f"step_no={step_no}", marker="o", markersize=3
        )

    min_alpha, min_penalty, min_step_no = min_result
    plt.axhline(y=min_penalty, color="r", linestyle="--", linewidth=1)
    plt.axvline(x=min_alpha, color="g", linestyle="--", linewidth=1)

    plt.annotate(
        f"Minimum\nn={min_step_no}, alpha={min_alpha:.2f}\npenalty={-min_penalty:.2f}",
        xy=(min_alpha, min_penalty),
        xytext=(45, -30),  # niżej od punktu
        textcoords="offset points",
        ha="center",
        va="top",
        fontsize=10,
        bbox=dict(
            boxstyle="round,pad=0.35",
            fc="white",
            ec="black",
            alpha=0.6,  # bardziej przezroczysty
        ),
        arrowprops=dict(
            arrowstyle="->",
            color="black",
            lw=1.2,
            shrinkA=0,
            shrinkB=5,
        ),
    )

    plt.xlabel("alpha")
    plt.ylabel("average eval penalty * -1")
    plt.title("Parametric Study Results")
    plt.legend()
    plt.savefig("raport_imgs/parametric_study_plot.png")


if __name__ == "__main__":
    main()
