from pathlib import Path
import re

import matplotlib.pyplot as plt
import numpy as np


ROOT = Path("parametric_study_results")

NUMBER_OF_EPISODES = 30000
DRAWING_FREQUENCY = 1000

COUNT_EPISODE_ZERO = True

OUTPUT_PATH = "drawn_epochs_progress.png"

IMAGE_EXTENSIONS = {
    ".png",
    ".jpg",
    ".jpeg",
    ".pdf",
    ".svg",
}


def expected_draw_count() -> int:
    count = NUMBER_OF_EPISODES // DRAWING_FREQUENCY

    if COUNT_EPISODE_ZERO:
        count += 1

    return count


def extract_epoch_number(path: Path) -> int | None:
    numbers = re.findall(r"\d+", path.stem)

    if not numbers:
        return None

    return int(numbers[-1])


def count_drawn_epochs(plots_dir: Path) -> tuple[int, list[int]]:
    if not plots_dir.exists():
        return 0, []

    epochs = set()

    for path in plots_dir.iterdir():
        if not path.is_file():
            continue

        if path.suffix.lower() not in IMAGE_EXTENSIONS:
            continue

        epoch = extract_epoch_number(path)

        if epoch is not None:
            epochs.add(epoch)

    return len(epochs), sorted(epochs)


def parse_case_dir(alpha_dir: Path) -> tuple[int, float]:
    step_no = int(alpha_dir.parent.name.replace("step_no_", ""))
    alpha = float(alpha_dir.name.replace("alpha_", ""))

    return step_no, alpha


def collect_results() -> list[dict]:
    rows = []

    for step_dir in sorted(ROOT.glob("step_no_*")):
        if not step_dir.is_dir():
            continue

        for alpha_dir in sorted(step_dir.glob("alpha_*")):
            if not alpha_dir.is_dir():
                continue

            plots_dir = alpha_dir / "plots"

            step_no, alpha = parse_case_dir(alpha_dir)
            drawn_count, drawn_epochs = count_drawn_epochs(plots_dir)

            max_drawn_epoch = max(drawn_epochs) if drawn_epochs else 0
            expected_count = expected_draw_count()

            rows.append(
                {
                    "step_no": step_no,
                    "alpha": alpha,
                    "plots_dir": plots_dir,
                    "drawn_count": drawn_count,
                    "drawn_epochs": drawn_epochs,
                    "max_drawn_epoch": max_drawn_epoch,
                    "expected_count": expected_count,
                    "progress_percent": min(drawn_count / expected_count, 1.0) * 100.0,
                    "label": f"n={step_no}, α={alpha:.2f}",
                }
            )

    rows.sort(key=lambda row: (row["step_no"], row["alpha"]))
    return rows


def plot_progress(rows: list[dict]) -> None:
    if not rows:
        print(f"No parametric study results found in: {ROOT}")
        return

    labels = [row["label"] for row in rows]
    values = [row["progress_percent"] for row in rows]

    fig_height = max(8, 0.35 * len(rows))
    plt.figure(figsize=(14, fig_height))

    y = np.arange(len(rows))
    plt.barh(y, values)

    plt.yticks(y, labels)
    plt.xlabel("Progress based on drawn epochs [%]")
    plt.ylabel("Case")
    plt.title("Parametric study progress based on files in plots directories")
    plt.xlim(0, 100)
    plt.grid(axis="x", alpha=0.3)

    for i, row in enumerate(rows):
        text = (
            f'{row["drawn_count"]}/{row["expected_count"]} '
            f'last={row["max_drawn_epoch"]}'
        )

        x = min(row["progress_percent"] + 1.0, 98.0)

        plt.text(
            x,
            i,
            text,
            va="center",
            fontsize=8,
        )

    plt.tight_layout()
    plt.savefig(OUTPUT_PATH, dpi=200)
    plt.show()

    print(f"Saved plot to: {OUTPUT_PATH}")


def print_summary(rows: list[dict]) -> None:
    for row in rows:
        print(
            f'{row["label"]:16s} | '
            f'drawn={row["drawn_count"]:3d}/{row["expected_count"]:3d} | '
            f'progress={row["progress_percent"]:6.2f}% | '
            f'last_epoch={row["max_drawn_epoch"]:6d} | '
            f'{row["plots_dir"]}'
        )


def main() -> None:
    rows = collect_results()
    print_summary(rows)
    plot_progress(rows)


if __name__ == "__main__":
    main()
