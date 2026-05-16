import os
from matplotlib import pyplot as plt
import json
import argparse

def main(files: list[str], labels: list[str], output: str) -> None:
    results = {}

    for file, label in zip(files, labels):
        with open(file, "r") as f:
            data = json.load(f)

        results[label] = {}
        
        for entry in data:
            name = entry["name"]
            value = entry["value"]
            step = entry.get("step", None)

            if name not in results[label]:
                results[label][name] = {"steps": [], "values": []}

            if step is not None:
                results[label][name]["steps"].append(step)
                results[label][name]["values"].append(value)
    # Plotting
    plot_dict = {}

    for algo_label, metrics in results.items():
        for metric_name, data in metrics.items():
            if metric_name not in plot_dict:
                plot_dict[metric_name] = {}

            plot_dict[metric_name][algo_label] = (data["steps"], data["values"])
            
    for metric_name, algos in plot_dict.items():
        plt.figure()
        for algo_label, (steps, values) in algos.items():
            plt.plot(steps, values, label=algo_label)
        plt.xlabel("Steps")
        plt.ylabel(metric_name)
        plt.title(f"{metric_name} over Steps")
        plt.legend()
        plt.grid()
        plt.savefig(f"{output}/{metric_name}.png")
        plt.close()
            
def file_label_pair(value: str) -> tuple[str, str]:
    try:
        file, label = value.split(":", 1)
    except ValueError:
        raise argparse.ArgumentTypeError(
            "Expected pair in format FILE:LABEL"
        )

    if not file:
        raise argparse.ArgumentTypeError("File path cannot be empty")

    if not label:
        raise argparse.ArgumentTypeError("Label cannot be empty")

    return file, label
    
if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("files_and_labels", nargs="+", help="Pairs of log files and their corresponding labels", type=file_label_pair)
    parser.add_argument("--output", type=str, help="Directory to save the plot", default="plots")
    args = parser.parse_args()

    files, labels = zip(*args.files_and_labels)

    if not os.path.exists(args.output):
        os.makedirs(args.output)

    main(files, labels, output=args.output)
