from problem import Corner, Experiment, Environment
from problem import OffPolicyNStepSarsaDriver, RandomDriver

import sys
import os

def main() -> None:
    if len(sys.argv) != 3:
        print("Usage: python visualize.py <driver_path> <corner_name>")
        return
    else:
        driver_path = sys.argv[1]
        corner_name = sys.argv[2]

    driver = OffPolicyNStepSarsaDriver.load(driver_path)
    driver.experiment_rate = 0.0
    driver.step_size = 0.0

    os.makedirs(f'test_plots_{driver_path}', exist_ok=True)

    experiment = Experiment(
        environment=Environment(
            corner=Corner(
                name=corner_name
            ),
            steering_fail_chance=0.01,
        ),
        driver=driver,
        number_of_episodes=10,
        plots_path=f'test_plots_{driver_path}',
        drawing_frequency=1,
        averaging_window_size=1,
    )

    experiment.run()

if __name__ == '__main__':
    main()
