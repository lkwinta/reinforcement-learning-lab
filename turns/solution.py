from problem import Corner, Experiment, Environment
from problem import OffPolicyNStepSarsaDriver, RandomDriver

import os

MAX_LEARNING_STEPS = 2000

def experiment(name, step_no, alpha, corner):
    os.makedirs(f'{name}_plots', exist_ok=True)

    driver = OffPolicyNStepSarsaDriver(
        step_no=step_no,
        step_size=alpha,
        experiment_rate=0.05,
        discount_factor=1.00,
        max_learning_steps=MAX_LEARNING_STEPS,
    )

    experiment = Experiment(
        environment=Environment(
            corner=Corner(
                name=corner
            ),
            steering_fail_chance=0.01,
        ),
        driver=driver,
        number_of_episodes=30000,
        drawing_frequency=1000,
        plots_path=f'{name}_plots',
    )

    experiment.run()

    driver.save(f'{name}_driver.pkl')

def main() -> None:
    experiment(
        name='corner_b',
        step_no=5,
        alpha=0.3,
        corner='corner_b',
    )

    experiment(
        name='corner_c',
        step_no=5,
        alpha=0.3,
        corner='corner_c',
    )

    experiment(
        name='corner_d',
        step_no=3,
        alpha=0.68,
        corner='corner_d',
    )

if __name__ == '__main__':
    main()
