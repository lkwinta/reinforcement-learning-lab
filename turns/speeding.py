from problem import Corner, Experiment, Environment
from problem import OffPolicyNStepSarsaDriver, RandomDriver

import os

MAX_LEARNING_STEPS = 2000

def experiment(name, step_no, alpha, corner, use_speeding_policy=False, speeding_rate=0.0, use_importance_sampling=True):
    os.makedirs(f'{name}_plots', exist_ok=True)

    driver = OffPolicyNStepSarsaDriver(
        step_no=step_no,
        step_size=alpha,
        experiment_rate=0.05,
        discount_factor=1.00,
        max_learning_steps=MAX_LEARNING_STEPS,
        use_speeding_policy=use_speeding_policy,
        speeding_rate=speeding_rate,
        use_importance_sampling=use_importance_sampling
    )

    experiment = Experiment(
        environment=Environment(
            corner=Corner(
                name=corner
            ),
            steering_fail_chance=0.01,
        ),
        driver=driver,
        number_of_episodes=60000,
        drawing_frequency=1000,
        plots_path=f'{name}_plots',
    )

    experiment.run()

    driver.save(f'{name}_driver.pkl')

def main() -> None:
    experiment(
        name='no_speeding_corner_c',
        step_no=3,
        alpha=0.68,
        corner='corner_c',
    )

    experiment(
        name='speeding_corner_c',
        step_no=5,
        alpha=0.3,
        corner='corner_c',
        use_speeding_policy=True,
        speeding_rate=0.2,
    )

    experiment(
        name='speeding_corner_c_no_is',
        step_no=3,
        alpha=0.68,
        corner='corner_c',
        use_speeding_policy=True,
        speeding_rate=0.2,
        use_importance_sampling=False
    )

if __name__ == '__main__':
    main()
