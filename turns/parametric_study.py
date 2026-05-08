from problem import Corner, Experiment, Environment
from problem import OffPolicyNStepSarsaDriver, RandomDriver

from itertools import product
from multiprocessing import Pool
import os
from tqdm import tqdm
import parse

import numpy as np

MAX_LEARNING_STEPS = 2000

param_grid = {
    "step_no": [2, 3, 4, 5, 6],
    "alpha": np.linspace(0.01, 0.99, 20),
}

def run_experiment(params) -> None:
    step_no, alpha = params

    result_path = f'parametric_study_results/step_no_{step_no}/alpha_{alpha:.2f}'

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
                name='corner_c'
            ),
            steering_fail_chance=0.01,
        ),
        driver=driver,
        number_of_episodes=30000,
        plots_path=f'{result_path}/plots',
        drawing_frequency=1000,
        show_progress=False,

        eval_active=True,
        eval_frequency=10,
        eval_episodes=10,

        save_driver=False,
    )

    experiment.run()

    with open(f'{result_path}/eval_penalties.txt', 'w') as f:
        for penalty in experiment.eval_penalties:
            f.write(f'{penalty}\n')

    with open(f'{result_path}/penalties.txt', 'w') as f:
        for penalty in experiment.penalties:
            f.write(f'{penalty}\n')

    driver.save(f'{result_path}/driver.pkl')

def main() -> None:
    keep_old_results = True
    param_space = list(product(*param_grid.values()))

    os.makedirs('parametric_study_results', exist_ok=True)

    existent_results = []
    
    for step_no, alpha in param_space:
        os.makedirs(f'parametric_study_results/step_no_{step_no}/alpha_{alpha:.2f}/plots', exist_ok=True)

        if os.path.exists(f'parametric_study_results/step_no_{step_no}/alpha_{alpha:.2f}/eval_penalties.txt') and keep_old_results:
            existent_results.append((step_no, alpha))

    param_space_reduced = [params for params in param_space if params not in existent_results]
    
    with Pool(16) as pool:
        for _ in tqdm(pool.imap_unordered(run_experiment, param_space_reduced), total=len(param_space_reduced)):
            pass

if __name__ == '__main__':
    main()
