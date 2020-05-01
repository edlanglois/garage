"""A regression test for automatic benchmarking garage-PyTorch-PPO."""
import gym
import torch

from garage import wrap_experiment
from garage.envs import normalize
from garage.experiment import deterministic, LocalRunner
from garage.np.baselines import LinearFeatureBaseline
from garage.tf.envs import TfEnv
from garage.torch.algos import PPO as PyTorch_PPO
from garage.torch.policies import GaussianMLPPolicy as PyTorch_GMP

hyper_parameters = {
    'n_epochs': 800,
    'max_path_length': 128,
    'batch_size': 1024,
}


@wrap_experiment
def ppo_garage_pytorch(ctxt, env_id, seed):
    """Create garage PyTorch PPO model and training.

    Args:
        ctxt (garage.experiment.ExperimentContext): The experiment
            configuration used by LocalRunner to create the
            snapshotter.
        env_id (str): Environment id of the task.
        seed (int): Random positive integer for the trial.

    """
    deterministic.set_seed(seed)

    runner = LocalRunner(ctxt)

    env = TfEnv(normalize(gym.make(env_id)))

    policy = PyTorch_GMP(env.spec,
                         hidden_sizes=(32, 32),
                         hidden_nonlinearity=torch.tanh,
                         output_nonlinearity=None)

    value_functions = LinearFeatureBaseline(env_spec=env.spec)

    algo = PyTorch_PPO(env_spec=env.spec,
                       policy=policy,
                       value_function=value_functions,
                       optimizer=torch.optim.Adam,
                       policy_lr=3e-4,
                       max_path_length=hyper_parameters['max_path_length'],
                       discount=0.99,
                       gae_lambda=0.95,
                       center_adv=True,
                       lr_clip_range=0.2,
                       minibatch_size=128,
                       max_optimization_epochs=10)

    runner.setup(algo, env)
    runner.train(n_epochs=hyper_parameters['n_epochs'],
                 batch_size=hyper_parameters['batch_size'])
