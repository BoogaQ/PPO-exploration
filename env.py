from stable_baselines3.common.vec_env import SubprocVecEnv, VecFrameStack, VecTransposeImage, VecNormalize
from stable_baselines3.common.cmd_util import make_atari_env, make_vec_env
import multiprocessing
import mujoco_py


def make_env(env_id, 
            n_envs,
            vec_env_cls = SubprocVecEnv):
    env = make_vec_env(env_id, n_envs, vec_env_cls = SubprocVecEnv)
    env = VecNormalize(env, norm_reward = True)
    return env
