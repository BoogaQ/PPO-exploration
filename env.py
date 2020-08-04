from stable_baselines3.common.vec_env import SubprocVecEnv, VecFrameStack, VecTransposeImage, VecNormalize
from stable_baselines3.common.cmd_util import make_atari_env, make_vec_env


def make_env(env_id, 
            n_envs, 
            frame_stack = True,
            clip_reward = False,
            terminal_on_life_loss = False,
            monitor_dir = './log/monitors', 
            vec_env_cls = SubprocVecEnv):

    wrapper_kwargs = {'terminal_on_life_loss':terminal_on_life_loss, 'clip_reward':clip_reward}

    env = make_atari_env(env_id, n_envs, monitor_dir = monitor_dir, vec_env_cls = vec_env_cls, wrapper_kwargs = wrapper_kwargs)
    env = VecFrameStack(env, 4)
    env = VecTransposeImage(env)
    env = VecNormalize(env, norm_reward=False)
    return env
