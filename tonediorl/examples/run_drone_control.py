#!/usr/bin/env python3
from ruamel.yaml import YAML

#
import os
import io
import math
import argparse
import numpy as np
import torch

#
# from stable_baselines import logger

#
# from rpg_baselines.common.policies import MlpPolicy
# from rpg_baselines.ppo.ppo2 import PPO2
# from rpg_baselines.ppo.ppo2_test import test_model
from tonedio_baselines.envs import vec_env_wrapper as wrapper
from tonedio_baselines.envs.vec_env_wrapper import (
    ObsNormUpdateCallback,
    CheckpointCallbackWithRMS
)
import tonedio_baselines.common.util as U
#
from flightgym import QuadrotorEnv_v1

from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import VecMonitor
from stable_baselines3.common.callbacks import EvalCallback, CallbackList, CheckpointCallback

import wandb
from wandb.integration.sb3 import WandbCallback

"""
python run_drone_control.py --train 1 --use_obs_norm 1 --render 0 --wandb_run_name "test_run_1"


"""


def configure_random_seed(seed, env=None):
    if env is not None:
        env.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def parser():
    parser = argparse.ArgumentParser()
    parser.add_argument('--train', type=int, default=1,
                        help="To train new model or simply test pre-trained model")
    parser.add_argument('--render', type=int, default=0,
                        help="Enable Unity Render")
    parser.add_argument('--save_dir', type=str, default=os.path.dirname(os.path.realpath(__file__)),
                        help="Directory where to save the checkpoints and training metrics")
    parser.add_argument('--seed', type=int, default=0,
                        help="Random seed")
    parser.add_argument('-w', '--weight', type=str, 
    default='/home/heejun/projects/flightmare/tonediorl/examples/saved/2026-01-20-10-09-03/ppo_final.zip',
                        help='trained weight path')
    
    # eval freq, model_save_freq 모두 timestep 기준
    parser.add_argument('--total_timesteps', type=int, default=25_000_000,
                   help="Total training timesteps")
    parser.add_argument('--eval_freq', type=int, default=10_000,
                   help="Eval frequency (timesteps)")
    parser.add_argument('--n_eval_episodes', type=int, default=5,
                   help="Number of eval episodes")
    
    # wandb
    parser.add_argument('--wandb', type=int, default=1, help="Enable wandb logging")
    parser.add_argument('--wandb_project', type=str, default='flightmare_ppo', help="wandb project name")
    parser.add_argument('--wandb_run_name', type=str, default=None, help="wandb run name")
    parser.add_argument('--use_obs_norm', type=int, default=1, help="Use observation normalization (1=True, 0=False)")
    parser.add_argument('--checkpoint_freq', type=int, default=100_000, help="Checkpoint save frequency (timesteps)")
    return parser

def build_env(cfg_yaml_str, use_obs_norm=True):
    env = wrapper.FlightEnvVec(QuadrotorEnv_v1(cfg_yaml_str, False), use_obs_norm=use_obs_norm)
    env = VecMonitor(env)  # episode stats logging
    # VecMonitor는 episode가 끝날 때마다 정보 업데이트. 
    return env


def main():
    args = parser().parse_args()

    yaml = YAML()  # 기본 typ='rt' (RoundTrip)
    cfg_path = os.path.join(os.environ["FLIGHTMARE_PATH"], "flightlib/configs/vec_env.yaml")
    with open(cfg_path, "r") as f:
        cfg = yaml.load(f)

    if not args.train:
        cfg["env"]["num_envs"] = 1
        cfg["env"]["num_threads"] = 1

    cfg["env"]["render"] = "yes" if args.render else "no"

    # cfg를 YAML "문자열"로 다시 dump (QuadrotorEnv_v1이 이걸 받는 구조)
    stream = io.StringIO()
    yaml.dump(cfg, stream)
    cfg_yaml_str = stream.getvalue()

    # print(cfg_yaml_str)

    # main env
    use_obs_norm = bool(args.use_obs_norm)
    env = build_env(cfg_yaml_str, use_obs_norm=use_obs_norm)

    # set random seed
    configure_random_seed(args.seed, env=env)

    #
    if args.train:
        # save the configuration and other files
        rsg_root = os.path.dirname(os.path.abspath(__file__))
        log_dir = rsg_root + '/saved'
        saver = U.ConfigurationSaver(log_dir=log_dir)

        n_envs = env.num_envs
        n_steps = 250
        batch_size = n_steps * n_envs  # emulate nminibatches=1

 

        # wandb init
        wandb_run = None
        if args.wandb:
            wandb_run = wandb.init(
                project=args.wandb_project,
                name=args.wandb_run_name,
                config={
                    "algo": "PPO",
                    "seed": args.seed,
                    "gamma": 0.99,
                    "gae_lambda": 0.95,
                    "n_steps": n_steps,
                    "batch_size": batch_size,
                    "n_epochs": 10,
                    "clip_range": 0.2,
                    "learning_rate": 3e-4,
                    "ent_coef": 0.0,
                    "vf_coef": 0.5,
                    "max_grad_norm": 0.5,
                    "num_envs": n_envs,
                },
                sync_tensorboard=True,  # SB3 TB 로그 자동 동기화
                monitor_gym=False,      # 우리는 VecMonitor를 이미 씀
                save_code=True,
            )

        model = PPO(
            policy="MlpPolicy",
            policy_kwargs=dict(
                activation_fn=torch.nn.ReLU,
                net_arch=[dict(pi=[256, 256], vf=[512, 512])],
                log_std_init=-0.5,
            ),
            env=env,
            learning_rate=3e-4,
            n_steps=n_steps,
            batch_size=batch_size,
            n_epochs=10,          # PPO2 noptepochs
            gamma=0.99,
            gae_lambda=0.95,      # PPO2 lam
            clip_range=0.2,
            ent_coef=0.0,
            vf_coef=0.5,
            max_grad_norm=0.5,
            tensorboard_log=saver.data_dir,
            use_sde=False,
            verbose=1,
            device="cuda",
        )

        # ---- Eval env: force num_envs=1 for evaluation ----
        eval_cfg = cfg.copy()
        eval_cfg["env"]["num_envs"] = 1
        eval_cfg["env"]["num_threads"] = 1
        eval_stream = io.StringIO()
        yaml.dump(eval_cfg, eval_stream)
        eval_cfg_yaml_str = eval_stream.getvalue()
        eval_env = build_env(eval_cfg_yaml_str, use_obs_norm=use_obs_norm)

        # eval_callback = EvalCallback(
        #     eval_env,
        #     best_model_save_path=os.path.join(saver.data_dir, "best_model"),
        #     log_path=os.path.join(saver.data_dir, "eval_logs"),
        #     eval_freq=args.eval_freq,
        #     n_eval_episodes=args.n_eval_episodes,
        #     deterministic=True,
        #     render=False,
        #     verbose=1,
        #     warn=False,  # 평가 중 경고 메시지 억제
        # )

        # callback_list = [eval_callback]

        callback_list = []
        
        # Add observation normalization callbacks (only if normalization is enabled)
        if use_obs_norm:
            # Add observation normalization update callback
            # This automatically calls update_rms() at the end of each rollout
            callback_list.append(ObsNormUpdateCallback())
            
            # Add checkpoint callback that also saves normalization statistics
            checkpoint_callback = CheckpointCallbackWithRMS(
                save_freq=args.checkpoint_freq,
                save_path=os.path.join(saver.data_dir, "checkpoints"),
                name_prefix="ppo_model",
                verbose=1,
            )
            callback_list.append(checkpoint_callback)
        else:
            # Use regular CheckpointCallback when normalization is disabled
            from stable_baselines3.common.callbacks import CheckpointCallback
            checkpoint_callback = CheckpointCallback(
                save_freq=args.checkpoint_freq,
                save_path=os.path.join(saver.data_dir, "checkpoints"),
                name_prefix="ppo_model",
                verbose=1,
            )
            callback_list.append(checkpoint_callback)

        if args.wandb:
            callback_list.append(WandbCallback(
                gradient_save_freq=0,
                model_save_freq=100_000,
                model_save_path=os.path.join(saver.data_dir, f"wandb_{wandb_run.id}"),
                verbose=2,
            ))

        callbacks = CallbackList(callback_list)

        try:
            model.learn(
                total_timesteps=int(args.total_timesteps),
                tb_log_name="PPO_Flightmare",
                callback=callbacks,
                progress_bar=True,
            )

            model.save(os.path.join(saver.data_dir, "ppo_final"))
        finally:
            # 환경 정리
            eval_env.close()
            env.close()

    else:
        # Test mode (simple loop)
        model = PPO.load(args.weight, env=env, device="auto")
        
        # Disable truncation for testing - allow episodes to run until crash or manual stop
        # This allows testing how long the model can hover without time limit
        env.wrapper.setTruncationEnabled(False)
        print(f"[Test Mode] Truncation disabled - episodes will run until crash or manual stop")
        
        max_ep_length = 1000  # Set a large limit for Python loop (C++ truncation is disabled)
        num_rollouts = 5

        if args.render:
            env.connectUnity()

        for n_roll in range(num_rollouts):
            print(f"\n=== Rollout {n_roll} ===")

            # rollout buffers (optional)
            pos, euler, dpos, deuler, actions = [], [], [], [], []

            obs = env.reset()
            done = np.array([False])
            ep_len = 0

            while not (done[0] or ep_len >= max_ep_length):
                # policy inference
                act, _ = model.predict(obs, deterministic=True)

                # env step
                obs, reward, done, info = env.step(act)

                ep_len += 1

                # ---- logging (obs shape: [1, 12]) ----
                pos.append(obs[0, 0:3].tolist())
                euler.append(obs[0, 3:6].tolist())
                dpos.append(obs[0, 6:9].tolist())
                deuler.append(obs[0, 9:12].tolist())
                actions.append(act[0].tolist())

            print(f"Rollout {n_roll} finished | length = {ep_len}")

        if args.render:
            env.disconnectUnity()


if __name__ == "__main__":
    main()
