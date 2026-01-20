import numpy as np
from gymnasium import spaces
from stable_baselines3.common.vec_env import VecEnv


class FlightEnvVec(VecEnv):
    """
    SB3-compatible VecEnv wrapper for Flightmare's C++ VecEnv binding (flightgym.QuadrotorEnv_v1).

    The underlying C++ API is assumed to be:
      - getObsDim(), getActDim(), getNumOfEnvs()
      - getExtraInfoNames()
      - reset(obs_out)
      - step(action_in, obs_out, rew_out, done_out, extra_out)
      - setSeed(seed)
      - close()
      - connectUnity(), disconnectUnity()  (optional)
      - curriculumUpdate()                (optional)
    """

    def __init__(self, impl):
        self.wrapper = impl

        self.num_obs = int(self.wrapper.getObsDim())
        self.num_acts = int(self.wrapper.getActDim())
        self._num_envs = int(self.wrapper.getNumOfEnvs())

        # SB3 expects per-env spaces (not batched)
        self._observation_space = spaces.Box(
            low=-np.inf * np.ones(self.num_obs, dtype=np.float32),
            high=np.inf * np.ones(self.num_obs, dtype=np.float32),
            dtype=np.float32,
        )
        self._action_space = spaces.Box(
            low=-1.0 * np.ones(self.num_acts, dtype=np.float32),
            high=1.0 * np.ones(self.num_acts, dtype=np.float32),
            dtype=np.float32,
        )

        self._actions = None
        


        # Buffers (batched)
        self._observation = np.zeros((self._num_envs, self.num_obs), dtype=np.float32)
        self._reward = np.zeros((self._num_envs,), dtype=np.float32)
        self._done = np.zeros((self._num_envs,), dtype=bool)

        self._extraInfoNames = list(self.wrapper.getExtraInfoNames())
        self._extraInfo = np.zeros((self._num_envs, len(self._extraInfoNames)), dtype=np.float32)

        # Episode bookkeeping (SB3 uses info["episode"] convention)
        self._ep_rewards = [[] for _ in range(self._num_envs)]

        self.max_episode_steps = 300

        print(f"[FlightEnvVecSB3] num_envs={self._num_envs}, obs_dim={self.num_obs}, act_dim={self.num_acts}")

    def seed(self, seed=0):
        self.wrapper.setSeed(seed)

    # def step(self, action):
    #     self.wrapper.step(action, self._observation,
    #                       self._reward, self._done, self._extraInfo)

    #     if len(self._extraInfoNames) is not 0:
    #         info = [{'extra_info': {
    #             self._extraInfoNames[j]: self._extraInfo[i, j] for j in range(0, len(self._extraInfoNames))
    #         }} for i in range(self.num_envs)]
    #     else:
    #         info = [{} for i in range(self.num_envs)]

    #     for i in range(self.num_envs):
    #         self.rewards[i].append(self._reward[i])
    #         if self._done[i]:
    #             eprew = sum(self.rewards[i])
    #             eplen = len(self.rewards[i])
    #             epinfo = {"r": eprew, "l": eplen}
    #             info[i]['episode'] = epinfo
    #             self.rewards[i].clear()

    #     return self._observation.copy(), self._reward.copy(), \
    #         self._done.copy(), info.copy()

    def stepUnity(self, action, send_id):
        receive_id = self.wrapper.stepUnity(action, self._observation,
                                            self._reward, self._done, self._extraInfo, send_id)

        return receive_id

    def sample_actions(self):
        actions = []
        for i in range(self.num_envs):
            action = self.action_space.sample().tolist()
            actions.append(action)
        return np.asarray(actions, dtype=np.float32)

    def reset(self):
        self._reward[:] = 0.0
        self._done[:] = False
        # Flightmare fills the provided obs buffer
        self.wrapper.reset(self._observation)
        return self._observation.copy()

    # def reset_and_update_info(self):
    #     return self.reset(), self._update_epi_info()

    # def _update_epi_info(self):
    #     info = [{} for _ in range(self.num_envs)]

    #     for i in range(self.num_envs):
    #         eprew = sum(self.rewards[i])
    #         eplen = len(self.rewards[i])
    #         epinfo = {"r": eprew, "l": eplen}
    #         info[i]['episode'] = epinfo
    #         self.rewards[i].clear()
    #     return info

    def render(self, mode='human'):
        raise RuntimeError('This method is not implemented')

    def close(self):
        self.wrapper.close()

    def connectUnity(self):
        self.wrapper.connectUnity()

    def disconnectUnity(self):
        self.wrapper.disconnectUnity()

    @property
    def num_envs(self):
        return self.wrapper.getNumOfEnvs()

    @property
    def observation_space(self):
        return self._observation_space

    @property
    def action_space(self):
        return self._action_space

    @property
    def extra_info_names(self):
        return self._extraInfoNames

    def start_recording_video(self, file_name):
        raise RuntimeError('This method is not implemented')

    def stop_recording_video(self):
        raise RuntimeError('This method is not implemented')

    def curriculum_callback(self):
        self.wrapper.curriculumUpdate()

    def step_async(self, actions):
        # SB3 may provide list, np.ndarray, or torch tensor converted to np by policy
        actions = np.asarray(actions, dtype=np.float32)

        # Expected shape: (num_envs, act_dim)
        if actions.ndim == 1:
            # If user accidentally provides (act_dim,), broadcast to all envs
            if actions.shape[0] != self.num_acts:
                raise ValueError(f"Invalid action shape {actions.shape}, expected ({self._num_envs}, {self.num_acts}) or ({self.num_acts},)")
            actions = np.tile(actions[None, :], (self._num_envs, 1))

        if actions.shape != (self._num_envs, self.num_acts):
            raise ValueError(f"Invalid action shape {actions.shape}, expected ({self._num_envs}, {self.num_acts})")

        self._actions = actions

    def step_wait(self):
        if self._actions is None:
            raise RuntimeError("step_wait() called before step_async().")

        # C++ fills buffers in-place
        self.wrapper.step(self._actions, self._observation, self._reward, self._done, self._extraInfo)

        # infos: extra_info만 넣어줌 (episode는 VecMonitor가 처리)
        if len(self._extraInfoNames) > 0:
            infos = [
                {"extra_info": {self._extraInfoNames[j]: float(self._extraInfo[i, j])
                                for j in range(len(self._extraInfoNames))}}
                for i in range(self._num_envs)
            ]
        else:
            infos = [{} for _ in range(self._num_envs)]

        obs = self._observation.copy()
        rews = self._reward.copy()
        dones = self._done.copy()

        self._actions = None

        return obs, rews, dones, infos
    
    def get_attr(self, attr_name, indices=None):
        """
        Return attribute from vectorized environment.
        :param attr_name: (str) The name of the attribute whose value to return
        :param indices: (list,int) Indices of envs to get attribute from
        :return: (list) List of values of 'attr_name' in all environments
        """
        indices = self._get_indices(indices)
        if not hasattr(self, attr_name):
            # SB3 expects a list with length=len(indices)
            return [None for _ in indices]
        value = getattr(self, attr_name)
        return [value for _ in indices]

    def set_attr(self, attr_name, value, indices=None):
        """
        Set attribute inside vectorized environments.
        :param attr_name: (str) The name of attribute to assign new value
        :param value: (obj) Value to assign to `attr_name`
        :param indices: (list,int) Indices of envs to assign value
        :return: (NoneType)
        """
        _ = self._get_indices(indices)
        setattr(self, attr_name, value)

    def env_method(self, method_name, *method_args, indices=None, **method_kwargs):
        """
        Call instance methods of vectorized environments.
        :param method_name: (str) The name of the environment method to invoke.
        :param indices: (list,int) Indices of envs whose method to call
        :param method_args: (tuple) Any positional arguments to provide in the call
        :param method_kwargs: (dict) Any keyword arguments to provide in the call
        :return: (list) List of items returned by the environment's method call
        """
        indices = self._get_indices(indices)

        # Prefer wrapper methods first
        if hasattr(self, method_name):
            method = getattr(self, method_name)
            result = method(*method_args, **method_kwargs)
            return [result for _ in indices]

        # Fallback to underlying C++ env
        if hasattr(self.wrapper, method_name):
            method = getattr(self.wrapper, method_name)
            result = method(*method_args, **method_kwargs)
            return [result for _ in indices]

        raise AttributeError(f"Method '{method_name}' not found in wrapper or underlying env.")
    
    def env_is_wrapped(self, wrapper_class, indices=None):
        """
        SB3 expects this to exist on VecEnv.
        We do not use Gymnasium wrappers here, so always return False.
        """
        indices = self._get_indices(indices) if hasattr(self, "_get_indices") else (
            list(range(self.num_envs)) if indices is None else ([indices] if isinstance(indices, int) else list(indices))
        )
        return [False for _ in indices]

    def get_images(self):
        """
        Optional method for SB3 VecEnv API.
        Flightmare rendering is handled by Unity; return empty list to satisfy interface.
        """
        return [None for _ in range(self.num_envs)]
