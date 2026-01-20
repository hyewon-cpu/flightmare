import warnings
from itertools import zip_longest
from abc import ABC, abstractmethod

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from gymnasium.spaces import Discrete, Box

from tonedio_baselines.common.distributions import make_proba_dist_type, CategoricalProbabilityDistribution, \
    MultiCategoricalProbabilityDistribution, DiagGaussianProbabilityDistribution, BernoulliProbabilityDistribution


# PyTorch helper functions
def batch_to_seq(tensor, n_env, n_steps):
    """Convert batch tensor to sequence format for RNN"""
    if not isinstance(tensor, torch.Tensor):
        tensor = torch.as_tensor(tensor)
    batch_size = tensor.shape[0]
    assert batch_size == n_env * n_steps, f"batch_size {batch_size} != n_env * n_steps {n_env * n_steps}"
    return tensor.view(n_env, n_steps, *tensor.shape[1:])


def seq_to_batch(sequence):
    """Convert sequence tensor back to batch format"""
    if not isinstance(sequence, torch.Tensor):
        sequence = torch.as_tensor(sequence)
    n_env, n_steps = sequence.shape[:2]
    return sequence.reshape(n_env * n_steps, *sequence.shape[2:])


def observation_input(ob_space, n_batch=None, scale=False):
    """
    Create observation input placeholders (PyTorch version)
    
    :param ob_space: (gymnasium.Space) observation space
    :param n_batch: (int) batch size
    :param scale: (bool) whether to scale observations
    :return: (torch.Tensor, torch.Tensor) observation placeholder and processed observation
    """
    if n_batch is None:
        obs_shape = ob_space.shape
    else:
        obs_shape = (n_batch,) + ob_space.shape
    
    obs_ph = torch.zeros(obs_shape, dtype=torch.float32)
    processed_obs = obs_ph
    
    if scale:
        if isinstance(ob_space, Box):
            # Scale to [0, 1] if needed
            low = torch.as_tensor(ob_space.low, dtype=torch.float32)
            high = torch.as_tensor(ob_space.high, dtype=torch.float32)
            if low.min() < 0 or high.max() > 1:
                processed_obs = (obs_ph - low) / (high - low + 1e-8)
    
    return obs_ph, processed_obs


class NatureCNN(nn.Module):
    """CNN from Nature paper (PyTorch version)"""
    def __init__(self, **kwargs):
        super(NatureCNN, self).__init__()
        self.conv1 = nn.Conv2d(3, 32, kernel_size=8, stride=4)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=4, stride=2)
        self.conv3 = nn.Conv2d(64, 64, kernel_size=3, stride=1)
        self.fc1 = nn.Linear(64 * 7 * 7, 512)  # Assuming 84x84 input
        
        # Initialize weights
        for m in self.modules():
            if isinstance(m, (nn.Conv2d, nn.Linear)):
                nn.init.orthogonal_(m.weight, gain=np.sqrt(2))
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
    
    def forward(self, x):
        x = F.relu(self.conv1(x))
        x = F.relu(self.conv2(x))
        x = F.relu(self.conv3(x))
        x = x.view(x.size(0), -1)  # Flatten
        x = F.relu(self.fc1(x))
        return x


def nature_cnn(scaled_images, **kwargs):
    """
    CNN from Nature paper (PyTorch version).

    :param scaled_images: (torch.Tensor) Image input tensor
    :param kwargs: (dict) Extra keywords parameters (not used in PyTorch version)
    :return: (torch.Tensor) The CNN output layer
    """
    if not hasattr(nature_cnn, '_module'):
        nature_cnn._module = NatureCNN(**kwargs)
    if not isinstance(scaled_images, torch.Tensor):
        scaled_images = torch.as_tensor(scaled_images)
    return nature_cnn._module(scaled_images)


def mlp_extractor(flat_observations, net_arch, act_fun):
    """
    Constructs an MLP that receives observations as an input and outputs a latent representation for the policy and
    a value network. The ``net_arch`` parameter allows to specify the amount and size of the hidden layers and how many
    of them are shared between the policy network and the value network. It is assumed to be a list with the following
    structure:

    1. An arbitrary length (zero allowed) number of integers each specifying the number of units in a shared layer.
       If the number of ints is zero, there will be no shared layers.
    2. An optional dict, to specify the following non-shared layers for the value network and the policy network.
       It is formatted like ``dict(vf=[<value layer sizes>], pi=[<policy layer sizes>])``.
       If it is missing any of the keys (pi or vf), no non-shared layers (empty list) is assumed.

    For example to construct a network with one shared layer of size 55 followed by two non-shared layers for the value
    network of size 255 and a single non-shared layer of size 128 for the policy network, the following layers_spec
    would be used: ``[55, dict(vf=[255, 255], pi=[128])]``. A simple shared network topology with two layers of size 128
    would be specified as [128, 128].

    :param flat_observations: (torch.Tensor) The observations to base policy and value function on.
    :param net_arch: ([int or dict]) The specification of the policy and value networks.
        See above for details on its formatting.
    :param act_fun: (callable) The activation function to use for the networks (e.g., F.relu, torch.tanh).
    :return: (torch.Tensor, torch.Tensor) latent_policy, latent_value of the specified network.
        If all layers are shared, then ``latent_policy == latent_value``
    """
    if not isinstance(flat_observations, torch.Tensor):
        flat_observations = torch.as_tensor(flat_observations, dtype=torch.float32)
    
    latent = flat_observations
    policy_only_layers = []  # Layer sizes of the network that only belongs to the policy network
    value_only_layers = []  # Layer sizes of the network that only belongs to the value network
    
    # Create a unique key for caching layers
    cache_key = str(id(net_arch))
    if not hasattr(mlp_extractor, '_layers_cache'):
        mlp_extractor._layers_cache = {}

    # Iterate through the shared layers and build the shared parts of the network
    for idx, layer in enumerate(net_arch):
        if isinstance(layer, int):  # Check that this is a shared layer
            layer_size = layer
            layer_key = f"{cache_key}_shared_fc{idx}"
            if layer_key not in mlp_extractor._layers_cache:
                linear_layer = nn.Linear(latent.shape[-1], layer_size)
                nn.init.orthogonal_(linear_layer.weight, gain=np.sqrt(2))
                nn.init.constant_(linear_layer.bias, 0)
                mlp_extractor._layers_cache[layer_key] = linear_layer
            else:
                linear_layer = mlp_extractor._layers_cache[layer_key]
            latent = act_fun(linear_layer(latent))
        else:
            assert isinstance(layer, dict), "Error: the net_arch list can only contain ints and dicts"
            if 'pi' in layer:
                assert isinstance(layer['pi'], list), "Error: net_arch[-1]['pi'] must contain a list of integers."
                policy_only_layers = layer['pi']

            if 'vf' in layer:
                assert isinstance(layer['vf'], list), "Error: net_arch[-1]['vf'] must contain a list of integers."
                value_only_layers = layer['vf']
            break  # From here on the network splits up in policy and value network

    # Build the non-shared part of the network
    latent_policy = latent
    latent_value = latent
    for idx, (pi_layer_size, vf_layer_size) in enumerate(zip_longest(policy_only_layers, value_only_layers)):
        if pi_layer_size is not None:
            assert isinstance(pi_layer_size, int), "Error: net_arch[-1]['pi'] must only contain integers."
            layer_key = f"{cache_key}_pi_fc{idx}"
            if layer_key not in mlp_extractor._layers_cache:
                linear_layer = nn.Linear(latent_policy.shape[-1], pi_layer_size)
                nn.init.orthogonal_(linear_layer.weight, gain=np.sqrt(2))
                nn.init.constant_(linear_layer.bias, 0)
                mlp_extractor._layers_cache[layer_key] = linear_layer
            else:
                linear_layer = mlp_extractor._layers_cache[layer_key]
            latent_policy = act_fun(linear_layer(latent_policy))

        if vf_layer_size is not None:
            assert isinstance(vf_layer_size, int), "Error: net_arch[-1]['vf'] must only contain integers."
            layer_key = f"{cache_key}_vf_fc{idx}"
            if layer_key not in mlp_extractor._layers_cache:
                linear_layer = nn.Linear(latent_value.shape[-1], vf_layer_size)
                nn.init.orthogonal_(linear_layer.weight, gain=np.sqrt(2))
                nn.init.constant_(linear_layer.bias, 0)
                mlp_extractor._layers_cache[layer_key] = linear_layer
            else:
                linear_layer = mlp_extractor._layers_cache[layer_key]
            latent_value = act_fun(linear_layer(latent_value))

    return latent_policy, latent_value


class BasePolicy(ABC):
    """
    The base policy object (PyTorch version)

    :param device: (torch.device or str) Device to run on (e.g., 'cpu', 'cuda')
    :param ob_space: (gymnasium.Space) The observation space of the environment
    :param ac_space: (gymnasium.Space) The action space of the environment
    :param n_env: (int) The number of environments to run
    :param n_steps: (int) The number of steps to run for each environment
    :param n_batch: (int) The number of batches to run (n_envs * n_steps)
    :param reuse: (bool) If the policy is reusable or not (not used in PyTorch)
    :param scale: (bool) whether or not to scale the input
    :param obs_phs: (torch.Tensor, torch.Tensor) a tuple containing an override for observation tensor
        and the processed observation tensor respectively
    :param add_action_ph: (bool) whether or not to create an action placeholder (not used in PyTorch)
    """

    recurrent = False

    def __init__(self, device=None, ob_space=None, ac_space=None, n_env=1, n_steps=1, n_batch=None, reuse=False, scale=False,
                 obs_phs=None, add_action_ph=False):
        self.n_env = n_env
        self.n_steps = n_steps
        self.n_batch = n_batch if n_batch is not None else n_env * n_steps
        
        # Set device
        if device is None:
            self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        else:
            self.device = torch.device(device) if isinstance(device, str) else device
        
        if ob_space is not None and ac_space is not None:
            if obs_phs is None:
                self._obs_ph, self._processed_obs = observation_input(ob_space, self.n_batch, scale=scale)
                self._obs_ph = self._obs_ph.to(self.device)
                self._processed_obs = self._processed_obs.to(self.device)
            else:
                self._obs_ph, self._processed_obs = obs_phs
                if isinstance(self._obs_ph, torch.Tensor):
                    self._obs_ph = self._obs_ph.to(self.device)
                if isinstance(self._processed_obs, torch.Tensor):
                    self._processed_obs = self._processed_obs.to(self.device)

            self._action_ph = None
            if add_action_ph:
                action_shape = (self.n_batch,) + ac_space.shape
                self._action_ph = torch.zeros(action_shape, dtype=torch.float32, device=self.device)
        
        self.reuse = reuse
        self.ob_space = ob_space
        self.ac_space = ac_space

    @property
    def is_discrete(self):
        """bool: is action space discrete."""
        return isinstance(self.ac_space, Discrete)

    @property
    def initial_state(self):
        """
        The initial state of the policy. For feedforward policies, None. For a recurrent policy,
        a NumPy array of shape (self.n_env, ) + state_shape.
        """
        assert not self.recurrent, "When using recurrent policies, you must overwrite `initial_state()` method"
        return None

    @property
    def obs_ph(self):
        """torch.Tensor: placeholder for observations, shape (self.n_batch, ) + self.ob_space.shape."""
        return self._obs_ph

    @property
    def processed_obs(self):
        """torch.Tensor: processed observations, shape (self.n_batch, ) + self.ob_space.shape.

        The form of processing depends on the type of the observation space, and the parameters
        whether scale is passed to the constructor; see observation_input for more information."""
        return self._processed_obs

    @property
    def action_ph(self):
        """torch.Tensor: placeholder for actions, shape (self.n_batch, ) + self.ac_space.shape."""
        return self._action_ph

    @staticmethod
    def _kwargs_check(feature_extraction, kwargs):
        """
        Ensure that the user is not passing wrong keywords
        when using policy_kwargs.

        :param feature_extraction: (str)
        :param kwargs: (dict)
        """
        # When using policy_kwargs parameter on model creation,
        # all keywords arguments must be consumed by the policy constructor except
        # the ones for the cnn_extractor network (cf nature_cnn()), where the keywords arguments
        # are not passed explicitly (using **kwargs to forward the arguments)
        # that's why there should be not kwargs left when using the mlp_extractor
        # (in that case the keywords arguments are passed explicitly)
        if feature_extraction == 'mlp' and len(kwargs) > 0:
            raise ValueError("Unknown keywords for policy: {}".format(kwargs))

    @abstractmethod
    def step(self, obs, state=None, mask=None):
        """
        Returns the policy for a single step

        :param obs: ([float] or [int]) The current observation of the environment
        :param state: ([float]) The last states (used in recurrent policies)
        :param mask: ([float]) The last masks (used in recurrent policies)
        :return: ([float], [float], [float], [float]) actions, values, states, neglogp
        """
        raise NotImplementedError

    @abstractmethod
    def proba_step(self, obs, state=None, mask=None):
        """
        Returns the action probability for a single step

        :param obs: ([float] or [int]) The current observation of the environment
        :param state: ([float]) The last states (used in recurrent policies)
        :param mask: ([float]) The last masks (used in recurrent policies)
        :return: ([float]) the action probability
        """
        raise NotImplementedError


class ActorCriticPolicy(BasePolicy):
    """
    Policy object that implements actor critic (PyTorch version)

    :param device: (torch.device or str) Device to run on
    :param ob_space: (gymnasium.Space) The observation space of the environment
    :param ac_space: (gymnasium.Space) The action space of the environment
    :param n_env: (int) The number of environments to run
    :param n_steps: (int) The number of steps to run for each environment
    :param n_batch: (int) The number of batch to run (n_envs * n_steps)
    :param reuse: (bool) If the policy is reusable or not (not used in PyTorch)
    :param scale: (bool) whether or not to scale the input
    """

    def __init__(self, device=None, ob_space=None, ac_space=None, n_env=1, n_steps=1, n_batch=None, reuse=False, scale=False):
        super(ActorCriticPolicy, self).__init__(device=device, ob_space=ob_space, ac_space=ac_space, n_env=n_env, n_steps=n_steps, n_batch=n_batch, reuse=reuse,
                                                scale=scale)
        self._pdtype = make_proba_dist_type(ac_space)
        self._policy = None
        self._proba_distribution = None
        self._value_fn = None
        self._action = None
        self._deterministic_action = None

    def _setup_init(self):
        """Sets up the distributions, actions, and value."""
        assert self.policy is not None and self.proba_distribution is not None and self.value_fn is not None
        self._action = self.proba_distribution.sample()
        self._tanh_action = torch.tanh(self._action)
        self._deterministic_action = self.proba_distribution.mode()
        self._tanh_deterministic_action = torch.tanh(self._deterministic_action)
        self._neglogp = self.proba_distribution.neglogp(self.action)
        self._tanh_neglogp = self.proba_distribution.tanh_neglogp(self.action)
        if isinstance(self.proba_distribution, CategoricalProbabilityDistribution):
            self._policy_proba = F.softmax(self.policy, dim=-1)
        elif isinstance(self.proba_distribution, DiagGaussianProbabilityDistribution):
            # changed by Yunlong.. add torch.tanh as the end of the last layer.
            self._policy_proba = [self.proba_distribution.mean, self.proba_distribution.std]
        elif isinstance(self.proba_distribution, BernoulliProbabilityDistribution):
            self._policy_proba = torch.sigmoid(self.policy)
        elif isinstance(self.proba_distribution, MultiCategoricalProbabilityDistribution):
            self._policy_proba = [F.softmax(categorical.flatparam(), dim=-1)
                                 for categorical in self.proba_distribution.categoricals]
        else:
            self._policy_proba = []  # it will return nothing, as it is not implemented
        if self.value_fn.ndim > 1:
            self._value_flat = self.value_fn[:, 0]
        else:
            self._value_flat = self.value_fn

    @property
    def pdtype(self):
        """ProbabilityDistributionType: type of the distribution for stochastic actions."""
        return self._pdtype

    @property
    def policy(self):
        """torch.Tensor: policy output, e.g. logits."""
        return self._policy

    @property
    def proba_distribution(self):
        """ProbabilityDistribution: distribution of stochastic actions."""
        return self._proba_distribution

    @property
    def value_fn(self):
        """torch.Tensor: value estimate, of shape (self.n_batch, 1)"""
        return self._value_fn

    @property
    def value_flat(self):
        """torch.Tensor: value estimate, of shape (self.n_batch, )"""
        return self._value_flat

    @property
    def action(self):
        """torch.Tensor: stochastic action, of shape (self.n_batch, ) + self.ac_space.shape."""
        return self._action
    
    @property
    def tanh_action(self):
        """torch.Tensor: stochastic action, of shape (self.n_batch, ) + self.ac_space.shape."""
        return self._tanh_action
    
    @property
    def tanh_deterministic_action(self):
        """torch.Tensor: deterministic action, of shape (self.n_batch, ) + self.ac_space.shape."""
        return self._tanh_deterministic_action

    @property
    def deterministic_action(self):
        """torch.Tensor: deterministic action, of shape (self.n_batch, ) + self.ac_space.shape."""
        return self._deterministic_action

    @property
    def neglogp(self):
        """torch.Tensor: negative log likelihood of the action sampled by self.action."""
        return self._neglogp
      
    @property
    def tanh_neglogp(self):
        # tanh_neglogp = torch.sum(torch.log(1 - self.tanh_action**2 + 1e-6), dim=1) - self._neglogp
        return self._tanh_neglogp

    @property
    def policy_proba(self):
        """torch.Tensor: parameters of the probability distribution. Depends on pdtype."""
        return self._policy_proba

    @abstractmethod
    def step(self, obs, state=None, mask=None, deterministic=False):
        """
        Returns the policy for a single step

        :param obs: ([float] or [int]) The current observation of the environment
        :param state: ([float]) The last states (used in recurrent policies)
        :param mask: ([float]) The last masks (used in recurrent policies)
        :param deterministic: (bool) Whether or not to return deterministic actions.
        :return: ([float], [float], [float], [float]) actions, values, states, neglogp
        """
        raise NotImplementedError

    @abstractmethod
    def value(self, obs, state=None, mask=None):
        """
        Returns the value for a single step

        :param obs: ([float] or [int]) The current observation of the environment
        :param state: ([float]) The last states (used in recurrent policies)
        :param mask: ([float]) The last masks (used in recurrent policies)
        :return: ([float]) The associated value of the action
        """
        raise NotImplementedError


class RecurrentActorCriticPolicy(ActorCriticPolicy):
    """
    Actor critic policy object uses a previous state in the computation for the current step (PyTorch version).
    NOTE: this class is not limited to recurrent neural network policies,
    see https://github.com/hill-a/stable-baselines/issues/241

    :param device: (torch.device or str) Device to run on
    :param ob_space: (gymnasium.Space) The observation space of the environment
    :param ac_space: (gymnasium.Space) The action space of the environment
    :param n_env: (int) The number of environments to run
    :param n_steps: (int) The number of steps to run for each environment
    :param n_batch: (int) The number of batch to run (n_envs * n_steps)
    :param state_shape: (tuple<int>) shape of the per-environment state space.
    :param reuse: (bool) If the policy is reusable or not (not used in PyTorch)
    :param scale: (bool) whether or not to scale the input
    """

    recurrent = True

    def __init__(self, device=None, ob_space=None, ac_space=None, n_env=1, n_steps=1, n_batch=None,
                 state_shape=None, reuse=False, scale=False):
        super(RecurrentActorCriticPolicy, self).__init__(device=device, ob_space=ob_space, ac_space=ac_space, n_env=n_env, n_steps=n_steps,
                                                         n_batch=n_batch, reuse=reuse, scale=scale)

        # Create placeholders for dones and states
        self._dones_ph = torch.zeros((self.n_batch,), dtype=torch.float32, device=self.device)  # (done t-1)
        if state_shape is not None:
            state_ph_shape = (self.n_env,) + tuple(state_shape)
            self._states_ph = torch.zeros(state_ph_shape, dtype=torch.float32, device=self.device)
            initial_state_shape = (self.n_env,) + tuple(state_shape)
            self._initial_state = np.zeros(initial_state_shape, dtype=np.float32)
        else:
            self._states_ph = None
            self._initial_state = None

    @property
    def initial_state(self):
        return self._initial_state

    @property
    def dones_ph(self):
        """torch.Tensor: placeholder for whether episode has terminated (done), shape (self.n_batch, ).
        Internally used to reset the state before the next episode starts."""
        return self._dones_ph

    @property
    def states_ph(self):
        """torch.Tensor: placeholder for states, shape (self.n_env, ) + state_shape."""
        return self._states_ph

    @abstractmethod
    def value(self, obs, state=None, mask=None):
        """
        Cf base class doc.
        """
        raise NotImplementedError


class LstmPolicy(RecurrentActorCriticPolicy):
    """
    Policy object that implements actor critic, using LSTMs.

    :param sess: (TensorFlow session) The current TensorFlow session
    :param ob_space: (Gym Space) The observation space of the environment
    :param ac_space: (Gym Space) The action space of the environment
    :param n_env: (int) The number of environments to run
    :param n_steps: (int) The number of steps to run for each environment
    :param n_batch: (int) The number of batch to run (n_envs * n_steps)
    :param n_lstm: (int) The number of LSTM cells (for recurrent policies)
    :param reuse: (bool) If the policy is reusable or not
    :param layers: ([int]) The size of the Neural network before the LSTM layer  (if None, default to [64, 64])
    :param net_arch: (list) Specification of the actor-critic policy network architecture. Notation similar to the
        format described in mlp_extractor but with additional support for a 'lstm' entry in the shared network part.
    :param act_fun: (tf.func) the activation function to use in the neural network.
    :param cnn_extractor: (function (TensorFlow Tensor, ``**kwargs``): (TensorFlow Tensor)) the CNN feature extraction
    :param layer_norm: (bool) Whether or not to use layer normalizing LSTMs
    :param feature_extraction: (str) The feature extraction type ("cnn" or "mlp")
    :param kwargs: (dict) Extra keyword arguments for the nature CNN feature extraction
    """

    recurrent = True

    def __init__(self, sess, ob_space, ac_space, n_env, n_steps, n_batch, n_lstm=256, reuse=False, layers=None,
                 net_arch=None, act_fun=tf.tanh, cnn_extractor=nature_cnn, layer_norm=False, feature_extraction="cnn",
                 **kwargs):
        # state_shape = [n_lstm * 2] dim because of the cell and hidden states of the LSTM
        super(LstmPolicy, self).__init__(sess, ob_space, ac_space, n_env, n_steps, n_batch,
                                         state_shape=(2 * n_lstm, ), reuse=reuse,
                                         scale=(feature_extraction == "cnn"))

        self._kwargs_check(feature_extraction, kwargs)

        if net_arch is None:  # Legacy mode
            if layers is None:
                layers = [64, 64]
            else:
                warnings.warn("The layers parameter is deprecated. Use the net_arch parameter instead.")

            with tf.variable_scope("model", reuse=reuse):
                if feature_extraction == "cnn":
                    extracted_features = cnn_extractor(self.processed_obs, **kwargs)
                else:
                    extracted_features = tf.layers.flatten(self.processed_obs)
                    for i, layer_size in enumerate(layers):
                        extracted_features = act_fun(linear(extracted_features, 'pi_fc' + str(i), n_hidden=layer_size,
                                                            init_scale=np.sqrt(2)))
                input_sequence = batch_to_seq(extracted_features, self.n_env, n_steps)
                masks = batch_to_seq(self.dones_ph, self.n_env, n_steps)
                rnn_output, self.snew = lstm(input_sequence, masks, self.states_ph, 'lstm1', n_hidden=n_lstm,
                                             layer_norm=layer_norm)
                rnn_output = seq_to_batch(rnn_output)
                value_fn = linear(rnn_output, 'vf', 1)

                self._proba_distribution, self._policy, self.q_value = \
                    self.pdtype.proba_distribution_from_latent(rnn_output, rnn_output)

            self._value_fn = value_fn
        else:  # Use the new net_arch parameter
            if layers is not None:
                warnings.warn("The new net_arch parameter overrides the deprecated layers parameter.")
            if feature_extraction == "cnn":
                raise NotImplementedError()

            with tf.variable_scope("model", reuse=reuse):
                latent = tf.layers.flatten(self.processed_obs)
                policy_only_layers = []  # Layer sizes of the network that only belongs to the policy network
                value_only_layers = []  # Layer sizes of the network that only belongs to the value network

                # Iterate through the shared layers and build the shared parts of the network
                lstm_layer_constructed = False
                for idx, layer in enumerate(net_arch):
                    if isinstance(layer, int):  # Check that this is a shared layer
                        layer_size = layer
                        latent = act_fun(linear(latent, "shared_fc{}".format(idx), layer_size, init_scale=np.sqrt(2)))
                    elif layer == "lstm":
                        if lstm_layer_constructed:
                            raise ValueError("The net_arch parameter must only contain one occurrence of 'lstm'!")
                        input_sequence = batch_to_seq(latent, self.n_env, n_steps)
                        masks = batch_to_seq(self.dones_ph, self.n_env, n_steps)
                        rnn_output, self.snew = lstm(input_sequence, masks, self.states_ph, 'lstm1', n_hidden=n_lstm,
                                                     layer_norm=layer_norm)
                        latent = seq_to_batch(rnn_output)
                        lstm_layer_constructed = True
                    else:
                        assert isinstance(layer, dict), "Error: the net_arch list can only contain ints and dicts"
                        if 'pi' in layer:
                            assert isinstance(layer['pi'],
                                              list), "Error: net_arch[-1]['pi'] must contain a list of integers."
                            policy_only_layers = layer['pi']

                        if 'vf' in layer:
                            assert isinstance(layer['vf'],
                                              list), "Error: net_arch[-1]['vf'] must contain a list of integers."
                            value_only_layers = layer['vf']
                        break  # From here on the network splits up in policy and value network

                # Build the non-shared part of the policy-network
                latent_policy = latent
                for idx, pi_layer_size in enumerate(policy_only_layers):
                    if pi_layer_size == "lstm":
                        raise NotImplementedError("LSTMs are only supported in the shared part of the policy network.")
                    assert isinstance(pi_layer_size, int), "Error: net_arch[-1]['pi'] must only contain integers."
                    latent_policy = act_fun(
                        linear(latent_policy, "pi_fc{}".format(idx), pi_layer_size, init_scale=np.sqrt(2)))

                # Build the non-shared part of the value-network
                latent_value = latent
                for idx, vf_layer_size in enumerate(value_only_layers):
                    if vf_layer_size == "lstm":
                        raise NotImplementedError("LSTMs are only supported in the shared part of the value function "
                                                  "network.")
                    assert isinstance(vf_layer_size, int), "Error: net_arch[-1]['vf'] must only contain integers."
                    latent_value = act_fun(
                        linear(latent_value, "vf_fc{}".format(idx), vf_layer_size, init_scale=np.sqrt(2)))

                if not lstm_layer_constructed:
                    raise ValueError("The net_arch parameter must contain at least one occurrence of 'lstm'!")

                self._value_fn = linear(latent_value, 'vf', 1)
                # TODO: why not init_scale = 0.001 here like in the feedforward
                self._proba_distribution, self._policy, self.q_value = \
                    self.pdtype.proba_distribution_from_latent(latent_policy, latent_value)
        self._setup_init()

    def step(self, obs, state=None, mask=None, deterministic=False):
        if deterministic:
            return self.sess.run([self.deterministic_action, self.value_flat, self.snew, self.neglogp],
                                 {self.obs_ph: obs, self.states_ph: state, self.dones_ph: mask})
        else:
            return self.sess.run([self.action, self.value_flat, self.snew, self.neglogp],
                                 {self.obs_ph: obs, self.states_ph: state, self.dones_ph: mask})

    def proba_step(self, obs, state=None, mask=None):
        return self.sess.run(self.policy_proba, {self.obs_ph: obs, self.states_ph: state, self.dones_ph: mask})

    def value(self, obs, state=None, mask=None):
        return self.sess.run(self.value_flat, {self.obs_ph: obs, self.states_ph: state, self.dones_ph: mask})


class FeedForwardPolicy(ActorCriticPolicy):
    """
    Policy object that implements actor critic, using a feed forward neural network (PyTorch version).

    :param device: (torch.device or str) Device to run on
    :param ob_space: (gymnasium.Space) The observation space of the environment
    :param ac_space: (gymnasium.Space) The action space of the environment
    :param n_env: (int) The number of environments to run
    :param n_steps: (int) The number of steps to run for each environment
    :param n_batch: (int) The number of batch to run (n_envs * n_steps)
    :param reuse: (bool) If the policy is reusable or not (not used in PyTorch)
    :param layers: ([int]) (deprecated, use net_arch instead) The size of the Neural network for the policy
        (if None, default to [64, 64])
    :param net_arch: (list) Specification of the actor-critic policy network architecture (see mlp_extractor
        documentation for details).
    :param act_fun: (callable) the activation function to use in the neural network (e.g., torch.tanh, F.relu).
    :param cnn_extractor: (function (torch.Tensor, ``**kwargs``): (torch.Tensor)) the CNN feature extraction
    :param feature_extraction: (str) The feature extraction type ("cnn" or "mlp")
    :param kwargs: (dict) Extra keyword arguments for the nature CNN feature extraction
    """

    def __init__(self, device=None, ob_space=None, ac_space=None, n_env=1, n_steps=1, n_batch=None, reuse=False, layers=None, net_arch=None,
                 act_fun=torch.tanh, cnn_extractor=nature_cnn, feature_extraction="cnn", **kwargs):
        super(FeedForwardPolicy, self).__init__(device=device, ob_space=ob_space, ac_space=ac_space, n_env=n_env, n_steps=n_steps, n_batch=n_batch, reuse=reuse,
                                                scale=(feature_extraction == "cnn"))

        self._kwargs_check(feature_extraction, kwargs)

        if layers is not None:
            warnings.warn("Usage of the `layers` parameter is deprecated! Use net_arch instead "
                          "(it has a different semantics though).", DeprecationWarning)
            if net_arch is not None:
                warnings.warn("The new `net_arch` parameter overrides the deprecated `layers` parameter!",
                              DeprecationWarning)

        if net_arch is None:
            if layers is None:
                layers = [64, 64]
            net_arch = [dict(vf=layers, pi=layers)]

        # Build the network
        if feature_extraction == "cnn":
            pi_latent = vf_latent = cnn_extractor(self.processed_obs, **kwargs)
        else:
            # Flatten observations
            if isinstance(self.processed_obs, torch.Tensor):
                flat_obs = self.processed_obs.view(self.processed_obs.size(0), -1)
            else:
                flat_obs = torch.flatten(torch.as_tensor(self.processed_obs, dtype=torch.float32), start_dim=1)
            pi_latent, vf_latent = mlp_extractor(flat_obs, net_arch, act_fun)

        # Value function head
        if not hasattr(self, '_vf_layer'):
            self._vf_layer = nn.Linear(vf_latent.shape[-1], 1)
            nn.init.orthogonal_(self._vf_layer.weight, gain=1.0)
            nn.init.constant_(self._vf_layer.bias, 0)
        self._value_fn = self._vf_layer(vf_latent)

        self._proba_distribution, self._policy, self.q_value = \
            self.pdtype.proba_distribution_from_latent(pi_latent, vf_latent, init_scale=1)

        self._setup_init()

    def step(self, obs, state=None, mask=None, deterministic=False):
        # Convert obs to tensor if needed
        if not isinstance(obs, torch.Tensor):
            obs = torch.as_tensor(obs, dtype=torch.float32, device=self.device)
        
        # Forward pass
        if self.ob_space is not None:
            # Process observation if needed
            if hasattr(self, '_processed_obs'):
                # Update processed observation
                if isinstance(obs, torch.Tensor):
                    processed_obs = obs.to(self.device)
                else:
                    processed_obs = torch.as_tensor(obs, dtype=torch.float32, device=self.device)
            else:
                processed_obs = obs
        else:
            processed_obs = obs
        
        # Changed by Yunlong 
        if deterministic:
            action = self.tanh_deterministic_action
            value = self.value_flat
            neglogp = self.neglogp
            # Convert to numpy for compatibility
            if isinstance(action, torch.Tensor):
                action = action.detach().cpu().numpy()
            if isinstance(value, torch.Tensor):
                value = value.detach().cpu().numpy()
            if isinstance(neglogp, torch.Tensor):
                neglogp = neglogp.detach().cpu().numpy()
            return action, value, self.initial_state, neglogp                                              
        else:
            tanh_action = self.tanh_action
            action = self.action
            value = self.value_flat
            neglogp = self.tanh_neglogp
            # Convert to numpy for compatibility
            if isinstance(tanh_action, torch.Tensor):
                tanh_action = tanh_action.detach().cpu().numpy()
            if isinstance(action, torch.Tensor):
                action = action.detach().cpu().numpy()
            if isinstance(value, torch.Tensor):
                value = value.detach().cpu().numpy()
            if isinstance(neglogp, torch.Tensor):
                neglogp = neglogp.detach().cpu().numpy()
            return tanh_action, action, value, self.initial_state, neglogp

    def proba_step(self, obs, state=None, mask=None):
        if not isinstance(obs, torch.Tensor):
            obs = torch.as_tensor(obs, dtype=torch.float32, device=self.device)
        # Forward pass would go here - for now return policy_proba
        result = self.policy_proba
        if isinstance(result, torch.Tensor):
            return result.detach().cpu().numpy()
        elif isinstance(result, list):
            return [r.detach().cpu().numpy() if isinstance(r, torch.Tensor) else r for r in result]
        return result

    def value(self, obs, state=None, mask=None):
        if not isinstance(obs, torch.Tensor):
            obs = torch.as_tensor(obs, dtype=torch.float32, device=self.device)
        result = self.value_flat
        if isinstance(result, torch.Tensor):
            return result.detach().cpu().numpy()
        return result


class CnnPolicy(FeedForwardPolicy):
    """
    Policy object that implements actor critic, using a CNN (the nature CNN)

    :param sess: (TensorFlow session) The current TensorFlow session
    :param ob_space: (Gym Space) The observation space of the environment
    :param ac_space: (Gym Space) The action space of the environment
    :param n_env: (int) The number of environments to run
    :param n_steps: (int) The number of steps to run for each environment
    :param n_batch: (int) The number of batch to run (n_envs * n_steps)
    :param reuse: (bool) If the policy is reusable or not
    :param _kwargs: (dict) Extra keyword arguments for the nature CNN feature extraction
    """

    def __init__(self, sess, ob_space, ac_space, n_env, n_steps, n_batch, reuse=False, **_kwargs):
        super(CnnPolicy, self).__init__(sess, ob_space, ac_space, n_env, n_steps, n_batch, reuse,
                                        feature_extraction="cnn", **_kwargs)


class CnnLstmPolicy(LstmPolicy):
    """
    Policy object that implements actor critic, using LSTMs with a CNN feature extraction

    :param sess: (TensorFlow session) The current TensorFlow session
    :param ob_space: (Gym Space) The observation space of the environment
    :param ac_space: (Gym Space) The action space of the environment
    :param n_env: (int) The number of environments to run
    :param n_steps: (int) The number of steps to run for each environment
    :param n_batch: (int) The number of batch to run (n_envs * n_steps)
    :param n_lstm: (int) The number of LSTM cells (for recurrent policies)
    :param reuse: (bool) If the policy is reusable or not
    :param kwargs: (dict) Extra keyword arguments for the nature CNN feature extraction
    """

    def __init__(self, sess, ob_space, ac_space, n_env, n_steps, n_batch, n_lstm=256, reuse=False, **_kwargs):
        super(CnnLstmPolicy, self).__init__(sess, ob_space, ac_space, n_env, n_steps, n_batch, n_lstm, reuse,
                                            layer_norm=False, feature_extraction="cnn", **_kwargs)


class CnnLnLstmPolicy(LstmPolicy):
    """
    Policy object that implements actor critic, using a layer normalized LSTMs with a CNN feature extraction

    :param sess: (TensorFlow session) The current TensorFlow session
    :param ob_space: (Gym Space) The observation space of the environment
    :param ac_space: (Gym Space) The action space of the environment
    :param n_env: (int) The number of environments to run
    :param n_steps: (int) The number of steps to run for each environment
    :param n_batch: (int) The number of batch to run (n_envs * n_steps)
    :param n_lstm: (int) The number of LSTM cells (for recurrent policies)
    :param reuse: (bool) If the policy is reusable or not
    :param kwargs: (dict) Extra keyword arguments for the nature CNN feature extraction
    """

    def __init__(self, sess, ob_space, ac_space, n_env, n_steps, n_batch, n_lstm=256, reuse=False, **_kwargs):
        super(CnnLnLstmPolicy, self).__init__(sess, ob_space, ac_space, n_env, n_steps, n_batch, n_lstm, reuse,
                                              layer_norm=True, feature_extraction="cnn", **_kwargs)


class MlpPolicy(FeedForwardPolicy):
    """
    Policy object that implements actor critic, using a MLP (2 layers of 64)

    :param sess: (TensorFlow session) The current TensorFlow session
    :param ob_space: (Gym Space) The observation space of the environment
    :param ac_space: (Gym Space) The action space of the environment
    :param n_env: (int) The number of environments to run
    :param n_steps: (int) The number of steps to run for each environment
    :param n_batch: (int) The number of batch to run (n_envs * n_steps)
    :param reuse: (bool) If the policy is reusable or not
    :param _kwargs: (dict) Extra keyword arguments for the nature CNN feature extraction
    """

    def __init__(self, sess, ob_space, ac_space, n_env, n_steps, n_batch, reuse=False, **_kwargs):
        super(MlpPolicy, self).__init__(sess, ob_space, ac_space, n_env, n_steps, n_batch, reuse,
                                        feature_extraction="mlp", **_kwargs)


class MlpLstmPolicy(LstmPolicy):
    """
    Policy object that implements actor critic, using LSTMs with a MLP feature extraction

    :param sess: (TensorFlow session) The current TensorFlow session
    :param ob_space: (Gym Space) The observation space of the environment
    :param ac_space: (Gym Space) The action space of the environment
    :param n_env: (int) The number of environments to run
    :param n_steps: (int) The number of steps to run for each environment
    :param n_batch: (int) The number of batch to run (n_envs * n_steps)
    :param n_lstm: (int) The number of LSTM cells (for recurrent policies)
    :param reuse: (bool) If the policy is reusable or not
    :param kwargs: (dict) Extra keyword arguments for the nature CNN feature extraction
    """

    def __init__(self, sess, ob_space, ac_space, n_env, n_steps, n_batch, n_lstm=256, reuse=False, **_kwargs):
        super(MlpLstmPolicy, self).__init__(sess, ob_space, ac_space, n_env, n_steps, n_batch, n_lstm, reuse,
                                            layer_norm=False, feature_extraction="mlp", **_kwargs)


class MlpLnLstmPolicy(LstmPolicy):
    """
    Policy object that implements actor critic, using a layer normalized LSTMs with a MLP feature extraction

    :param sess: (TensorFlow session) The current TensorFlow session
    :param ob_space: (Gym Space) The observation space of the environment
    :param ac_space: (Gym Space) The action space of the environment
    :param n_env: (int) The number of environments to run
    :param n_steps: (int) The number of steps to run for each environment
    :param n_batch: (int) The number of batch to run (n_envs * n_steps)
    :param n_lstm: (int) The number of LSTM cells (for recurrent policies)
    :param reuse: (bool) If the policy is reusable or not
    :param kwargs: (dict) Extra keyword arguments for the nature CNN feature extraction
    """

    def __init__(self, sess, ob_space, ac_space, n_env, n_steps, n_batch, n_lstm=256, reuse=False, **_kwargs):
        super(MlpLnLstmPolicy, self).__init__(sess, ob_space, ac_space, n_env, n_steps, n_batch, n_lstm, reuse,
                                              layer_norm=True, feature_extraction="mlp", **_kwargs)


_policy_registry = {
    ActorCriticPolicy: {
        "CnnPolicy": CnnPolicy,
        "CnnLstmPolicy": CnnLstmPolicy,
        "CnnLnLstmPolicy": CnnLnLstmPolicy,
        "MlpPolicy": MlpPolicy,
        "MlpLstmPolicy": MlpLstmPolicy,
        "MlpLnLstmPolicy": MlpLnLstmPolicy,
    }
}


def get_policy_from_name(base_policy_type, name):
    """
    returns the registed policy from the base type and name

    :param base_policy_type: (BasePolicy) the base policy object
    :param name: (str) the policy name
    :return: (base_policy_type) the policy
    """
    if base_policy_type not in _policy_registry:
        raise ValueError("Error: the policy type {} is not registered!".format(base_policy_type))
    if name not in _policy_registry[base_policy_type]:
        raise ValueError("Error: unknown policy type {}, the only registed policy type are: {}!"
                         .format(name, list(_policy_registry[base_policy_type].keys())))
    return _policy_registry[base_policy_type][name]


def register_policy(name, policy):
    """
    returns the registed policy from the base type and name

    :param name: (str) the policy name
    :param policy: (subclass of BasePolicy) the policy
    """
    sub_class = None
    for cls in BasePolicy.__subclasses__():
        if issubclass(policy, cls):
            sub_class = cls
            break
    if sub_class is None:
        raise ValueError("Error: the policy {} is not of any known subclasses of BasePolicy!".format(policy))

    if sub_class not in _policy_registry:
        _policy_registry[sub_class] = {}
    if name in _policy_registry[sub_class]:
        raise ValueError("Error: the name {} is alreay registered for a different policy, will not override."
                         .format(name))
    _policy_registry[sub_class][name] = policy
