import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from gymnasium import spaces

EPS = 1e-6  # Avoid NaN (prevents division by zero or log of zero)
# CAP the standard deviation of the actor
LOG_STD_MAX = 2
LOG_STD_MIN = -20

class ProbabilityDistribution(object):
    """
    Base class for describing a probability distribution.
    """
    def __init__(self):
        super(ProbabilityDistribution, self).__init__()

    def flatparam(self):
        """
        Return the direct probabilities

        :return: ([float]) the probabilities
        """
        raise NotImplementedError

    def mode(self):
        """
        Returns the probability

        :return: (Tensorflow Tensor) the deterministic action
        """
        raise NotImplementedError

    def neglogp(self, x):
        """
        returns the of the negative log likelihood

        :param x: (str) the labels of each index
        :return: ([float]) The negative log likelihood of the distribution
        """
        # Usually it's easier to define the negative logprob
        raise NotImplementedError

    def kl(self, other):
        """
        Calculates the Kullback-Leibler divergence from the given probability distribution

        :param other: ([float]) the distribution to compare with
        :return: (float) the KL divergence of the two distributions
        """
        raise NotImplementedError

    def entropy(self):
        """
        Returns Shannon's entropy of the probability

        :return: (float) the entropy
        """
        raise NotImplementedError

    def sample(self):
        """
        returns a sample from the probability distribution

        :return: (Tensorflow Tensor) the stochastic action
        """
        raise NotImplementedError

    def logp(self, x):
        """
        returns the of the log likelihood

        :param x: (str) the labels of each index
        :return: ([float]) The log likelihood of the distribution
        """
        return - self.neglogp(x)


class ProbabilityDistributionType(object):
    """
    Parametrized family of probability distributions
    """

    def probability_distribution_class(self):
        """
        returns the ProbabilityDistribution class of this type

        :return: (Type ProbabilityDistribution) the probability distribution class associated
        """
        raise NotImplementedError

    def proba_distribution_from_flat(self, flat):
        """
        Returns the probability distribution from flat probabilities
        flat: flattened vector of parameters of probability distribution

        :param flat: ([float]) the flat probabilities
        :return: (ProbabilityDistribution) the instance of the ProbabilityDistribution associated
        """
        return self.probability_distribution_class()(flat)

    def proba_distribution_from_latent(self, pi_latent_vector, vf_latent_vector, init_scale=1.0, init_bias=0.0):
        """
        returns the probability distribution from latent values

        :param pi_latent_vector: ([float]) the latent pi values
        :param vf_latent_vector: ([float]) the latent vf values
        :param init_scale: (float) the initial scale of the distribution
        :param init_bias: (float) the initial bias of the distribution
        :return: (ProbabilityDistribution) the instance of the ProbabilityDistribution associated
        """
        raise NotImplementedError

    def param_shape(self):
        """
        returns the shape of the input parameters

        :return: ([int]) the shape
        """
        raise NotImplementedError

    def sample_shape(self):
        """
        returns the shape of the sampling

        :return: ([int]) the shape
        """
        raise NotImplementedError

    def sample_dtype(self):
        """
        returns the type of the sampling

        :return: (type) the type
        """
        raise NotImplementedError

    def param_placeholder(self, prepend_shape, name=None):
        """
        returns a PyTorch tensor placeholder for the input parameters (for compatibility)

        :param prepend_shape: ([int]) the prepend shape
        :param name: (str) the placeholder name (not used in PyTorch)
        :return: (torch.Tensor) a zero tensor with the specified shape
        """
        shape = prepend_shape + self.param_shape()
        return torch.zeros(shape, dtype=torch.float32)

    def sample_placeholder(self, prepend_shape, name=None):
        """
        returns a PyTorch tensor placeholder for the sampling (for compatibility)

        :param prepend_shape: ([int]) the prepend shape
        :param name: (str) the placeholder name (not used in PyTorch)
        :return: (torch.Tensor) a zero tensor with the specified shape
        """
        shape = prepend_shape + self.sample_shape()
        dtype = self.sample_dtype()
        return torch.zeros(shape, dtype=dtype)


class CategoricalProbabilityDistributionType(ProbabilityDistributionType):
    def __init__(self, n_cat):
        """
        The probability distribution type for categorical input

        :param n_cat: (int) the number of categories
        """
        self.n_cat = n_cat

    def probability_distribution_class(self):
        return CategoricalProbabilityDistribution

    def proba_distribution_from_latent(self, pi_latent_vector, vf_latent_vector, init_scale=1.0, init_bias=0.0):
        # Create linear layers for policy and value
        if not hasattr(self, '_pi_layer'):
            self._pi_layer = nn.Linear(pi_latent_vector.shape[-1], self.n_cat)
            nn.init.orthogonal_(self._pi_layer.weight, gain=init_scale)
            nn.init.constant_(self._pi_layer.bias, init_bias)
        if not hasattr(self, '_q_layer'):
            self._q_layer = nn.Linear(vf_latent_vector.shape[-1], self.n_cat)
            nn.init.orthogonal_(self._q_layer.weight, gain=init_scale)
            nn.init.constant_(self._q_layer.bias, init_bias)
        
        pdparam = self._pi_layer(pi_latent_vector)
        q_values = self._q_layer(vf_latent_vector)
        return self.proba_distribution_from_flat(pdparam), pdparam, q_values

    def param_shape(self):
        return [self.n_cat]

    def sample_shape(self):
        return []

    def sample_dtype(self):
        return torch.int64


class MultiCategoricalProbabilityDistributionType(ProbabilityDistributionType):
    def __init__(self, n_vec):
        """
        The probability distribution type for multiple categorical input

        :param n_vec: ([int]) the vectors
        """
        # Cast the variable because tf does not allow uint32
        self.n_vec = n_vec.astype(np.int32)
        # Check that the cast was valid
        assert (self.n_vec > 0).all(), "Casting uint32 to int32 was invalid"

    def probability_distribution_class(self):
        return MultiCategoricalProbabilityDistribution

    def proba_distribution_from_flat(self, flat):
        return MultiCategoricalProbabilityDistribution(self.n_vec, flat)

    def proba_distribution_from_latent(self, pi_latent_vector, vf_latent_vector, init_scale=1.0, init_bias=0.0):
        # Create linear layers for policy and value
        if not hasattr(self, '_pi_layer'):
            self._pi_layer = nn.Linear(pi_latent_vector.shape[-1], sum(self.n_vec))
            nn.init.orthogonal_(self._pi_layer.weight, gain=init_scale)
            nn.init.constant_(self._pi_layer.bias, init_bias)
        if not hasattr(self, '_q_layer'):
            self._q_layer = nn.Linear(vf_latent_vector.shape[-1], sum(self.n_vec))
            nn.init.orthogonal_(self._q_layer.weight, gain=init_scale)
            nn.init.constant_(self._q_layer.bias, init_bias)
        
        pdparam = self._pi_layer(pi_latent_vector)
        q_values = self._q_layer(vf_latent_vector)
        return self.proba_distribution_from_flat(pdparam), pdparam, q_values

    def param_shape(self):
        return [sum(self.n_vec)]

    def sample_shape(self):
        return [len(self.n_vec)]

    def sample_dtype(self):
        return torch.int64


class DiagGaussianProbabilityDistributionType(ProbabilityDistributionType):
    def __init__(self, size):
        """
        The probability distribution type for multivariate Gaussian input

        :param size: (int) the number of dimensions of the multivariate gaussian
        """
        self.size = size

    def probability_distribution_class(self):
        return DiagGaussianProbabilityDistribution

    def proba_distribution_from_flat(self, flat):
        """
        returns the probability distribution from flat probabilities

        :param flat: ([float]) the flat probabilities
        :return: (ProbabilityDistribution) the instance of the ProbabilityDistribution associated
        """
        return self.probability_distribution_class()(flat)

    def proba_distribution_from_latent(self, pi_latent_vector, vf_latent_vector, init_scale=1.0, init_bias=0.0):
        # Create linear layers for policy and value
        if not hasattr(self, '_pi_layer'):
            self._pi_layer = nn.Linear(pi_latent_vector.shape[-1], self.size)
            nn.init.orthogonal_(self._pi_layer.weight, gain=init_scale)
            nn.init.constant_(self._pi_layer.bias, init_bias)
        if not hasattr(self, '_q_layer'):
            self._q_layer = nn.Linear(vf_latent_vector.shape[-1], self.size)
            nn.init.orthogonal_(self._q_layer.weight, gain=init_scale)
            nn.init.constant_(self._q_layer.bias, init_bias)
        if not hasattr(self, '_logstd'):
            self._logstd = nn.Parameter(torch.zeros(1, self.size))
        
        mean = self._pi_layer(pi_latent_vector)
        logstd = self._logstd.expand_as(mean)
        pdparam = torch.cat([mean, logstd], dim=-1)
        q_values = self._q_layer(vf_latent_vector)
        return self.proba_distribution_from_flat(pdparam), mean, q_values

    def param_shape(self):
        return [2 * self.size]

    def sample_shape(self):
        return [self.size]

    def sample_dtype(self):
        return torch.float32


class BernoulliProbabilityDistributionType(ProbabilityDistributionType):
    def __init__(self, size):
        """
        The probability distribution type for Bernoulli input

        :param size: (int) the number of dimensions of the Bernoulli distribution
        """
        self.size = size

    def probability_distribution_class(self):
        return BernoulliProbabilityDistribution

    def proba_distribution_from_latent(self, pi_latent_vector, vf_latent_vector, init_scale=1.0, init_bias=0.0):
        # Create linear layers for policy and value
        if not hasattr(self, '_pi_layer'):
            self._pi_layer = nn.Linear(pi_latent_vector.shape[-1], self.size)
            nn.init.orthogonal_(self._pi_layer.weight, gain=init_scale)
            nn.init.constant_(self._pi_layer.bias, init_bias)
        if not hasattr(self, '_q_layer'):
            self._q_layer = nn.Linear(vf_latent_vector.shape[-1], self.size)
            nn.init.orthogonal_(self._q_layer.weight, gain=init_scale)
            nn.init.constant_(self._q_layer.bias, init_bias)
        
        pdparam = self._pi_layer(pi_latent_vector)
        q_values = self._q_layer(vf_latent_vector)
        return self.proba_distribution_from_flat(pdparam), pdparam, q_values

    def param_shape(self):
        return [self.size]

    def sample_shape(self):
        return [self.size]

    def sample_dtype(self):
        return torch.int32


class CategoricalProbabilityDistribution(ProbabilityDistribution):
    def __init__(self, logits):
        """
        Probability distributions from categorical input

        :param logits: ([float]) the categorical logits input
        """
        self.logits = logits
        super(CategoricalProbabilityDistribution, self).__init__()

    def flatparam(self):
        return self.logits

    def mode(self):
        return torch.argmax(self.logits, dim=-1)

    def neglogp(self, x):
        # Convert to torch tensor if needed
        if not isinstance(x, torch.Tensor):
            x = torch.as_tensor(x, dtype=torch.long)
        if not isinstance(self.logits, torch.Tensor):
            self.logits = torch.as_tensor(self.logits)
        
        # Use cross_entropy which is numerically stable
        return F.cross_entropy(self.logits, x, reduction='none')

    def kl(self, other):
        if not isinstance(self.logits, torch.Tensor):
            self.logits = torch.as_tensor(self.logits)
        if not isinstance(other.logits, torch.Tensor):
            other.logits = torch.as_tensor(other.logits)
        
        a_0 = self.logits - torch.max(self.logits, dim=-1, keepdim=True)[0]
        a_1 = other.logits - torch.max(other.logits, dim=-1, keepdim=True)[0]
        exp_a_0 = torch.exp(a_0)
        exp_a_1 = torch.exp(a_1)
        z_0 = torch.sum(exp_a_0, dim=-1, keepdim=True)
        z_1 = torch.sum(exp_a_1, dim=-1, keepdim=True)
        p_0 = exp_a_0 / z_0
        return torch.sum(p_0 * (a_0 - torch.log(z_0 + EPS) - a_1 + torch.log(z_1 + EPS)), dim=-1)

    def entropy(self):
        if not isinstance(self.logits, torch.Tensor):
            self.logits = torch.as_tensor(self.logits)
        
        a_0 = self.logits - torch.max(self.logits, dim=-1, keepdim=True)[0]
        exp_a_0 = torch.exp(a_0)
        z_0 = torch.sum(exp_a_0, dim=-1, keepdim=True)
        p_0 = exp_a_0 / z_0
        return torch.sum(p_0 * (torch.log(z_0 + EPS) - a_0), dim=-1)

    def sample(self):
        # Gumbel-max trick to sample
        # a categorical distribution (see http://amid.fish/humble-gumbel)
        if not isinstance(self.logits, torch.Tensor):
            self.logits = torch.as_tensor(self.logits)
        
        uniform = torch.rand_like(self.logits)
        return torch.argmax(self.logits - torch.log(-torch.log(uniform + EPS) + EPS), dim=-1)

    @classmethod
    def fromflat(cls, flat):
        """
        Create an instance of this from new logits values

        :param flat: ([float]) the categorical logits input
        :return: (ProbabilityDistribution) the instance from the given categorical input
        """
        return cls(flat)


class MultiCategoricalProbabilityDistribution(ProbabilityDistribution):
    def __init__(self, nvec, flat):
        """
        Probability distributions from multicategorical input

        :param nvec: ([int]) the sizes of the different categorical inputs
        :param flat: ([float]) the categorical logits input
        """
        self.flat = flat
        if not isinstance(flat, torch.Tensor):
            flat = torch.as_tensor(flat)
        if not isinstance(nvec, torch.Tensor):
            nvec = torch.as_tensor(nvec, dtype=torch.long)
        
        # Split flat into separate logits for each categorical
        splits = torch.split(flat, nvec.tolist(), dim=-1)
        self.categoricals = [CategoricalProbabilityDistribution(split) for split in splits]
        super(MultiCategoricalProbabilityDistribution, self).__init__()

    def flatparam(self):
        return self.flat

    def mode(self):
        modes = [p.mode() for p in self.categoricals]
        return torch.stack(modes, dim=-1)

    def neglogp(self, x):
        if not isinstance(x, torch.Tensor):
            x = torch.as_tensor(x)
        # Unstack along last dimension
        x_unstacked = torch.unbind(x, dim=-1)
        neglogps = [p.neglogp(px) for p, px in zip(self.categoricals, x_unstacked)]
        return sum(neglogps)

    def kl(self, other):
        kls = [p.kl(q) for p, q in zip(self.categoricals, other.categoricals)]
        return sum(kls)

    def entropy(self):
        entropies = [p.entropy() for p in self.categoricals]
        return sum(entropies)

    def sample(self):
        samples = [p.sample() for p in self.categoricals]
        return torch.stack(samples, dim=-1)

    @classmethod
    def fromflat(cls, flat):
        """
        Create an instance of this from new logits values

        :param flat: ([float]) the multi categorical logits input
        :return: (ProbabilityDistribution) the instance from the given multi categorical input
        """
        raise NotImplementedError


class DiagGaussianProbabilityDistribution(ProbabilityDistribution):
    def __init__(self, flat):
        """
        Probability distributions from multivariate Gaussian input

        :param flat: ([float]) the multivariate Gaussian input data
        """
        self.flat = flat
        if not isinstance(flat, torch.Tensor):
            flat = torch.as_tensor(flat)
        
        # Split flat into mean and logstd
        split_size = flat.shape[-1] // 2
        mean, logstd = torch.split(flat, split_size, dim=-1)
        self.mean = mean
        # self.logstd = torch.clamp(logstd, LOG_STD_MIN, LOG_STD_MAX)
        self.logstd = logstd
        self.std = torch.exp(logstd)
        super(DiagGaussianProbabilityDistribution, self).__init__()

    def flatparam(self):
        return self.flat

    def mode(self):
        # Bounds are taken into account outside this class (during training only)
        return self.mean

    def neglogp(self, x):
        if not isinstance(x, torch.Tensor):
            x = torch.as_tensor(x)
        
        return 0.5 * torch.sum(torch.square((x - self.mean) / (self.std + EPS)), dim=-1) \
                + 0.5 * np.log(2.0 * np.pi) * float(x.shape[-1]) \
                + torch.sum(self.logstd, dim=-1)

    def tanh_neglogp(self, u):
        if not isinstance(u, torch.Tensor):
            u = torch.as_tensor(u)
        
        logp = - self.neglogp(u)
        tanh_logp = logp - torch.sum(torch.log(1 - torch.tanh(u)**2 + EPS), dim=1)
        return -tanh_logp
        
    def kl(self, other):
        assert isinstance(other, DiagGaussianProbabilityDistribution)
        if not isinstance(self.mean, torch.Tensor):
            self.mean = torch.as_tensor(self.mean)
            self.logstd = torch.as_tensor(self.logstd)
            self.std = torch.as_tensor(self.std)
        if not isinstance(other.mean, torch.Tensor):
            other.mean = torch.as_tensor(other.mean)
            other.logstd = torch.as_tensor(other.logstd)
            other.std = torch.as_tensor(other.std)
        
        return torch.sum(other.logstd - self.logstd + (torch.square(self.std) + torch.square(self.mean - other.mean)) /
                         (2.0 * torch.square(other.std) + EPS) - 0.5, dim=-1)

    def entropy(self):
        if not isinstance(self.logstd, torch.Tensor):
            self.logstd = torch.as_tensor(self.logstd)
        
        return torch.sum(self.logstd + .5 * np.log(2.0 * np.pi * np.e), dim=-1)

    def sample(self):
        # Bounds are taken into account outside this class (during training only)
        # Otherwise, it changes the distribution and breaks PPO2 for instance
        if not isinstance(self.mean, torch.Tensor):
            self.mean = torch.as_tensor(self.mean)
            self.std = torch.as_tensor(self.std)
        
        return self.mean + self.std * torch.randn_like(self.mean)
        
    @classmethod
    def fromflat(cls, flat):
        """
        Create an instance of this from new multivariate Gaussian input

        :param flat: ([float]) the multivariate Gaussian input data
        :return: (ProbabilityDistribution) the instance from the given multivariate Gaussian input data
        """
        return cls(flat)


class BernoulliProbabilityDistribution(ProbabilityDistribution):
    def __init__(self, logits):
        """
        Probability distributions from Bernoulli input

        :param logits: ([float]) the Bernoulli input data
        """
        self.logits = logits
        if not isinstance(logits, torch.Tensor):
            logits = torch.as_tensor(logits)
            self.logits = logits
        self.probabilities = torch.sigmoid(logits)
        super(BernoulliProbabilityDistribution, self).__init__()

    def flatparam(self):
        return self.logits

    def mode(self):
        return torch.round(self.probabilities)

    def neglogp(self, x):
        if not isinstance(x, torch.Tensor):
            x = torch.as_tensor(x, dtype=torch.float32)
        if not isinstance(self.logits, torch.Tensor):
            self.logits = torch.as_tensor(self.logits)
        
        # Binary cross entropy with logits
        return torch.sum(F.binary_cross_entropy_with_logits(self.logits, x, reduction='none'), dim=-1)

    def kl(self, other):
        if not isinstance(self.logits, torch.Tensor):
            self.logits = torch.as_tensor(self.logits)
            self.probabilities = torch.as_tensor(self.probabilities)
        if not isinstance(other.logits, torch.Tensor):
            other.logits = torch.as_tensor(other.logits)
        
        return torch.sum(F.binary_cross_entropy_with_logits(other.logits, self.probabilities, reduction='none'), dim=-1) - \
               torch.sum(F.binary_cross_entropy_with_logits(self.logits, self.probabilities, reduction='none'), dim=-1)

    def entropy(self):
        if not isinstance(self.logits, torch.Tensor):
            self.logits = torch.as_tensor(self.logits)
            self.probabilities = torch.as_tensor(self.probabilities)
        
        return torch.sum(F.binary_cross_entropy_with_logits(self.logits, self.probabilities, reduction='none'), dim=-1)

    def sample(self):
        if not isinstance(self.probabilities, torch.Tensor):
            self.probabilities = torch.as_tensor(self.probabilities)
        
        samples_from_uniform = torch.rand_like(self.probabilities)
        return (samples_from_uniform < self.probabilities).float()

    @classmethod
    def fromflat(cls, flat):
        """
        Create an instance of this from new Bernoulli input

        :param flat: ([float]) the Bernoulli input data
        :return: (ProbabilityDistribution) the instance from the given Bernoulli input data
        """
        return cls(flat)


def make_proba_dist_type(ac_space):
    """
    return an instance of ProbabilityDistributionType for the correct type of action space

    :param ac_space: (Gym Space) the input action space
    :return: (ProbabilityDistributionType) the appropriate instance of a ProbabilityDistributionType
    """
    if isinstance(ac_space, spaces.Box):
        assert len(ac_space.shape) == 1, "Error: the action space must be a vector"
        return DiagGaussianProbabilityDistributionType(ac_space.shape[0])
    elif isinstance(ac_space, spaces.Discrete):
        return CategoricalProbabilityDistributionType(ac_space.n)
    elif isinstance(ac_space, spaces.MultiDiscrete):
        return MultiCategoricalProbabilityDistributionType(ac_space.nvec)
    elif isinstance(ac_space, spaces.MultiBinary):
        return BernoulliProbabilityDistributionType(ac_space.n)
    else:
        raise NotImplementedError("Error: probability distribution, not implemented for action space of type {}."
                                  .format(type(ac_space)) +
                                  " Must be of type Gym Spaces: Box, Discrete, MultiDiscrete or MultiBinary.")


def shape_el(tensor, index):
    """
    get the shape of a PyTorch Tensor element

    :param tensor: (torch.Tensor) the input tensor
    :param index: (int) the element
    :return: (int) the shape
    """
    if not isinstance(tensor, torch.Tensor):
        tensor = torch.as_tensor(tensor)
    return tensor.shape[index]
