from typing import Optional

import torch

_MIN_SCALE = 1e-4


class ValueHead(torch.nn.Module):
    def __init__(self, fn=None):
        super().__init__()
        self.fn = fn

    def initialize(self, input_size, return_normalizer=None):
        self.return_normalizer = return_normalizer
        self.v_layer = torch.nn.Linear(input_size, 1)
        if self.fn:
            self.v_layer.apply(self.fn)

    def forward(self, inputs):
        out = self.v_layer(inputs)
        out = torch.squeeze(out, -1)
        if self.return_normalizer:
            out = self.return_normalizer(out)
        return out


class CategoricalWithSupport:
    def __init__(self, values, logits):
        self.values = values
        self.logits = logits
        self.probabilities = torch.nn.functional.softmax(logits, dim=-1)

    def mean(self):
        return (self.probabilities * self.values).sum(dim=-1)

    def project(self, returns):
        vmin, vmax = self.values[0], self.values[-1]
        d_pos = torch.cat([self.values, vmin[None]], 0)[1:]
        d_pos = (d_pos - self.values)[None, :, None]
        d_neg = torch.cat([vmax[None], self.values], 0)[:-1]
        d_neg = (self.values - d_neg)[None, :, None]

        clipped_returns = torch.clamp(returns, vmin, vmax)
        delta_values = clipped_returns[:, None] - self.values[None, :, None]
        delta_sign = (delta_values >= 0).float()
        delta_hat = (delta_sign * delta_values / d_pos) - (
            (1 - delta_sign) * delta_values / d_neg
        )
        delta_clipped = torch.clamp(1 - delta_hat, 0, 1)

        return (delta_clipped * self.probabilities[:, None]).sum(dim=2)


class DistributionalValueHead(torch.nn.Module):
    def __init__(self, vmin, vmax, num_atoms, fn=None):
        super().__init__()
        self.num_atoms = num_atoms
        self.fn = fn
        self.values = torch.linspace(vmin, vmax, num_atoms).float()

    def initialize(self, input_size, return_normalizer=None):
        if return_normalizer:
            raise ValueError(
                "Return normalizers cannot be used with distributional value"
                "heads."
            )
        self.distributional_layer = torch.nn.Linear(input_size, self.num_atoms)
        if self.fn:
            self.distributional_layer.apply(self.fn)

    def forward(self, inputs):
        logits = self.distributional_layer(inputs)
        return CategoricalWithSupport(values=self.values, logits=logits)


class DistributionalMoGValueHead(torch.nn.Module):
    def __init__(
        self,
        num_dimensions: int = 1,
        num_components: int = 5,
        init_scale: Optional[float] = None,
    ):
        super().__init__()
        self._num_components = num_components
        self._num_dimensions = num_dimensions
        if init_scale is not None:
            self._scale_factor = init_scale / torch.nn.functional.softplus(
                torch.zeros((1,))
            )
        else:
            self._scale_factor = (
                1.0  # Corresponds to init_scale = softplus(0).
            )

    def initialize(self, input_size, return_normalizer=None):
        if return_normalizer:
            raise ValueError(
                "Return normalizers cannot be used with distributional value"
                "heads."
            )
        # 1 mean and 1 std output per output dimension per component
        self._loc_layer = torch.nn.Linear(
            input_size, self._num_components * self._num_dimensions
        )
        self._scale_layer = torch.nn.Linear(
            input_size, self._num_components * self._num_dimensions
        )
        # if multi dimension but univariate -> we have one gaussian mixture per output dimension
        logits_size = self._num_components * self._num_dimensions
        # TODO check input dimension
        self._logit_layer = torch.nn.Linear(input_size, logits_size)

    def __call__(self, inputs: torch.Tensor, low_noise_policy: bool = False):
        """Run the networks through inputs.

        Args:
          inputs: hidden activations of the policy network body.
          low_noise_policy: whether to set vanishingly small scales for each
            component. If this flag is set to True, the policy is effectively run
            without Gaussian noise.

        Returns:
          Mixture Gaussian distribution.
        """

        # Compute logits, locs, and scales if necessary.
        logits = self._logit_layer(inputs)
        locs = self._loc_layer(inputs)

        # When a low_noise_policy is requested, set the scales to its minimum value.
        if low_noise_policy:
            scales = torch.full(locs.shape, _MIN_SCALE)
        else:
            scales = self._scale_layer(inputs)
            scales = (
                self._scale_factor * torch.nn.functional.softplus(scales)
                + _MIN_SCALE
            )

        shape = [-1, self._num_dimensions, self._num_components]
        # Reshape the mixture's location and scale parameters appropriately.
        locs = torch.reshape(locs, shape)
        scales = torch.reshape(scales, shape)
        components_distribution = torch.distributions.normal.Normal(
            loc=locs, scale=scales
        )
        logits = torch.reshape(logits, shape)

        # Create the mixture distribution.
        distribution = torch.distributions.MixtureSameFamily(
            mixture_distribution=torch.distributions.Categorical(
                logits=logits
            ),
            component_distribution=components_distribution,
        )
        distribution = torch.distributions.Independent(distribution, 1)
        return distribution


class Critic(torch.nn.Module):
    def __init__(self, encoder, torso, head):
        super().__init__()
        self.encoder = encoder
        self.torso = torso
        self.head = head

    def initialize(
        self,
        observation_space,
        action_space,
        observation_normalizer=None,
        return_normalizer=None,
    ):
        size = self.encoder.initialize(
            observation_space=observation_space,
            action_space=action_space,
            observation_normalizer=observation_normalizer,
        )
        size = self.torso.initialize(size)
        self.head.initialize(size, return_normalizer)

    def forward(self, *inputs):
        out = self.encoder(*inputs)
        out = self.torso(out)
        return self.head(out)
