import numpy as np
import torch

from deprl import custom_torso
from deprl.utils import JacobianReg
from deprl.vendor.tonic import logger, replays
from deprl.vendor.tonic.torch import agents, models, updaters
from deprl.vendor.tonic.torch.models.critics import CategoricalWithSupport

FLOAT_EPSILON = 1e-8


class TunedExpectedSARSA(updaters.critics.ExpectedSARSA):
    def __init__(
        self, num_samples=20, loss=None, optimizer=None, gradient_clip=0
    ):
        self.num_samples = num_samples
        self.loss = loss or torch.nn.MSELoss()
        self.optimizer = optimizer or (
            lambda params: torch.optim.Adam(params, lr=3e-4, weight_decay=1e-5)
        )
        self.gradient_clip = gradient_clip


class TunedMPOActor(updaters.actors.MaximumAPosterioriPolicyOptimization):
    def __init__(
        self,
        num_samples=20,
        epsilon=1e-1,
        epsilon_penalty=1e-3,
        epsilon_mean=1e-3,
        epsilon_std=1e-6,
        initial_log_temperature=1.0,
        initial_log_alpha_mean=1.0,
        initial_log_alpha_std=10.0,
        min_log_dual=-18.0,
        per_dim_constraining=True,
        action_penalization=True,
        actor_optimizer=None,
        dual_optimizer=None,
        gradient_clip=0,
    ):
        self.num_samples = num_samples
        self.epsilon = epsilon
        self.epsilon_mean = epsilon_mean
        self.epsilon_std = epsilon_std
        self.initial_log_temperature = initial_log_temperature
        self.initial_log_alpha_mean = initial_log_alpha_mean
        self.initial_log_alpha_std = initial_log_alpha_std
        self.min_log_dual = torch.as_tensor(min_log_dual, dtype=torch.float32)
        self.action_penalization = action_penalization
        self.epsilon_penalty = epsilon_penalty
        self.per_dim_constraining = per_dim_constraining
        self.actor_optimizer = actor_optimizer or (
            lambda params: torch.optim.Adam(params, lr=3e-4, weight_decay=1e-5)
        )
        self.dual_optimizer = dual_optimizer or (
            lambda params: torch.optim.Adam(params, lr=1e-2, weight_decay=1e-5)
        )
        self.gradient_clip = gradient_clip


class JacMPOActor(TunedMPOActor):
    def __init__(self, jacob_lr=0.01, *args, **kwargs):
        self.jac_lr = jacob_lr
        self.jac_reg = JacobianReg()
        super().__init__(*args, **kwargs)

    def __call__(self, observations):
        def parametric_kl_and_dual_losses(kl, alpha, epsilon):
            kl_mean = kl.mean(dim=0)
            kl_loss = (alpha.detach() * kl_mean).sum()
            alpha_loss = (alpha * (epsilon - kl_mean.detach())).sum()
            return kl_loss, alpha_loss

        def weights_and_temperature_loss(q_values, epsilon, temperature):
            tempered_q_values = q_values.detach() / temperature
            weights = torch.nn.functional.softmax(tempered_q_values, dim=0)
            weights = weights.detach()

            # Temperature loss (dual of the E-step).
            q_log_sum_exp = torch.logsumexp(tempered_q_values, dim=0)
            num_actions = torch.as_tensor(
                q_values.shape[0], dtype=torch.float32
            )
            log_num_actions = torch.log(num_actions)
            loss = epsilon + (q_log_sum_exp).mean() - log_num_actions
            loss = temperature * loss

            return weights, loss

        # Use independent normals to satisfy KL constraints per-dimension.
        def independent_normals(distribution_1, distribution_2=None):
            distribution_2 = distribution_2 or distribution_1
            return torch.distributions.independent.Independent(
                torch.distributions.normal.Normal(
                    distribution_1.mean, distribution_2.stddev
                ),
                -1,
            )

        with torch.no_grad():
            self.log_temperature.data.copy_(
                torch.maximum(self.min_log_dual, self.log_temperature)
            )
            self.log_alpha_mean.data.copy_(
                torch.maximum(self.min_log_dual, self.log_alpha_mean)
            )
            self.log_alpha_std.data.copy_(
                torch.maximum(self.min_log_dual, self.log_alpha_std)
            )
            if self.action_penalization:
                self.log_penalty_temperature.data.copy_(
                    torch.maximum(
                        self.min_log_dual, self.log_penalty_temperature
                    )
                )

            target_distributions = self.model.target_actor(observations)
            actions = target_distributions.sample((self.num_samples,))

            tiled_observations = updaters.tile(observations, self.num_samples)
            flat_observations = updaters.merge_first_two_dims(
                tiled_observations
            )
            flat_actions = updaters.merge_first_two_dims(actions)
            values = self.model.target_critic(flat_observations, flat_actions)
            values = values.view(self.num_samples, -1)

            assert isinstance(
                target_distributions, torch.distributions.normal.Normal
            )
            target_distributions = independent_normals(target_distributions)

        self.actor_optimizer.zero_grad()
        self.dual_optimizer.zero_grad()

        distributions = self.model.actor(observations)
        distributions = independent_normals(distributions)

        temperature = (
            torch.nn.functional.softplus(self.log_temperature) + FLOAT_EPSILON
        )
        alpha_mean = (
            torch.nn.functional.softplus(self.log_alpha_mean) + FLOAT_EPSILON
        )
        alpha_std = (
            torch.nn.functional.softplus(self.log_alpha_std) + FLOAT_EPSILON
        )
        weights, temperature_loss = weights_and_temperature_loss(
            values, self.epsilon, temperature
        )

        # Action penalization is quadratic beyond [-1, 1].
        if self.action_penalization:
            penalty_temperature = (
                torch.nn.functional.softplus(self.log_penalty_temperature)
                + FLOAT_EPSILON
            )
            diff_bounds = actions - torch.clamp(actions, -1, 1)
            action_bound_costs = -torch.norm(diff_bounds, dim=-1)
            (
                penalty_weights,
                penalty_temperature_loss,
            ) = weights_and_temperature_loss(
                action_bound_costs, self.epsilon_penalty, penalty_temperature
            )
            weights += penalty_weights
            temperature_loss += penalty_temperature_loss

        # Decompose the policy into fixed-mean and fixed-std distributions.
        fixed_std_distribution = independent_normals(
            distributions.base_dist, target_distributions.base_dist
        )
        fixed_mean_distribution = independent_normals(
            target_distributions.base_dist, distributions.base_dist
        )

        # Compute the decomposed policy losses.
        policy_mean_losses = (
            fixed_std_distribution.base_dist.log_prob(actions).sum(dim=-1)
            * weights
        ).sum(dim=0)
        policy_mean_loss = -(policy_mean_losses).mean()
        policy_std_losses = (
            fixed_mean_distribution.base_dist.log_prob(actions).sum(dim=-1)
            * weights
        ).sum(dim=0)
        policy_std_loss = -policy_std_losses.mean()

        # Compute the decomposed KL between the target and online policies.
        if self.per_dim_constraining:
            kl_mean = torch.distributions.kl.kl_divergence(
                target_distributions.base_dist,
                fixed_std_distribution.base_dist,
            )
            kl_std = torch.distributions.kl.kl_divergence(
                target_distributions.base_dist,
                fixed_mean_distribution.base_dist,
            )
        else:
            kl_mean = torch.distributions.kl.kl_divergence(
                target_distributions, fixed_std_distribution
            )
            kl_std = torch.distributions.kl.kl_divergence(
                target_distributions, fixed_mean_distribution
            )

        # Compute the alpha-weighted KL-penalty and dual losses.
        kl_mean_loss, alpha_mean_loss = parametric_kl_and_dual_losses(
            kl_mean, alpha_mean, self.epsilon_mean
        )
        kl_std_loss, alpha_std_loss = parametric_kl_and_dual_losses(
            kl_std, alpha_std, self.epsilon_std
        )

        # Combine losses.
        policy_loss = policy_mean_loss + policy_std_loss
        kl_loss = kl_mean_loss + kl_std_loss
        dual_loss = alpha_mean_loss + alpha_std_loss + temperature_loss
        observations.requires_grad = True
        actions_actor = self.model.actor(observations)
        output = torch.concat(
            [actions_actor.loc, actions_actor.scale], axis=-1
        )
        jac_loss = self.jac_reg(observations, output)
        loss = policy_loss + kl_loss + dual_loss + self.jac_lr * jac_loss

        loss.backward()
        if self.gradient_clip > 0:
            torch.nn.utils.clip_grad_norm_(
                self.actor_variables, self.gradient_clip
            )
            torch.nn.utils.clip_grad_norm_(
                self.dual_variables, self.gradient_clip
            )
        self.actor_optimizer.step()
        self.dual_optimizer.step()

        dual_variables = dict(
            temperature=temperature.detach(),
            alpha_mean=alpha_mean.detach(),
            alpha_std=alpha_std.detach(),
        )
        if self.action_penalization:
            dual_variables[
                "penalty_temperature"
            ] = penalty_temperature.detach()

        return dict(
            policy_mean_loss=policy_mean_loss.detach(),
            policy_std_loss=policy_std_loss.detach(),
            kl_mean_loss=kl_mean_loss.detach(),
            kl_std_loss=kl_std_loss.detach(),
            alpha_mean_loss=alpha_mean_loss.detach(),
            alpha_std_loss=alpha_std_loss.detach(),
            temperature_loss=temperature_loss.detach(),
            jac_loss=jac_loss.detach(),
            **dual_variables,
        )


class DistributionalMPO(agents.MPO):
    def __init__(
        self,
        model=None,
        replay=None,
        actor_updater=None,
        critic_updater=None,
        hidden_size=[256],
    ):
        self.model = model or custom_torso.custom_mog_distributional_mpo(
            actor_size=hidden_size, critic_size=hidden_size
        )
        self.replay = replay or replays.Buffer(return_steps=5)
        self.actor_updater = actor_updater or DistributionalMoGActorMPO()
        self.critic_updater = critic_updater or DistributionalMoGCritic()

    def set_params(
        self,
        lr_critic=1e-3,
        grad_clip_critic=0,
        lr_actor=3e-4,
        lr_dual=1e-2,
        jacob_lr=0.0,
        grad_clip_actor=0,
        hidden_size=None,
        actor_size=[256, 256],
        critic_size=[256, 256],
        batch_size=None,
        retnorm=None,
        return_steps=None,
        critic_type="c51",
    ):
        self.jacob_lr = jacob_lr

        def optim_critic(params):
            return torch.optim.Adam(params, lr_critic)

        def optim_actor(params):
            return torch.optim.Adam(params, lr=lr_actor)

        def optim_dual(params):
            return torch.optim.Adam(params, lr=lr_dual)

        if critic_type == "c51":
            self.critic_updater = DistributionalC51Critic(
                optimizer=optim_critic, gradient_clip=grad_clip_critic
            )
        elif critic_type == "mog":
            self.critic_updater = DistributionalMoGCritic(
                optimizer=optim_critic, gradient_clip=grad_clip_critic
            )
        elif critic_type == "default":
            self.critic_updater = TunedExpectedSARSA(
                optimizer=optim_critic, gradient_clip=grad_clip_critic
            )
        else:
            raise NotImplementedError

        if critic_type == "c51":
            self.actor_updater = DistributionalC51ActorMPO(
                actor_optimizer=optim_actor,
                dual_optimizer=optim_dual,
                gradient_clip=grad_clip_actor,
                jacob_lr=jacob_lr,
            )
        elif critic_type == "mog":
            self.actor_updater = DistributionalMoGActorMPO(
                actor_optimizer=optim_actor,
                dual_optimizer=optim_dual,
                gradient_clip=grad_clip_actor,
                jacob_lr=jacob_lr,
            )
        elif critic_type == "default":
            self.actor_updater = JacMPOActor(
                actor_optimizer=optim_actor,
                dual_optimizer=optim_dual,
                gradient_clip=grad_clip_actor,
                jacob_lr=jacob_lr,
            )

        else:
            raise NotImplementedError

        if retnorm is not None:
            raise NotImplementedError
            self.model = custom_torso.custom_return_mpo(
                critic_size=critic_size, actor_size=actor_size
            )
        else:
            if critic_type == "c51":
                self.model = custom_torso.custom_c51_distributional_mpo(
                    critic_size=critic_size, actor_size=actor_size
                )
            elif critic_type == "mog":
                self.model = custom_torso.custom_mog_distributional_mpo(
                    critic_size=critic_size, actor_size=actor_size
                )
            else:
                self.model = custom_torso.custom_model_mpo(
                    critic_size=critic_size, actor_size=actor_size
                )
        if batch_size is not None:
            self.replay.batch_size = batch_size
        if return_steps is not None:
            self.replay.return_steps = return_steps

    def initialize(self, *args, **kwargs):
        super().initialize(*args, **kwargs)


class DistributionalC51Critic:
    def __init__(self, optimizer=None, gradient_clip=0, num_samples=20):
        self.optimizer = optimizer or (
            lambda params: torch.optim.Adam(params, lr=1e-3)
        )
        self.gradient_clip = gradient_clip
        self.num_samples = 20

    def initialize(self, model):
        self.model = model
        self.variables = models.trainable_variables(self.model.critic)
        self.optimizer = self.optimizer(self.variables)

    def __call__(
        self, observations, actions, next_observations, rewards, discounts
    ):
        # TODO does this line make sense?
        with torch.no_grad():
            # sample actions
            self.model.actor(next_observations)
            next_target_action_distributions = self.model.target_actor(
                next_observations
            )
            next_actions = next_target_action_distributions.rsample(
                (self.num_samples,)
            )

            # prepare dimensions
            next_actions = updaters.merge_first_two_dims(next_actions)
            next_observations = updaters.tile(
                next_observations, self.num_samples
            )
            next_observations = updaters.merge_first_two_dims(
                next_observations
            )
            # sample next value distribution
            next_value_distributions = self.model.target_critic(
                next_observations, next_actions
            )
            new_shape = [self.num_samples, observations.shape[0], -1]
            sampled_logits = torch.reshape(
                next_value_distributions.logits, new_shape
            )
            sampled_logprobs = torch.log_softmax(sampled_logits, dim=-1)
            # logsumexp of log(p) is just a sum over probabilities -> then back to log(p)
            # assume that log(sum p) and log(sum p/A) is similar as log(A) is just
            # a constant -> average over sampled policy actions num_samples
            averaged_logits = torch.logsumexp(sampled_logprobs, dim=0)
            next_value_distributions = CategoricalWithSupport(
                values=next_value_distributions.values, logits=averaged_logits
            )
            values = next_value_distributions.values
            returns = rewards[:, None] + discounts[:, None] * values
            targets = next_value_distributions.project(returns)

        self.optimizer.zero_grad()
        value_distributions = self.model.critic(observations, actions)
        log_probabilities = torch.nn.functional.log_softmax(
            value_distributions.logits, dim=-1
        )
        loss = -(targets * log_probabilities).sum(dim=-1).mean()

        loss.backward()
        if self.gradient_clip > 0:
            torch.nn.utils.clip_grad_norm_(self.variables, self.gradient_clip)
        self.optimizer.step()

        return dict(loss=loss.detach())


class DistributionalC51ActorMPO(TunedMPOActor):
    def __init__(self, jacob_lr=0.001, *args, **kwargs):
        """
        C51 Distributional MPO with Jacobian regularizer
        """
        self.jacob_lr = jacob_lr
        self.jac_reg = JacobianReg()
        super().__init__(*args, **kwargs)

    def __call__(self, observations):
        self.action_penalization = False

        def parametric_kl_and_dual_losses(kl, alpha, epsilon):
            kl_mean = kl.mean(dim=0)
            kl_loss = (alpha.detach() * kl_mean).sum()
            alpha_loss = (alpha * (epsilon - kl_mean.detach())).sum()
            return kl_loss, alpha_loss

        def weights_and_temperature_loss(q_values, epsilon, temperature):
            tempered_q_values = q_values.detach() / temperature
            weights = torch.nn.functional.softmax(tempered_q_values, dim=0)
            weights = weights.detach()

            # Temperature loss (dual of the E-step).
            q_log_sum_exp = torch.logsumexp(tempered_q_values, dim=0)
            num_actions = torch.as_tensor(
                q_values.shape[0], dtype=torch.float32
            )
            log_num_actions = torch.log(num_actions)
            loss = epsilon + (q_log_sum_exp).mean() - log_num_actions
            loss = temperature * loss

            return weights, loss

        # Use independent normals to satisfy KL constraints per-dimension.
        def independent_normals(distribution_1, distribution_2=None):
            distribution_2 = distribution_2 or distribution_1
            return torch.distributions.independent.Independent(
                torch.distributions.normal.Normal(
                    distribution_1.mean, distribution_2.stddev
                ),
                -1,
            )

        with torch.no_grad():
            self.log_temperature.data.copy_(
                torch.maximum(self.min_log_dual, self.log_temperature)
            )
            self.log_alpha_mean.data.copy_(
                torch.maximum(self.min_log_dual, self.log_alpha_mean)
            )
            self.log_alpha_std.data.copy_(
                torch.maximum(self.min_log_dual, self.log_alpha_std)
            )
            if self.action_penalization:
                self.log_penalty_temperature.data.copy_(
                    torch.maximum(
                        self.min_log_dual, self.log_penalty_temperature
                    )
                )

            target_distributions = self.model.target_actor(observations)
            actions = target_distributions.sample((self.num_samples,))

            tiled_observations = updaters.tile(observations, self.num_samples)
            flat_observations = updaters.merge_first_two_dims(
                tiled_observations
            )
            flat_actions = updaters.merge_first_two_dims(actions)
            values = self.model.target_critic(
                flat_observations, flat_actions
            ).mean()
            values = values.view(self.num_samples, -1)

            assert isinstance(
                target_distributions, torch.distributions.normal.Normal
            )
            target_distributions = independent_normals(target_distributions)

        self.actor_optimizer.zero_grad()
        self.dual_optimizer.zero_grad()

        distributions = self.model.actor(observations)
        distributions = independent_normals(distributions)

        temperature = (
            torch.nn.functional.softplus(self.log_temperature) + FLOAT_EPSILON
        )
        alpha_mean = (
            torch.nn.functional.softplus(self.log_alpha_mean) + FLOAT_EPSILON
        )
        alpha_std = (
            torch.nn.functional.softplus(self.log_alpha_std) + FLOAT_EPSILON
        )
        weights, temperature_loss = weights_and_temperature_loss(
            values, self.epsilon, temperature
        )

        # Action penalization is quadratic beyond [-1, 1].
        if self.action_penalization:
            penalty_temperature = (
                torch.nn.functional.softplus(self.log_penalty_temperature)
                + FLOAT_EPSILON
            )
            diff_bounds = actions - torch.clamp(actions, -1, 1)
            action_bound_costs = -torch.norm(diff_bounds, dim=-1)
            (
                penalty_weights,
                penalty_temperature_loss,
            ) = weights_and_temperature_loss(
                action_bound_costs, self.epsilon_penalty, penalty_temperature
            )
            weights += penalty_weights
            temperature_loss += penalty_temperature_loss

        # Decompose the policy into fixed-mean and fixed-std distributions.
        fixed_std_distribution = independent_normals(
            distributions.base_dist, target_distributions.base_dist
        )
        fixed_mean_distribution = independent_normals(
            target_distributions.base_dist, distributions.base_dist
        )

        # Compute the decomposed policy losses.
        policy_mean_losses = (
            fixed_std_distribution.base_dist.log_prob(actions).sum(dim=-1)
            * weights
        ).sum(dim=0)
        policy_mean_loss = -(policy_mean_losses).mean()
        policy_std_losses = (
            fixed_mean_distribution.base_dist.log_prob(actions).sum(dim=-1)
            * weights
        ).sum(dim=0)
        policy_std_loss = -policy_std_losses.mean()

        # Compute the decomposed KL between the target and online policies.
        if self.per_dim_constraining:
            kl_mean = torch.distributions.kl.kl_divergence(
                target_distributions.base_dist,
                fixed_std_distribution.base_dist,
            )
            kl_std = torch.distributions.kl.kl_divergence(
                target_distributions.base_dist,
                fixed_mean_distribution.base_dist,
            )
        else:
            kl_mean = torch.distributions.kl.kl_divergence(
                target_distributions, fixed_std_distribution
            )
            kl_std = torch.distributions.kl.kl_divergence(
                target_distributions, fixed_mean_distribution
            )

        # Compute the alpha-weighted KL-penalty and dual losses.
        kl_mean_loss, alpha_mean_loss = parametric_kl_and_dual_losses(
            kl_mean, alpha_mean, self.epsilon_mean
        )
        kl_std_loss, alpha_std_loss = parametric_kl_and_dual_losses(
            kl_std, alpha_std, self.epsilon_std
        )

        # Combine losses.
        policy_loss = policy_mean_loss + policy_std_loss
        kl_loss = kl_mean_loss + kl_std_loss
        dual_loss = alpha_mean_loss + alpha_std_loss + temperature_loss
        if np.abs(self.jacob_lr - 0) > 1e-5:
            observations.requires_grad = True
            actions_actor = self.model.actor(observations)
            output = torch.concat(
                [actions_actor.loc, actions_actor.scale], axis=-1
            )
            jac_loss = self.jac_reg(observations, output)
            loss = policy_loss + kl_loss + dual_loss + self.jacob_lr * jac_loss
        else:
            loss = policy_loss + kl_loss + dual_loss
        loss.backward()
        if self.gradient_clip > 0:
            torch.nn.utils.clip_grad_norm_(
                self.actor_variables, self.gradient_clip
            )
            torch.nn.utils.clip_grad_norm_(
                self.dual_variables, self.gradient_clip
            )
        self.actor_optimizer.step()
        self.dual_optimizer.step()

        dual_variables = dict(
            temperature=temperature.detach(),
            alpha_mean=alpha_mean.detach(),
            alpha_std=alpha_std.detach(),
        )
        if self.action_penalization:
            dual_variables[
                "penalty_temperature"
            ] = penalty_temperature.detach()

        return dict(
            policy_mean_loss=policy_mean_loss.detach(),
            policy_std_loss=policy_std_loss.detach(),
            kl_mean_loss=kl_mean_loss.detach(),
            kl_std_loss=kl_std_loss.detach(),
            alpha_mean_loss=alpha_mean_loss.detach(),
            alpha_std_loss=alpha_std_loss.detach(),
            temperature_loss=temperature_loss.detach(),
            **dual_variables,
        )


class DistributionalMoGActorMPO(TunedMPOActor):
    def __init__(self, jacob_lr=0.001, *args, **kwargs):
        """
        MoG Distributional MPO with Jacobian regularizer
        """
        self.jacob_lr = jacob_lr
        self.jac_reg = JacobianReg()
        super().__init__(*args, **kwargs)

    def __call__(self, observations):
        self.action_penalization = False

        def parametric_kl_and_dual_losses(kl, alpha, epsilon):
            kl_mean = kl.mean(dim=0)
            kl_loss = (alpha.detach() * kl_mean).sum()
            alpha_loss = (alpha * (epsilon - kl_mean.detach())).sum()
            return kl_loss, alpha_loss

        def weights_and_temperature_loss(q_values, epsilon, temperature):
            tempered_q_values = q_values.detach() / temperature
            weights = torch.nn.functional.softmax(tempered_q_values, dim=0)
            weights = weights.detach()

            # Temperature loss (dual of the E-step).
            q_log_sum_exp = torch.logsumexp(tempered_q_values, dim=0)
            num_actions = torch.as_tensor(
                q_values.shape[0], dtype=torch.float32
            )
            log_num_actions = torch.log(num_actions)
            loss = epsilon + (q_log_sum_exp).mean() - log_num_actions
            loss = temperature * loss

            return weights, loss

        # Use independent normals to satisfy KL constraints per-dimension.
        def independent_normals(distribution_1, distribution_2=None):
            distribution_2 = distribution_2 or distribution_1
            return torch.distributions.independent.Independent(
                torch.distributions.normal.Normal(
                    distribution_1.mean, distribution_2.stddev
                ),
                -1,
            )

        with torch.no_grad():
            self.log_temperature.data.copy_(
                torch.maximum(self.min_log_dual, self.log_temperature)
            )
            self.log_alpha_mean.data.copy_(
                torch.maximum(self.min_log_dual, self.log_alpha_mean)
            )
            self.log_alpha_std.data.copy_(
                torch.maximum(self.min_log_dual, self.log_alpha_std)
            )
            if self.action_penalization:
                self.log_penalty_temperature.data.copy_(
                    torch.maximum(
                        self.min_log_dual, self.log_penalty_temperature
                    )
                )

            target_distributions = self.model.target_actor(observations)
            actions = target_distributions.sample((self.num_samples,))

            tiled_observations = updaters.tile(observations, self.num_samples)
            flat_observations = updaters.merge_first_two_dims(
                tiled_observations
            )
            flat_actions = updaters.merge_first_two_dims(actions)
            values = self.model.target_critic(
                flat_observations, flat_actions
            ).mean
            values = values.view(self.num_samples, -1)

            assert isinstance(
                target_distributions, torch.distributions.normal.Normal
            )
            target_distributions = independent_normals(target_distributions)

        self.actor_optimizer.zero_grad()
        self.dual_optimizer.zero_grad()

        distributions = self.model.actor(observations)
        distributions = independent_normals(distributions)

        temperature = (
            torch.nn.functional.softplus(self.log_temperature) + FLOAT_EPSILON
        )
        alpha_mean = (
            torch.nn.functional.softplus(self.log_alpha_mean) + FLOAT_EPSILON
        )
        alpha_std = (
            torch.nn.functional.softplus(self.log_alpha_std) + FLOAT_EPSILON
        )
        weights, temperature_loss = weights_and_temperature_loss(
            values, self.epsilon, temperature
        )

        # Action penalization is quadratic beyond [-1, 1].
        if self.action_penalization:
            penalty_temperature = (
                torch.nn.functional.softplus(self.log_penalty_temperature)
                + FLOAT_EPSILON
            )
            diff_bounds = actions - torch.clamp(actions, -1, 1)
            action_bound_costs = -torch.norm(diff_bounds, dim=-1)
            (
                penalty_weights,
                penalty_temperature_loss,
            ) = weights_and_temperature_loss(
                action_bound_costs, self.epsilon_penalty, penalty_temperature
            )
            weights += penalty_weights
            temperature_loss += penalty_temperature_loss

        # Decompose the policy into fixed-mean and fixed-std distributions.
        fixed_std_distribution = independent_normals(
            distributions.base_dist, target_distributions.base_dist
        )
        fixed_mean_distribution = independent_normals(
            target_distributions.base_dist, distributions.base_dist
        )

        # Compute the decomposed policy losses.
        policy_mean_losses = (
            fixed_std_distribution.base_dist.log_prob(actions).sum(dim=-1)
            * weights
        ).sum(dim=0)
        policy_mean_loss = -(policy_mean_losses).mean()
        policy_std_losses = (
            fixed_mean_distribution.base_dist.log_prob(actions).sum(dim=-1)
            * weights
        ).sum(dim=0)
        policy_std_loss = -policy_std_losses.mean()

        # Compute the decomposed KL between the target and online policies.
        if self.per_dim_constraining:
            kl_mean = torch.distributions.kl.kl_divergence(
                target_distributions.base_dist,
                fixed_std_distribution.base_dist,
            )
            kl_std = torch.distributions.kl.kl_divergence(
                target_distributions.base_dist,
                fixed_mean_distribution.base_dist,
            )
        else:
            kl_mean = torch.distributions.kl.kl_divergence(
                target_distributions, fixed_std_distribution
            )
            kl_std = torch.distributions.kl.kl_divergence(
                target_distributions, fixed_mean_distribution
            )

        # Compute the alpha-weighted KL-penalty and dual losses.
        kl_mean_loss, alpha_mean_loss = parametric_kl_and_dual_losses(
            kl_mean, alpha_mean, self.epsilon_mean
        )
        kl_std_loss, alpha_std_loss = parametric_kl_and_dual_losses(
            kl_std, alpha_std, self.epsilon_std
        )

        # Combine losses.
        policy_loss = policy_mean_loss + policy_std_loss
        kl_loss = kl_mean_loss + kl_std_loss
        dual_loss = alpha_mean_loss + alpha_std_loss + temperature_loss
        if np.abs(self.jacob_lr - 0) > 1e-5:
            observations.requires_grad = True
            actions_actor = self.model.actor(observations)
            output = torch.concat(
                [actions_actor.loc, actions_actor.scale], axis=-1
            )
            jac_loss = self.jac_reg(observations, output)
            loss = policy_loss + kl_loss + dual_loss + self.jacob_lr * jac_loss
        else:
            loss = policy_loss + kl_loss + dual_loss
        loss.backward()
        if self.gradient_clip > 0:
            torch.nn.utils.clip_grad_norm_(
                self.actor_variables, self.gradient_clip
            )
            torch.nn.utils.clip_grad_norm_(
                self.dual_variables, self.gradient_clip
            )
        self.actor_optimizer.step()
        self.dual_optimizer.step()

        dual_variables = dict(
            temperature=temperature.detach(),
            alpha_mean=alpha_mean.detach(),
            alpha_std=alpha_std.detach(),
        )
        if self.action_penalization:
            dual_variables[
                "penalty_temperature"
            ] = penalty_temperature.detach()

        return dict(
            policy_mean_loss=policy_mean_loss.detach(),
            policy_std_loss=policy_std_loss.detach(),
            kl_mean_loss=kl_mean_loss.detach(),
            kl_std_loss=kl_std_loss.detach(),
            alpha_mean_loss=alpha_mean_loss.detach(),
            alpha_std_loss=alpha_std_loss.detach(),
            temperature_loss=temperature_loss.detach(),
            mean_q_values=torch.mean(values).detach(),
            **dual_variables,
        )


class DistributionalMoGCritic:
    def __init__(self, optimizer=None, gradient_clip=0, num_samples=20):
        """
        Distributional MoG Critic for MPO
        """
        self.optimizer = optimizer or (
            lambda params: torch.optim.Adam(params, lr=1e-3)
        )
        self.gradient_clip = gradient_clip
        self.num_samples = 20
        self.num_value_samples = 128
        self.num_joint_samples = self.num_samples * self.num_value_samples

    def initialize(self, model):
        logger.log("MoG Critic active!")
        self.model = model
        self.variables = models.trainable_variables(self.model.critic)
        self.optimizer = self.optimizer(self.variables)

    def __call__(
        self, observations, actions, next_observations, rewards, discounts
    ):
        with torch.no_grad():
            # sample actions
            self.model.actor(next_observations)
            next_target_action_distributions = self.model.target_actor(
                next_observations
            )
            next_actions = next_target_action_distributions.rsample(
                (self.num_samples,)
            )
            # prepare dimensions; next_actions are sampled from policy -> tiling
            next_actions = updaters.merge_first_two_dims(next_actions)
            next_observations = updaters.tile(
                next_observations, self.num_samples
            )
            next_observations = updaters.merge_first_two_dims(
                next_observations
            )
            # sample next value distribution
            next_value_distributions = self.model.target_critic(
                next_observations, next_actions
            )

            z_distributions = next_value_distributions
            z_samples = z_distributions.sample((self.num_value_samples,))
            z_samples = torch.reshape(
                z_samples, (self.num_joint_samples, -1, 1)
            )

            target_q = rewards[:, None] + discounts[:, None] * z_samples
        # from here on we need grad
        self.optimizer.zero_grad()
        # sample current value distribution; no tiling because actions are not sampled
        current_value_distribution = self.model.critic(observations, actions)
        target_q.requires_grad = True
        log_probs_q = current_value_distribution.log_prob(target_q)
        loss = -log_probs_q.mean()
        loss.backward()

        if self.gradient_clip > 0:
            torch.nn.utils.clip_grad_norm_(self.variables, self.gradient_clip)
        self.optimizer.step()

        return dict(loss=loss.detach())
