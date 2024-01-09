from typing import Sequence

import torch

from deprl.vendor.tonic.torch import models, normalizers


def custom_model_mpo(
    actor_size: Sequence[int] = [256, 256],
    critic_size: Sequence[int] = [256, 256],
):
    return models.ActorCriticWithTargets(
        actor=models.Actor(
            encoder=models.ObservationEncoder(),
            torso=models.MLP(actor_size, torch.nn.ReLU),
            head=models.GaussianPolicyHead(),
        ),
        critic=models.Critic(
            encoder=models.ObservationActionEncoder(),
            torso=models.MLP(critic_size, torch.nn.ReLU),
            head=models.ValueHead(),
        ),
        observation_normalizer=normalizers.MeanStd(),
    )


def custom_c51_distributional_mpo(
    actor_size: Sequence[int] = [256, 256],
    critic_size: Sequence[int] = [256, 256],
):
    return models.ActorCriticWithTargets(
        actor=models.Actor(
            encoder=models.ObservationEncoder(),
            torso=models.MLP(actor_size, torch.nn.ReLU),
            head=models.GaussianPolicyHead(),
        ),
        critic=models.Critic(
            encoder=models.ObservationActionEncoder(),
            torso=models.MLP(critic_size, torch.nn.ReLU),
            head=models.DistributionalValueHead(-150.0, 150.0, 51),
        ),
        observation_normalizer=normalizers.MeanStd(),
    )


def custom_mog_distributional_mpo(
    actor_size: Sequence[int] = [256, 256],
    critic_size: Sequence[int] = [256, 256],
):
    return models.ActorCriticWithTargets(
        actor=models.Actor(
            encoder=models.ObservationEncoder(),
            torso=models.MLP(actor_size, torch.nn.ReLU),
            head=models.GaussianPolicyHead(),
        ),
        critic=models.Critic(
            encoder=models.ObservationActionEncoder(),
            torso=models.MLP(critic_size, torch.nn.ReLU),
            head=models.critics.DistributionalMoGValueHead(
                num_dimensions=1, init_scale=1e-3, num_components=5
            ),
        ),
        observation_normalizer=normalizers.MeanStd(),
    )


def custom_return_distributional_mpo(
    actor_size: Sequence[int] = [256, 256],
    critic_size: Sequence[int] = [256, 256],
):
    return models.ActorCriticWithTargets(
        actor=models.Actor(
            encoder=models.ObservationEncoder(),
            torso=models.MLP(actor_size, torch.nn.ReLU),
            head=models.GaussianPolicyHead(),
        ),
        critic=models.Critic(
            encoder=models.ObservationActionEncoder(),
            torso=models.MLP(critic_size, torch.nn.ReLU),
            head=models.DistributionalValueHead(-150.0, 150.0, 51),
        ),
        observation_normalizer=normalizers.MeanStd(),
        return_normalizer=normalizers.returns.Return(0.99),
    )


def custom_return_mpo(hidden_size: Sequence[int] = [256, 256]):
    return models.ActorCriticWithTargets(
        actor=models.Actor(
            encoder=models.ObservationEncoder(),
            torso=models.MLP(hidden_size, torch.nn.ReLU),
            head=models.GaussianPolicyHead(),
        ),
        critic=models.Critic(
            encoder=models.ObservationActionEncoder(),
            torso=models.MLP(hidden_size, torch.nn.ReLU),
            head=models.ValueHead(),
        ),
        observation_normalizer=normalizers.MeanStd(),
        return_normalizer=normalizers.returns.Return(0.99),
    )
