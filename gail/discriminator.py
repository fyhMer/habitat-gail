#!/usr/bin/env python3

# Created by Yunhai Feng at 4:23 pm, 2022/4/30.


from typing import Dict, Tuple, Sequence, Union

import numpy as np
from gym import spaces

import torch
from torch import Tensor
import torch.nn as nn
from torch import optim as optim

from habitat.utils import profiling_wrapper

from habitat.tasks.nav.nav import (
    EpisodicCompassSensor,
    EpisodicGPSSensor,
    HeadingSensor,
    ImageGoalSensor,
    IntegratedPointGoalGPSAndCompassSensor,
    PointGoalSensor,
    ProximitySensor,
)
from habitat.tasks.nav.object_nav_task import ObjectGoalSensor
from habitat_baselines.rl.ddppo.policy import resnet
from habitat_baselines.rl.ddppo.policy.resnet_policy import ResNetEncoder
from habitat_baselines.rl.models.rnn_state_encoder import (
    build_rnn_state_encoder,
)
from gail.common.rollout_storage import RolloutStorage


class DiscriminatorNet(nn.Module):
    """Network which passes the input image through CNN and concatenates
    goal vector with CNN's output and passes that through RNN.
    """
    # Yunhai's Notes: Modified from PointNavResNetNet

    prev_action_embedding: nn.Module
    curr_action_embedding: nn.Module

    def __init__(
        self,
        observation_space: spaces.Dict,
        action_space,
        hidden_size: int,
        backbone,
        resnet_baseplanes,
        normalize_visual_inputs: bool,
        force_blind_policy: bool = False,
        discrete_actions: bool = True,
        use_rnn: bool = False,
        num_recurrent_layers: int = 1,
        rnn_type: str = "GRU",
    ):
        super().__init__()
        self.prev_action_embedding: nn.Module
        self.curr_action_embedding: nn.Module
        self.discrete_actions = discrete_actions
        self._n_action = 32
        if discrete_actions:
            self.prev_action_embedding = nn.Embedding(action_space.n + 1, self._n_action)
            self.curr_action_embedding = nn.Embedding(action_space.n + 1, self._n_action)
        else:
            self.prev_action_embedding = nn.Linear(action_space.n, self._n_action)
            self.curr_action_embedding = nn.Linear(action_space.n, self._n_action)

        total_embedding_size = self._n_action * 2

        if (
            IntegratedPointGoalGPSAndCompassSensor.cls_uuid
            in observation_space.spaces
        ):
            n_input_goal = (
                observation_space.spaces[
                    IntegratedPointGoalGPSAndCompassSensor.cls_uuid
                ].shape[0]
                + 1
            )
            self.tgt_embeding = nn.Linear(n_input_goal, 32)
            total_embedding_size += 32

        if ObjectGoalSensor.cls_uuid in observation_space.spaces:
            self._n_object_categories = (
                int(
                    observation_space.spaces[ObjectGoalSensor.cls_uuid].high[0]
                )
                + 1
            )
            # self._n_object_categories = 29 # TODO: add param in config
            self.obj_categories_embedding = nn.Embedding(
                self._n_object_categories, 32
            )
            total_embedding_size += 32

        if EpisodicGPSSensor.cls_uuid in observation_space.spaces:
            input_gps_dim = observation_space.spaces[
                EpisodicGPSSensor.cls_uuid
            ].shape[0]
            self.gps_embedding = nn.Linear(input_gps_dim, 32)
            total_embedding_size += 32

        if PointGoalSensor.cls_uuid in observation_space.spaces:
            input_pointgoal_dim = observation_space.spaces[
                PointGoalSensor.cls_uuid
            ].shape[0]
            self.pointgoal_embedding = nn.Linear(input_pointgoal_dim, 32)
            total_embedding_size += 32

        if HeadingSensor.cls_uuid in observation_space.spaces:
            input_heading_dim = (
                observation_space.spaces[HeadingSensor.cls_uuid].shape[0] + 1
            )
            assert input_heading_dim == 2, "Expected heading with 2D rotation."
            self.heading_embedding = nn.Linear(input_heading_dim, 32)
            total_embedding_size += 32

        if ProximitySensor.cls_uuid in observation_space.spaces:
            input_proximity_dim = observation_space.spaces[
                ProximitySensor.cls_uuid
            ].shape[0]
            self.proximity_embedding = nn.Linear(input_proximity_dim, 32)
            total_embedding_size += 32

        if EpisodicCompassSensor.cls_uuid in observation_space.spaces:
            assert (
                observation_space.spaces[EpisodicCompassSensor.cls_uuid].shape[
                    0
                ]
                == 1
            ), "Expected compass with 2D rotation."
            input_compass_dim = 2  # cos and sin of the angle
            self.compass_embedding = nn.Linear(input_compass_dim, 32)
            total_embedding_size += 32

        if ImageGoalSensor.cls_uuid in observation_space.spaces:
            goal_observation_space = spaces.Dict(
                {"rgb": observation_space.spaces[ImageGoalSensor.cls_uuid]}
            )
            self.goal_visual_encoder = ResNetEncoder(
                goal_observation_space,
                baseplanes=resnet_baseplanes,
                ngroups=resnet_baseplanes // 2,
                make_backbone=getattr(resnet, backbone),
                normalize_visual_inputs=normalize_visual_inputs,
            )

            self.goal_visual_fc = nn.Sequential(
                nn.Flatten(),
                nn.Linear(
                    np.prod(self.goal_visual_encoder.output_shape), hidden_size
                ),
                nn.ReLU(True),
            )

            total_embedding_size += hidden_size

        self._hidden_size = hidden_size

        self.visual_encoder = ResNetEncoder(
            observation_space if not force_blind_policy else spaces.Dict({}),
            baseplanes=resnet_baseplanes,
            ngroups=resnet_baseplanes // 2,
            make_backbone=getattr(resnet, backbone),
            normalize_visual_inputs=normalize_visual_inputs,
        )

        if not self.visual_encoder.is_blind:
            self.visual_fc = nn.Sequential(
                nn.Flatten(),
                nn.Linear(
                    np.prod(self.visual_encoder.output_shape), hidden_size
                ),
                nn.ReLU(True),
            )

            total_embedding_size += hidden_size

        self._use_rnn = use_rnn
        if use_rnn:
            self.state_encoder = build_rnn_state_encoder(
                total_embedding_size,
                self._hidden_size,
                rnn_type=rnn_type,
                num_layers=num_recurrent_layers,
            )
            total_embedding_size = self._hidden_size

        self.fc_layers = nn.Sequential(
            nn.Linear(total_embedding_size, 256),
            nn.ReLU(inplace=True),
            nn.Linear(256, 128),
            nn.ReLU(inplace=True),
            nn.Linear(128, 1),
            nn.Sigmoid()
        )

        self.train()

    @property
    def output_size(self):
        # return self._hidden_size
        return 1

    @property
    def is_blind(self):
        return self.visual_encoder.is_blind

    def forward(
        self,
        observations: Dict[str, torch.Tensor],
        prev_actions,
        masks,
        curr_actions,
        rnn_hidden_states=None
    ) -> Union[torch.Tensor, Tuple[torch.Tensor, torch.Tensor]]:
        x = []
        if not self.is_blind:
            visual_feats = observations.get(
                "visual_features", self.visual_encoder(observations)
            )
            visual_feats = self.visual_fc(visual_feats)
            x.append(visual_feats)

        if IntegratedPointGoalGPSAndCompassSensor.cls_uuid in observations:
            goal_observations = observations[
                IntegratedPointGoalGPSAndCompassSensor.cls_uuid
            ]
            if goal_observations.shape[1] == 2:
                # Polar Dimensionality 2
                # 2D polar transform
                goal_observations = torch.stack(
                    [
                        goal_observations[:, 0],
                        torch.cos(-goal_observations[:, 1]),
                        torch.sin(-goal_observations[:, 1]),
                    ],
                    -1,
                )
            else:
                assert (
                    goal_observations.shape[1] == 3
                ), "Unsupported dimensionality"
                vertical_angle_sin = torch.sin(goal_observations[:, 2])
                # Polar Dimensionality 3
                # 3D Polar transformation
                goal_observations = torch.stack(
                    [
                        goal_observations[:, 0],
                        torch.cos(-goal_observations[:, 1])
                        * vertical_angle_sin,
                        torch.sin(-goal_observations[:, 1])
                        * vertical_angle_sin,
                        torch.cos(goal_observations[:, 2]),
                    ],
                    -1,
                )

            x.append(self.tgt_embeding(goal_observations))

        if PointGoalSensor.cls_uuid in observations:
            goal_observations = observations[PointGoalSensor.cls_uuid]
            x.append(self.pointgoal_embedding(goal_observations))

        if ProximitySensor.cls_uuid in observations:
            sensor_observations = observations[ProximitySensor.cls_uuid]
            x.append(self.proximity_embedding(sensor_observations))

        if HeadingSensor.cls_uuid in observations:
            sensor_observations = observations[HeadingSensor.cls_uuid]
            sensor_observations = torch.stack(
                [
                    torch.cos(sensor_observations[0]),
                    torch.sin(sensor_observations[0]),
                ],
                -1,
            )
            x.append(self.heading_embedding(sensor_observations))

        if ObjectGoalSensor.cls_uuid in observations:
            object_goal = observations[ObjectGoalSensor.cls_uuid].long()
            # print("******** _n_object_categories:", self._n_object_categories)
            # print("object_goal", object_goal)
            x.append(self.obj_categories_embedding(object_goal).squeeze(dim=1))

        if EpisodicCompassSensor.cls_uuid in observations:
            compass_observations = torch.stack(
                [
                    torch.cos(observations[EpisodicCompassSensor.cls_uuid]),
                    torch.sin(observations[EpisodicCompassSensor.cls_uuid]),
                ],
                -1,
            )
            x.append(
                self.compass_embedding(compass_observations.squeeze(dim=1))
            )

        if EpisodicGPSSensor.cls_uuid in observations:
            x.append(
                self.gps_embedding(observations[EpisodicGPSSensor.cls_uuid])
            )

        if ImageGoalSensor.cls_uuid in observations:
            goal_image = observations[ImageGoalSensor.cls_uuid]
            goal_output = self.goal_visual_encoder({"rgb": goal_image})
            x.append(self.goal_visual_fc(goal_output))

        if self.discrete_actions:
            prev_actions = prev_actions.squeeze(-1)
            start_token = torch.zeros_like(prev_actions)
            prev_actions = self.prev_action_embedding(
                torch.where(masks.view(-1), prev_actions + 1, start_token)
                # check why +1 here: https://github.com/facebookresearch/habitat-lab/issues/528
            )
            curr_actions = curr_actions.squeeze(-1)
            curr_actions = self.curr_action_embedding(curr_actions + 1)
        else:
            prev_actions = self.prev_action_embedding(
                masks * prev_actions.float()
            )
            curr_actions = self.curr_action_embedding(curr_actions.float())

        x.append(prev_actions)
        x.append(curr_actions)

        # for xx in x:
        #     print(xx.shape)

        feature_embedding = torch.cat(x, dim=1)
        # print("feature size", feature_embedding.shape)

        if self._use_rnn:
            assert rnn_hidden_states is not None
            feature_embedding, rnn_hidden_states = self.state_encoder(
                feature_embedding,
                rnn_hidden_states,
                masks
            )
            out = self.fc_layers(feature_embedding)
            return out, rnn_hidden_states
        else:
            out = self.fc_layers(feature_embedding)
            return out


class Discriminator(nn.Module):

    def __init__(self,
                 net: DiscriminatorNet,
                 discriminator_epoch: int = 8,
                 num_mini_batch: int = 2,
                 lr: float = 1e-4,
                 eps: float = 1e-5,
                 max_grad_norm: float = 0.5
                 ):
        super(Discriminator, self).__init__()
        self.net = net
        self.discriminator_epoch = discriminator_epoch
        self.num_mini_batch = num_mini_batch
        self.max_grad_norm = max_grad_norm

        self.optimizer = optim.Adam(
            list(filter(lambda p: p.requires_grad, net.parameters())),
            lr=lr,
            eps=eps,
        )

    def forward(self, *x):
        raise NotImplementedError

    def get_reward(self, *args, **kwargs) -> torch.Tensor:
        # TODO: tune parameters
        if "discriminator_score" not in kwargs:
            if self.net._use_rnn:
                d, _ = self.net.forward(*args, **kwargs)
            else:
                d = self.net.forward(*args, **kwargs)
        else:
            d = kwargs["discriminator_score"]
        reward = -torch.log(
            torch.clip(torch.ones_like(d) - d, min=1e-6)
        )
        return torch.clip(reward, max=2.0) * 0.1

    def update(self, rollouts: RolloutStorage):

        loss_agent_epoch = 0.0
        loss_demo_epoch = 0.0
        total_loss_epoch = 0.0

        for _e in range(self.discriminator_epoch):
            profiling_wrapper.range_push("Discriminator.update epoch")
            data_generator = rollouts.gail_data_generator(self.num_mini_batch)

            self.net.train()  # TODO: check this
            for batch_agent, batch_demo, agent_env_inds, demo_env_inds in data_generator:
                agent_discrim_ret = self.net(
                    observations=batch_agent["observations"],
                    prev_actions=batch_agent["prev_actions"],
                    masks=batch_agent["masks"],
                    curr_actions=batch_agent["actions"],
                    rnn_hidden_states=batch_agent["discrim_start_hidden_states"]
                )
                if self.net._use_rnn:
                    agent_score, agent_discrim_hidden_states = agent_discrim_ret
                    if _e + 1 == self.discriminator_epoch:
                        rollouts.insert_dscrim_start_hidden_states(agent_discrim_hidden_states, agent_env_inds)
                else:
                    agent_score = agent_discrim_ret

                # TODO: check expert prev action
                demo_discrim_ret = self.net(
                    observations=batch_demo["observations"],
                    prev_actions=batch_demo["prev_actions"],
                    masks=batch_demo["masks"],
                    curr_actions=batch_demo["actions"],
                    rnn_hidden_states=batch_demo["discrim_start_hidden_states"]
                )
                if self.net._use_rnn:
                    demo_score, demo_discrim_hidden_states = demo_discrim_ret
                    if _e + 1 == self.discriminator_epoch:
                        rollouts.insert_dscrim_start_hidden_states(demo_discrim_hidden_states, demo_env_inds)
                else:
                    demo_score = demo_discrim_ret

                loss_agent = -torch.log(
                    torch.clip(torch.ones_like(agent_score) - agent_score, min=1e-6)
                ).mean()
                loss_demo = -torch.log(
                    torch.clip(demo_score, min=1e-6)
                ).mean()

                self.optimizer.zero_grad()
                total_loss = loss_demo + loss_agent

                self.before_backward(total_loss)
                total_loss.backward()
                self.after_backward(total_loss)

                self.before_step()
                self.optimizer.step()
                self.after_step()

                loss_agent_epoch += loss_agent.item()
                loss_demo_epoch += loss_demo.item()
                total_loss_epoch += total_loss.item()

            profiling_wrapper.range_pop()  # Discriminator.update epoch

        num_updates = self.discriminator_epoch * self.num_mini_batch

        loss_agent_epoch /= num_updates
        loss_demo_epoch /= num_updates
        total_loss_epoch /= num_updates

        return loss_agent_epoch, loss_demo_epoch, total_loss_epoch

    def before_backward(self, loss: Tensor) -> None:
        pass

    def after_backward(self, loss: Tensor) -> None:
        pass

    def before_step(self) -> None:
        nn.utils.clip_grad_norm_(
            self.net.parameters(), self.max_grad_norm
        )

    def after_step(self) -> None:
        pass
