#!/usr/bin/env python3

# Created by Yunhai Feng at 4:05 pm, 2022/5/2.

import warnings
from typing import Iterator, Optional, Tuple

import numpy as np
import torch

from habitat_baselines.common.tensor_dict import TensorDict


class RolloutStorage:
    r"""Class for storing rollout information for RL trainers."""

    def __init__(
        self,
        numsteps,
        num_envs,
        observation_space,
        action_space,
        recurrent_hidden_state_size,
        num_recurrent_layers=1,
        action_shape: Optional[Tuple[int]] = None,
        is_double_buffered: bool = False,
        demo_buffer_idx: int = 1, # index of buffer that collect demonstration rollouts
        discrete_actions: bool = True,
    ):
        self.buffers = TensorDict()
        self.buffers["observations"] = TensorDict()

        for sensor in observation_space.spaces:
            self.buffers["observations"][sensor] = torch.from_numpy(
                np.zeros(
                    (
                        numsteps + 1,
                        num_envs,
                        *observation_space.spaces[sensor].shape,
                    ),
                    dtype=observation_space.spaces[sensor].dtype,
                )
            )

        self.buffers["recurrent_hidden_states"] = torch.zeros(
            numsteps + 1,
            num_envs,
            num_recurrent_layers,
            recurrent_hidden_state_size,
        )

        # for rnn in discriminator
        self.buffers["discrim_start_hidden_states"] = torch.zeros(
            numsteps + 1, # TODO: only index 0 is used, optimize in future
            num_envs,
            num_recurrent_layers,
            recurrent_hidden_state_size
        )

        # self.buffers["rewards"] = torch.zeros(numsteps + 1, num_envs, 1)
        self.buffers["total_rewards"] = torch.zeros(numsteps + 1, num_envs, 1)
        self.buffers["task_rewards"] = torch.zeros(numsteps + 1, num_envs, 1)
        self.buffers["gail_rewards"] = torch.zeros(numsteps + 1, num_envs, 1)
        self.buffers["value_preds"] = torch.zeros(numsteps + 1, num_envs, 1)
        self.buffers["returns"] = torch.zeros(numsteps + 1, num_envs, 1)

        self.buffers["action_log_probs"] = torch.zeros(
            numsteps + 1, num_envs, 1
        )

        if action_shape is None:
            if action_space.__class__.__name__ == "ActionSpace":
                action_shape = (1,)
            else:
                action_shape = action_space.shape

        self.action_shape = action_shape

        self.buffers["actions"] = torch.zeros(
            numsteps + 1, num_envs, *action_shape
        )
        self.buffers["prev_actions"] = torch.zeros(
            numsteps + 1, num_envs, *action_shape
        )
        if (
            discrete_actions
            and action_space.__class__.__name__ == "ActionSpace"
        ):
            assert isinstance(self.buffers["actions"], torch.Tensor)
            assert isinstance(self.buffers["prev_actions"], torch.Tensor)
            self.buffers["actions"] = self.buffers["actions"].long()
            self.buffers["prev_actions"] = self.buffers["prev_actions"].long()

        self.buffers["masks"] = torch.zeros(
            numsteps + 1, num_envs, 1, dtype=torch.bool
        )

        self.buffers["is_demo"] = torch.zeros(
            numsteps + 1, num_envs, 1, dtype=torch.bool
        )

        self.is_double_buffered = is_double_buffered
        self._nbuffers = 2 if is_double_buffered else 1
        self._demo_buffer_idx = demo_buffer_idx  # *****
        self._num_envs = num_envs

        assert (self._num_envs % self._nbuffers) == 0

        self.numsteps = numsteps
        self.current_rollout_step_idxs = [0 for _ in range(self._nbuffers)]

    @property
    def current_rollout_step_idx(self) -> int:
        assert all(
            s == self.current_rollout_step_idxs[0]
            for s in self.current_rollout_step_idxs
        )
        return self.current_rollout_step_idxs[0]

    def to(self, device):
        self.buffers.map_in_place(lambda v: v.to(device))

    def insert_dscrim_start_hidden_states(
        self,
        dscrim_start_hidden_states,
        env_slice
    ):
        self.buffers.set(
            (self.current_rollout_step_idx, env_slice),
            {"discrim_start_hidden_states": dscrim_start_hidden_states},
            strict=False,
        )

    def insert(
        self,
        next_observations=None,
        next_recurrent_hidden_states=None,
        actions=None,
        action_log_probs=None,
        value_preds=None,
        task_rewards=None,
        gail_rewards=None,
        total_rewards=None,
        next_masks=None,
        buffer_index: int = 0,
    ):
        if not self.is_double_buffered:
            assert buffer_index == 0

        next_step = dict(
            observations=next_observations,
            recurrent_hidden_states=next_recurrent_hidden_states,
            prev_actions=actions,
            masks=next_masks,
        )

        current_step = dict(
            actions=actions,
            action_log_probs=action_log_probs,
            value_preds=value_preds,
            task_rewards=task_rewards,
            gail_rewards=gail_rewards,
            total_rewards=total_rewards,
        )

        next_step = {k: v for k, v in next_step.items() if v is not None}
        current_step = {k: v for k, v in current_step.items() if v is not None}

        env_slice = slice(
            int(buffer_index * self._num_envs / self._nbuffers),
            int((buffer_index + 1) * self._num_envs / self._nbuffers),
        )

        if len(next_step) > 0:
            self.buffers.set(
                (self.current_rollout_step_idxs[buffer_index] + 1, env_slice),
                next_step,
                strict=False,
            )

        if len(current_step) > 0:
            self.buffers.set(
                (self.current_rollout_step_idxs[buffer_index], env_slice),
                current_step,
                strict=False,
            )

    def advance_rollout(self, buffer_index: int = 0):
        self.current_rollout_step_idxs[buffer_index] += 1

    def after_update(self):
        self.buffers[0] = self.buffers[self.current_rollout_step_idx]

        self.current_rollout_step_idxs = [
            0 for _ in self.current_rollout_step_idxs
        ]

    def compute_returns(self, next_value, use_gae, gamma, tau):
        if use_gae:
            assert isinstance(self.buffers["value_preds"], torch.Tensor)
            self.buffers["value_preds"][
                self.current_rollout_step_idx
            ] = next_value
            gae = 0.0
            for step in reversed(range(self.current_rollout_step_idx)):
                delta = (
                    self.buffers["total_rewards"][step]
                    + gamma
                    * self.buffers["value_preds"][step + 1]
                    * self.buffers["masks"][step + 1]
                    - self.buffers["value_preds"][step]
                )
                gae = (
                    delta + gamma * tau * gae * self.buffers["masks"][step + 1]
                )
                self.buffers["returns"][step] = (  # type: ignore
                    gae + self.buffers["value_preds"][step]  # type: ignore
                )
        else:
            self.buffers["returns"][self.current_rollout_step_idx] = next_value
            for step in reversed(range(self.current_rollout_step_idx)):
                self.buffers["returns"][step] = (
                    gamma
                    * self.buffers["returns"][step + 1]
                    * self.buffers["masks"][step + 1]
                    + self.buffers["total_rewards"][step]
                )

    def recurrent_generator(
        self, advantages, num_mini_batch
    ) -> Iterator[TensorDict]:
        num_environments = advantages.size(1)
        assert num_environments >= num_mini_batch, (
            "Trainer requires the number of environments ({}) "
            "to be greater than or equal to the number of "
            "trainer mini batches ({}).".format(
                num_environments, num_mini_batch
            )
        )
        if num_environments % num_mini_batch != 0:
            warnings.warn(
                "Number of environments ({}) is not a multiple of the"
                " number of mini batches ({}).  This results in mini batches"
                " of different sizes, which can harm training performance.".format(
                    num_environments, num_mini_batch
                )
            )
        for inds in torch.randperm(num_environments).chunk(num_mini_batch):
            batch = self.buffers[0 : self.current_rollout_step_idx, inds]
            batch["advantages"] = advantages[
                0 : self.current_rollout_step_idx, inds
            ]
            batch["recurrent_hidden_states"] = batch[
                "recurrent_hidden_states"
            ][0:1]

            yield batch.map(lambda v: v.flatten(0, 1))

    def gail_data_generator(
        self, num_mini_batch
    ) -> Iterator[TensorDict]: # TODO: return type
        agent_env_slice = slice( # TODO: remove hardcoding
            int(0 * self._num_envs / self._nbuffers),
            int((0 + 1) * self._num_envs / self._nbuffers),
        )
        demo_env_slice = slice(
            int(self._demo_buffer_idx * self._num_envs / self._nbuffers),
            int((self._demo_buffer_idx + 1) * self._num_envs / self._nbuffers),
        )
        num_homo_environments = self._num_envs / self._nbuffers
        assert num_homo_environments >= num_mini_batch, (
            "Trainer requires the number of environments ({}) "
            "to be greater than or equal to the number of "
            "trainer mini batches ({}).".format(
                num_homo_environments, num_mini_batch
            )
        )
        if num_homo_environments % num_mini_batch != 0:
            warnings.warn(
                "Number of environments ({}) is not a multiple of the"
                " number of mini batches ({}).  This results in mini batches"
                " of different sizes, which can harm training performance.".format(
                    num_homo_environments, num_mini_batch
                )
            )
        for agent_env_inds, demo_env_inds in zip(
            (agent_env_slice.start + torch.randperm(agent_env_slice.stop - agent_env_slice.start)).chunk(num_mini_batch),
            (demo_env_slice.start + torch.randperm(demo_env_slice.stop - demo_env_slice.start)).chunk(num_mini_batch)
        ):
            batch_agent = self.buffers[0: self.current_rollout_step_idx, agent_env_inds]
            batch_agent["discrim_start_hidden_states"] = batch_agent["discrim_start_hidden_states"][0:1]
            batch_demo = self.buffers[0: self.current_rollout_step_idx, demo_env_inds]
            batch_demo["discrim_start_hidden_states"] = batch_demo["discrim_start_hidden_states"][0:1]

            yield (
                batch_agent.map(lambda v: v.flatten(0, 1)),
                batch_demo.map(lambda v: v.flatten(0, 1)),
                agent_env_inds,
                demo_env_inds
            )
