#!/usr/bin/env python3

# Created by Yunhai Feng at 4:09 pm, 2022/4/30.


import contextlib
import os
import random
import time
from collections import defaultdict, deque
from typing import Any, Dict, List, Optional

import numpy as np
import torch
import tqdm
from gym import spaces
from torch import nn
from torch.optim.lr_scheduler import LambdaLR

from habitat import Config, VectorEnv, logger
from habitat.utils import profiling_wrapper
from habitat.utils.visualizations.utils import observations_to_image
from habitat_baselines.common.base_trainer import BaseRLTrainer
from habitat_baselines.common.baseline_registry import baseline_registry
from habitat_baselines.common.environments import get_env_class
from habitat_baselines.common.obs_transformers import (
    apply_obs_transforms_batch,
    apply_obs_transforms_obs_space,
    get_active_obs_transforms,
)
from gail.common.rollout_storage import RolloutStorage
from habitat_baselines.common.tensorboard_utils import TensorboardWriter
from habitat_baselines.rl.ddppo.ddp_utils import (
    EXIT,
    add_signal_handlers,
    get_distrib_size,
    init_distrib_slurm,
    is_slurm_batch_job,
    load_resume_state,
    rank0_only,
    requeue_job,
    save_resume_state,
)
from habitat_baselines.rl.ddppo.policy import (  # noqa: F401.
    PointNavResNetPolicy,
)
from gail.gail import GAIL
from habitat_baselines.rl.ppo.policy import Policy
from habitat_baselines.utils.common import (
    ObservationBatchingCache,
    action_to_velocity_control,
    batch_obs,
    generate_video,
)
from habitat_baselines.utils.env_utils import construct_envs
from gail.env import construct_gail_envs
from habitat.tasks.nav.object_nav_task import ObjectGoalSensor
from gail.discriminator import Discriminator, DiscriminatorNet

@baseline_registry.register_trainer(name="gail")
class GAILTrainer(BaseRLTrainer):
    # TODO: comment
    r"""Trainer class for PPO algorithm
    Paper: https://arxiv.org/abs/1707.06347.
    """
    supported_tasks = ["Nav-v0"]

    SHORT_ROLLOUT_THRESHOLD: float = 0.25
    _is_distributed: bool
    _obs_batching_cache: ObservationBatchingCache
    envs: VectorEnv
    agent: GAIL
    actor_critic: Policy
    discriminator: Discriminator
    discriminator_net: DiscriminatorNet

    def __init__(self, config=None):
        super().__init__(config)
        self.actor_critic = None
        self.agent = None
        self.discriminator = None
        self.envs = None
        self.obs_transforms = []

        self._static_encoder = False
        self._encoder = None
        self._obs_space = None

        self.semantic_predictor = None

        # Distributed if the world size would be
        # greater than 1
        self._is_distributed = get_distrib_size()[2] > 1
        self._obs_batching_cache = ObservationBatchingCache()

        self.using_velocity_ctrl = (
            self.config.TASK_CONFIG.TASK.POSSIBLE_ACTIONS
        ) == ["VELOCITY_CONTROL"]

    @property
    def obs_space(self):
        if self._obs_space is None and self.envs is not None:
            self._obs_space = self.envs.observation_spaces[0]
            self._obs_space[ObjectGoalSensor.cls_uuid].high[0] = 28 #TODO: add param to config

            print("=" * 40, self._obs_space) # TODO: check the output
            print(self._obs_space[ObjectGoalSensor.cls_uuid].high)
            print(self._obs_space[ObjectGoalSensor.cls_uuid].high[0])

            if self.config.MODEL.USE_PRED_SEMANTICS and "semantic" not in self._obs_space.spaces:
                sem_embedding_size = self.config.MODEL.SEMANTIC_ENCODER.embedding_size
                rgb_shape = self._obs_space.spaces["rgb"].shape
                self._obs_space["semantic"] = spaces.Box(
                    low=0,
                    high=255,
                    shape=(rgb_shape[0], rgb_shape[1]),
                    dtype=np.uint8,
                )
            print(self._obs_space)

        return self._obs_space

    @obs_space.setter
    def obs_space(self, new_obs_space):
        self._obs_space = new_obs_space

    def _all_reduce(self, t: torch.Tensor) -> torch.Tensor:
        r"""All reduce helper method that moves things to the correct
        device and only runs if distributed
        """
        if not self._is_distributed:
            return t

        orig_device = t.device
        t = t.to(device=self.device)
        torch.distributed.all_reduce(t)

        return t.to(device=orig_device)

    def _setup_actor_critic_agent(self, ppo_cfg: Config) -> None:
        r"""Sets up actor critic and agent for PPO.

        Args:
            ppo_cfg: config node with relevant params

        Returns:
            None
        """
        logger.add_filehandler(self.config.LOG_FILE)

        policy = baseline_registry.get_policy(self.config.RL.POLICY.name)
        observation_space = self.obs_space
        self.obs_transforms = get_active_obs_transforms(self.config)
        observation_space = apply_obs_transforms_obs_space(
            observation_space, self.obs_transforms
        )

        self.actor_critic = policy.from_config(
            self.config, observation_space, self.policy_action_space
        )
        self.obs_space = observation_space
        self.actor_critic.to(self.device)

        self.finetune = False
        if (
            self.config.RL.DDPPO.pretrained_encoder
            or self.config.RL.DDPPO.pretrained
        ):
            pretrained_state = torch.load(
                self.config.RL.DDPPO.pretrained_weights, map_location="cpu"
            )
            logger.info(f"Pretrained Weights Loaded from {self.config.RL.DDPPO.pretrained_weights}")

            if self.config.RL.DDPPO.pretrained_encoder:
                # only load encoders
                logger.info("Load weights for encoders only!!!!!!!")
                for encoder_name in ["rgb_encoder", "depth_encoder", "sem_seg_encoder"]:
                    if encoder_name == "rgb_encoder":
                        encoder = self.actor_critic.net.rgb_encoder
                    elif encoder_name == "depth_encoder":
                        encoder = self.actor_critic.net.depth_encoder
                    elif encoder_name == "sem_seg_encoder":
                        encoder = self.actor_critic.net.sem_seg_encoder
                    else:
                         raise ValueError
                    missing_keys = encoder.load_state_dict(
                        {
                            k.replace(f"model.net.{encoder_name}.", ""): v
                            for k, v in pretrained_state["state_dict"].items()
                        }, strict=False
                    )
                    logger.info(f"{encoder_name} missing keys: {missing_keys}")
                if self.config.RL.DDPPO.freeze_encoders:
                    self.actor_critic.freeze_visual_encoders()
                    logger.info("Using frozen pretraiend encoders")

            else:
                if "agent_state_dict" in pretrained_state.keys():
                    print("@@@@@@ agent_state_dict")
                    state_dict = {
                        k.replace("actor_critic.", ""): v
                        for k, v in pretrained_state["agent_state_dict"].items()
                    }
                elif "state_dict" in pretrained_state.keys():
                    print("@@@@@@ state_dict")
                    state_dict = {
                        k.replace("actor_critic.", ""): v
                        for k, v in pretrained_state["state_dict"].items()
                    }
                else:
                    assert 0
                missing_keys = self.actor_critic.load_state_dict(
                    {
                        k.replace("model.", ""): v
                        for k, v in state_dict.items()
                    }, strict=False
                )
                logger.info("Loading checkpoint missing keys: {}".format(missing_keys))

                if self.config.RL.DDPPO.freeze_encoders:
                    self.actor_critic.freeze_visual_encoders()
                    logger.info("Using frozen pretraiend encoders")

                if hasattr(self.config.RL, "Finetune"):
                    logger.info("Start Freeze encoder")
                    self.finetune = True
                    self.actor_finetuning_update = self.config.RL.Finetune.start_actor_finetuning_at
                    self.actor_lr_warmup_update = self.config.RL.Finetune.actor_lr_warmup_update
                    self.critic_lr_decay_update = self.config.RL.Finetune.critic_lr_decay_update
                    self.start_critic_warmup_at = self.config.RL.Finetune.start_critic_warmup_at

        # if not self.config.RL.DDPPO.train_encoder:
        #     self._static_encoder = True
        #     for param in self.actor_critic.net.visual_encoder.parameters():
        #         param.requires_grad_(False)

        if self.config.RL.DDPPO.reset_critic:
            nn.init.orthogonal_(self.actor_critic.critic.fc.weight)
            nn.init.constant_(self.actor_critic.critic.fc.bias, 0)
        elif self.config.RL.DDPPO.use_pretrained_critic_weights:
            ckpt = torch.load(
                self.config.RL.DDPPO.pretrained_critic_weights, map_location="cpu"
            )
            logger.info("Pretrained weights loaded for critic")
            missing_keys = self.actor_critic.critic.load_state_dict(
                {
                    k.replace("actor_critic.critic.", ""): v
                    for k, v in ckpt["agent_state_dict"].items()
                    if k.startswith("actor_critic.critic")
                }, strict=False
            )
            logger.info("missing keys: {}".format(missing_keys))
            del ckpt

        self.agent = GAIL(
            actor_critic=self.actor_critic,
            clip_param=ppo_cfg.clip_param,
            ppo_epoch=ppo_cfg.ppo_epoch,
            num_mini_batch=ppo_cfg.num_mini_batch,
            value_loss_coef=ppo_cfg.value_loss_coef,
            entropy_coef=ppo_cfg.entropy_coef,
            lr=ppo_cfg.lr,
            eps=ppo_cfg.eps,
            max_grad_norm=ppo_cfg.max_grad_norm,
            use_normalized_advantage=ppo_cfg.use_normalized_advantage,
            finetune=self.finetune
        )

    def _setup_discriminator(self):

        discrim_obs_transforms = []
        discrim_obs_transform_names = (
            self.config.GAIL.DISCRIMINATOR.OBS_TRANSFORMS.ENABLED_TRANSFORMS
        )
        for obs_transform_name in discrim_obs_transform_names:
            print("T", obs_transform_name)
            obs_trans_cls = baseline_registry.get_obs_transformer(
                obs_transform_name
            )
            obs_transform = obs_trans_cls.from_config(self.config)
            discrim_obs_transforms.append(obs_transform)

        discriminator_observation_space = spaces.Dict({k: v for k, v in self._obs_space.items() if k != "semantic"})
        print("discriminator_observation_space", discriminator_observation_space)
        discriminator_observation_space = apply_obs_transforms_obs_space(
            discriminator_observation_space, discrim_obs_transforms
        )

        self.discriminator_net = DiscriminatorNet(
            observation_space=discriminator_observation_space,
            action_space=self.policy_action_space,
            hidden_size=self.config.GAIL.DISCRIMINATOR.hidden_size,
            backbone=self.config.GAIL.DISCRIMINATOR.backbone,
            resnet_baseplanes=self.config.GAIL.DISCRIMINATOR.resnet_baseplanes,
            normalize_visual_inputs=self.config.GAIL.DISCRIMINATOR.normalize_visual_inputs,
            force_blind_policy=False,
            discrete_actions=True,
            use_rnn=self.config.GAIL.DISCRIMINATOR.use_rnn,
            num_recurrent_layers=self.config.GAIL.DISCRIMINATOR.num_recurrent_layers,
            rnn_type=self.config.GAIL.DISCRIMINATOR.rnn_type,
            obs_transforms=discrim_obs_transforms
        )

        # Load pretrained discriminator weights
        if hasattr(self.config.GAIL.DISCRIMINATOR, "pretrained") and self.config.GAIL.DISCRIMINATOR.pretrained:
            ckpt = torch.load(self.config.GAIL.DISCRIMINATOR.pretrained_weights, map_location="cpu")
            logger.info(f"Discriminator pretrained weights loaded from {self.config.GAIL.DISCRIMINATOR.pretrained_weights}")
            missing_keys = self.discriminator_net.load_state_dict(
                {
                    k.replace("net.", ""): v
                    for k, v in ckpt["discrim_state_dict"].items()
                }, strict=False
            )
            logger.info("Loading checkpoint missing keys: {}".format(missing_keys))
            del ckpt

        self.discriminator_net.to(self.device)
        self.discriminator = Discriminator(
            net=self.discriminator_net,
            discriminator_epoch=self.config.GAIL.DISCRIMINATOR.discriminator_epoch,
            num_mini_batch=self.config.GAIL.DISCRIMINATOR.num_mini_batch,
            max_grad_norm=self.config.GAIL.DISCRIMINATOR.max_grad_norm
        )

    def _init_envs(self, config=None):
        if config is None:
            config = self.config

        self.envs = construct_gail_envs(
            config,
            get_env_class(config.ENV_NAME),
            workers_ignore_signals=is_slurm_batch_job(),
        )

    def _init_eval_envs(self, config=None):
        if config is None:
            config = self.config
        self.envs = construct_envs(
            config,
            get_env_class(config.ENV_NAME),
            workers_ignore_signals=is_slurm_batch_job(),
        )

    def _init_train(self):
        resume_state = load_resume_state(self.config)
        if resume_state is not None:
            self.config: Config = resume_state["config"]
            self.config.defrost()
            self.config.GAIL.is_demonstration_env = False
            self.config.freeze()
            self.using_velocity_ctrl = (
                self.config.TASK_CONFIG.TASK.POSSIBLE_ACTIONS
            ) == ["VELOCITY_CONTROL"]

        if self.config.RL.DDPPO.force_distributed:
            self._is_distributed = True

        if is_slurm_batch_job():
            add_signal_handlers()

        if self._is_distributed:
            local_rank, tcp_store = init_distrib_slurm(
                self.config.RL.DDPPO.distrib_backend
            )
            if rank0_only():
                logger.info(
                    "Initialized DD-PPO with {} workers".format(
                        torch.distributed.get_world_size()
                    )
                )

            self.config.defrost()
            self.config.TORCH_GPU_ID = local_rank
            self.config.SIMULATOR_GPU_ID = local_rank
            # Multiply by the number of simulators to make sure they also get unique seeds
            self.config.TASK_CONFIG.SEED += (
                torch.distributed.get_rank() * self.config.NUM_ENVIRONMENTS
            )
            self.config.freeze()

            random.seed(self.config.TASK_CONFIG.SEED)
            np.random.seed(self.config.TASK_CONFIG.SEED)
            torch.manual_seed(self.config.TASK_CONFIG.SEED)
            self.num_rollouts_done_store = torch.distributed.PrefixStore(
                "rollout_tracker", tcp_store
            )
            self.num_rollouts_done_store.set("num_done", "0")

        if rank0_only() and self.config.VERBOSE:
            logger.info(">" * 20 + " config " + ">" * 20 + "\n" +
                        f"{self.config}" + "\n" +
                        "<" * 20 + " end of config " + "<" * 20)

        profiling_wrapper.configure(
            capture_start_step=self.config.PROFILING.CAPTURE_START_STEP,
            num_steps_to_capture=self.config.PROFILING.NUM_STEPS_TO_CAPTURE,
        )

        self._init_envs()

        if self.using_velocity_ctrl:
            self.policy_action_space = self.envs.action_spaces[0][
                "VELOCITY_CONTROL"
            ]
            action_shape = (2,)
            discrete_actions = False
        else:
            self.policy_action_space = self.envs.action_spaces[0]
            action_shape = None
            discrete_actions = True

        ppo_cfg = self.config.RL.PPO
        if torch.cuda.is_available():
            self.device = torch.device("cuda", self.config.TORCH_GPU_ID)
            torch.cuda.set_device(self.device)
        else:
            self.device = torch.device("cpu")

        if rank0_only() and not os.path.isdir(self.config.CHECKPOINT_FOLDER):
            os.makedirs(self.config.CHECKPOINT_FOLDER)

        # Load RedNet for semantics prediction
        if self.config.MODEL.USE_PRED_SEMANTICS:
            from gail.models.rednet import load_rednet
            self.semantic_predictor = load_rednet(
                self.device,
                ckpt=self.config.MODEL.SEMANTIC_ENCODER.rednet_ckpt,
                resize=True, # since we train on half-vision
                num_classes=self.config.MODEL.SEMANTIC_ENCODER.num_classes
            )
            self.semantic_predictor.eval()

        self._setup_actor_critic_agent(ppo_cfg)
        self._setup_discriminator()
        if self._is_distributed:
            self.agent.init_distributed(find_unused_params=True)  # type: ignore
            # TODO: discriminator.init_distributed()

        logger.info(
            "agent number of parameters: {}".format(
                sum(param.numel() for param in self.agent.parameters())
            )
        )

        obs_space = self.obs_space
        if self._static_encoder:
            self._encoder = self.actor_critic.net.visual_encoder
            obs_space = spaces.Dict(
                {
                    "visual_features": spaces.Box(
                        low=np.finfo(np.float32).min,
                        high=np.finfo(np.float32).max,
                        shape=self._encoder.output_shape,
                        dtype=np.float32,
                    ),
                    **obs_space.spaces,
                }
            )

        self._nbuffers = 2 if self.config.GAIL.use_double_buffer else 1
        self._agent_buffer_idx = 0
        self._demo_buffer_idx = 1 # TODO: replace hardcoding by config
        self._agent_env_slice = slice(
            int(self._agent_buffer_idx * self.envs.num_envs / self._nbuffers),
            int((self._agent_buffer_idx + 1) * self.envs.num_envs / self._nbuffers),
        )
        self._demo_env_slice = slice(
            int(self._demo_buffer_idx * self.envs.num_envs / self._nbuffers),
            int((self._demo_buffer_idx + 1) * self.envs.num_envs / self._nbuffers),
        )

        self.rollouts = RolloutStorage(
            ppo_cfg.num_steps,
            self.envs.num_envs,
            obs_space,
            self.policy_action_space,
            self.config.MODEL.STATE_ENCODER.hidden_size,# ppo_cfg.hidden_size,
            num_recurrent_layers=self.actor_critic.net.num_recurrent_layers,
            discrim_recurrent_hidden_state_size=self.config.GAIL.DISCRIMINATOR.hidden_size,
            discrim_num_recurrent_layers=self.discriminator.net.num_recurrent_layers,
            is_double_buffered=self.config.GAIL.use_double_buffer,
            demo_buffer_idx=self._demo_buffer_idx,
            action_shape=action_shape,
            discrete_actions=discrete_actions,
        )
        self.rollouts.to(self.device)

        observations = self.envs.reset()
        batch = batch_obs(
            observations, device=self.device, cache=self._obs_batching_cache
        )
        batch = apply_obs_transforms_batch(batch, self.obs_transforms)  # type: ignore

        if self._static_encoder:
            with torch.no_grad():
                batch["visual_features"] = self._encoder(batch)

        if self.config.MODEL.USE_PRED_SEMANTICS:
            with torch.no_grad():
                # print("*** keys ****", batch.keys())
                batch["semantic"] = self.semantic_predictor(
                    batch["rgb"],
                    batch["depth"]
                ) - 1
                print(">>> batch[semantic].shape", batch["semantic"].shape)
                print(">>>", self.rollouts.buffers["observations"]["semantic"].shape)

        self.rollouts.buffers["observations"][0] = batch  # type: ignore

        # self.current_episode_reward = torch.zeros(self.envs.num_envs, 1)
        self.current_episode_stats = dict(
            task_reward=torch.zeros(self.envs.num_envs, 1),
            gail_reward=torch.zeros(self.envs.num_envs, 1),
            total_reward=torch.zeros(self.envs.num_envs, 1)
        )
        self.running_episode_stats = dict(
            count=torch.zeros(self.envs.num_envs, 1),
            task_reward=torch.zeros(self.envs.num_envs, 1),
            gail_reward=torch.zeros(self.envs.num_envs, 1),
            total_reward=torch.zeros(self.envs.num_envs, 1)
        )
        self.window_episode_stats = defaultdict(
            lambda: deque(maxlen=ppo_cfg.reward_window_size)
        )

        self.env_time = 0.0
        self.pth_time = 0.0
        self.t_start = time.time()

    @rank0_only
    @profiling_wrapper.RangeContext("save_checkpoint")
    def save_checkpoint(
        self, file_name: str, extra_state: Optional[Dict] = None
    ) -> None:
        r"""Save checkpoint with specified name.

        Args:
            file_name: file name for checkpoint

        Returns:
            None
        """
        checkpoint = {
            "agent_state_dict": self.agent.state_dict(),
            "discrim_state_dict": self.discriminator.state_dict(),
            "config": self.config,
        }
        if extra_state is not None:
            checkpoint["extra_state"] = extra_state

        torch.save(
            checkpoint, os.path.join(self.config.CHECKPOINT_FOLDER, file_name)
        )

    def load_checkpoint(self, checkpoint_path: str, *args, **kwargs) -> Dict:
        r"""Load checkpoint of specified path as a dict.

        Args:
            checkpoint_path: path of target checkpoint
            *args: additional positional args
            **kwargs: additional keyword args

        Returns:
            dict containing checkpoint info
        """
        return torch.load(checkpoint_path, *args, **kwargs)

    # METRICS_BLACKLIST = {"top_down_map", "collisions.is_collision"}
    INFO_BLACKLIST = {"demo_action", "top_down_map", "collisions.is_collision"}

    @classmethod
    def _extract_scalars_from_info(
        cls, info: Dict[str, Any]
    ) -> Dict[str, float]:
        result = {}
        for k, v in info.items():
            if k in cls.INFO_BLACKLIST:
                continue

            if isinstance(v, dict):
                result.update(
                    {
                        k + "." + subk: subv
                        for subk, subv in cls._extract_scalars_from_info(
                            v
                        ).items()
                        if (k + "." + subk) not in cls.INFO_BLACKLIST
                    }
                )
            # Things that are scalar-like will have an np.size of 1.
            # Strings also have an np.size of 1, so explicitly ban those
            elif np.size(v) == 1 and not isinstance(v, str):
                result[k] = float(v)

        return result

    @classmethod
    def _extract_scalars_from_infos(
        cls, infos: List[Dict[str, Any]]
    ) -> Dict[str, List[float]]:

        results = defaultdict(list)
        for i in range(len(infos)):
            for k, v in cls._extract_scalars_from_info(infos[i]).items():
                results[k].append(v)
        return results

    def _compute_actions_and_step_envs(self, buffer_index: int = 0):
        num_envs = self.envs.num_envs
        env_slice = slice(
            int(buffer_index * num_envs / self._nbuffers),
            int((buffer_index + 1) * num_envs / self._nbuffers),
        )
        t_sample_action = time.time()

        # sample actions
        if buffer_index == self._demo_buffer_idx:
            # dummy actions, will be overwritten in env
            actions = torch.zeros(int(num_envs / self._nbuffers), 1,
                                  dtype=torch.long,
                                  device=torch.device("cpu"))
        else:
            with torch.no_grad():
                step_batch = self.rollouts.buffers[
                    self.rollouts.current_rollout_step_idxs[buffer_index],
                    env_slice,
                ]

                profiling_wrapper.range_push("compute actions")
                (
                    values,
                    actions,
                    actions_log_probs,
                    recurrent_hidden_states,
                ) = self.actor_critic.act(
                    step_batch["observations"],
                    step_batch["recurrent_hidden_states"],
                    step_batch["prev_actions"],
                    step_batch["masks"],
                )

            # NB: Move actions to CPU.  If CUDA tensors are
            # sent in to env.step(), that will create CUDA contexts
            # in the subprocesses.
            # For backwards compatibility, we also call .item() to convert to
            # an int
            actions = actions.to(device="cpu")

        self.pth_time += time.time() - t_sample_action

        profiling_wrapper.range_pop()  # compute actions

        t_step_env = time.time()

        for index_env, act in zip(
            range(env_slice.start, env_slice.stop), actions.unbind(0)
        ):
            if self.using_velocity_ctrl:
                step_action = action_to_velocity_control(act)
            else:
                step_action = act.item()
            self.envs.async_step_at(index_env, step_action)

        self.env_time += time.time() - t_step_env

        if buffer_index != self._demo_buffer_idx:
            self.rollouts.insert(
                next_recurrent_hidden_states=recurrent_hidden_states,
                actions=actions,
                action_log_probs=actions_log_probs,
                value_preds=values,
                buffer_index=buffer_index,
            )

    def _collect_environment_result(self, buffer_index: int = 0):
        device = self.current_episode_stats["total_reward"].device  # device: cpu
        num_envs = self.envs.num_envs
        env_slice = slice(
            int(buffer_index * num_envs / self._nbuffers),
            int((buffer_index + 1) * num_envs / self._nbuffers),
        )

        t_step_env = time.time()
        outputs = [
            self.envs.wait_step_at(index_env)
            for index_env in range(env_slice.start, env_slice.stop)
        ]

        observations, task_rewards_l, dones, infos = [
            list(x) for x in zip(*outputs)
        ]

        self.env_time += time.time() - t_step_env

        t_update_stats = time.time()
        batch = batch_obs(
            observations, device=self.device, cache=self._obs_batching_cache
        )
        batch = apply_obs_transforms_batch(batch, self.obs_transforms)  # type: ignore

        task_rewards = torch.tensor(
            task_rewards_l,
            dtype=torch.float,
            device=device,
        )
        task_rewards = task_rewards.unsqueeze(1)

        # if buffer_index == self._agent_buffer_idx:
        #     # TODO: Compute GAIL rewards
        #     with torch.no_grad():
        #         step_batch = self.rollouts.buffers[
        #             self.rollouts.current_rollout_step_idxs[buffer_index],
        #             env_slice,
        #         ]
        #         gail_rewards = self.discriminator.get_reward(
        #             observations=step_batch["observations"],
        #             prev_actions=step_batch["prev_actions"],
        #             masks=step_batch["masks"],
        #             curr_actions=step_batch["actions"]
        #         ).to(device)
        #
        #         total_rewards = self.config.GAIL.gail_reward_coef * gail_rewards \
        #                       + self.config.GAIL.task_reward_coef * task_rewards
        # else:
        #     gail_rewards = None
        #     total_rewards = None

        not_done_masks = torch.tensor(
            [[not done] for done in dones],
            dtype=torch.bool,
            device=device,
        )
        done_masks = torch.logical_not(not_done_masks)

        # TODO: read demo actions and store them into rollout
        if buffer_index == self._demo_buffer_idx:
            demo_actions = torch.tensor(
                [[info["demo_action"]] for info in infos],
                device=device, # TODO: check this
            )

        self.running_episode_stats["count"][env_slice] += done_masks.float()  # type: ignore

        self.current_episode_stats["task_reward"][env_slice] += task_rewards
        current_ep_task_reward = self.current_episode_stats["task_reward"][env_slice]
        self.running_episode_stats["task_reward"][env_slice] += current_ep_task_reward.where(done_masks, current_ep_task_reward.new_zeros(()))  # type: ignore

        # if buffer_index == self._agent_buffer_idx:
        #     self.current_episode_stats["gail_reward"][env_slice] += gail_rewards # TODO: check gail_rewards None
        #     self.current_episode_stats["total_reward"][env_slice] += total_rewards # TODO: compute total rewards
        #     current_ep_gail_reward = self.current_episode_stats["gail_reward"][env_slice]
        #     current_ep_total_reward = self.current_episode_stats["total_reward"][env_slice]
        #     self.running_episode_stats["gail_reward"][env_slice] += current_ep_gail_reward.where(done_masks, current_ep_gail_reward.new_zeros(()))  # type: ignore
        #     self.running_episode_stats["total_reward"][env_slice] += current_ep_total_reward.where(done_masks, current_ep_total_reward.new_zeros(()))  # type: ignore

        for k, v_k in self._extract_scalars_from_infos(infos).items():
            v = torch.tensor(
                v_k,
                dtype=torch.float,
                device=device,
            ).unsqueeze(1)
            if k not in self.running_episode_stats:
                self.running_episode_stats[k] = torch.zeros_like(
                    self.running_episode_stats["count"]
                )

            self.running_episode_stats[k][env_slice] += v.where(done_masks, v.new_zeros(()))  # type: ignore

        self.current_episode_stats["task_reward"][env_slice].masked_fill_(done_masks, 0.0)
        # if buffer_index == self._agent_buffer_idx:
        #     self.current_episode_stats["gail_reward"][env_slice].masked_fill_(done_masks, 0.0)
        #     self.current_episode_stats["total_reward"][env_slice].masked_fill_(done_masks, 0.0)

        if self._static_encoder:
            with torch.no_grad():
                batch["visual_features"] = self._encoder(batch)

        if self.config.MODEL.USE_PRED_SEMANTICS:
            with torch.no_grad():
                batch["semantic"] = self.semantic_predictor(
                    batch["rgb"],
                    batch["depth"]
                ) - 1

        self.rollouts.insert(
            actions=demo_actions if buffer_index == self._demo_buffer_idx else None, # TODO: check this
            next_observations=batch,
            task_rewards=task_rewards,
            gail_rewards=None, # gail_rewards,
            total_rewards=None, # total_rewards,
            next_masks=not_done_masks,
            buffer_index=buffer_index,
        )

        self.rollouts.advance_rollout(buffer_index)

        self.pth_time += time.time() - t_update_stats

        return env_slice.stop - env_slice.start

    @profiling_wrapper.RangeContext("_collect_rollout_step")
    def _collect_rollout_step(self):
        self._compute_actions_and_step_envs()
        return self._collect_environment_result()

    def _compute_gail_rewards_for_rollouts(self):
        with torch.no_grad():
            batch = self.rollouts.buffers[
                0: self.rollouts.current_rollout_step_idxs[self._agent_buffer_idx],
                self._agent_env_slice,
            ]
            n_steps, n_envs, _ = batch["masks"].shape
            # print("n_steps, n_envs:",n_steps, n_envs)
            batch = batch.map(lambda v: v.flatten(0, 1))
            gail_rewards = self.discriminator.get_reward(
                observations=batch["observations"],
                prev_actions=batch["prev_actions"],
                masks=batch["masks"],
                curr_actions=batch["actions"],
                rnn_hidden_states=batch["discrim_start_hidden_states"][0:1]
            ).view(n_steps, n_envs, -1).to(self.rollouts.buffers["gail_rewards"].device)

            self.rollouts.buffers["gail_rewards"][0: n_steps, self._agent_env_slice] = gail_rewards

            total_rewards = self.config.GAIL.gail_reward_coef * gail_rewards \
                          + self.config.GAIL.task_reward_coef \
                          * self.rollouts.buffers["task_rewards"][0: n_steps, self._agent_env_slice]
            self.rollouts.buffers["total_rewards"][0: n_steps, self._agent_env_slice] = total_rewards

            stats_device = self.current_episode_stats["gail_reward"].device
            gail_rewards = gail_rewards.to(stats_device)
            total_rewards = total_rewards.to(stats_device)
            done_masks = torch.logical_not(batch["masks"][:, self._agent_env_slice]).to(stats_device)
            for i in range(n_steps):
                # print(self.current_episode_stats["gail_reward"][self._agent_env_slice].shape, gail_rewards[i].shape)
                # print(device, self.current_episode_stats["gail_reward"][self._agent_env_slice].device, gail_rewards[i].device)
                self.current_episode_stats["gail_reward"][self._agent_env_slice] += gail_rewards[i]
                self.current_episode_stats["total_reward"][self._agent_env_slice] += total_rewards[i]
                current_ep_gail_reward = self.current_episode_stats["gail_reward"][self._agent_env_slice]
                current_ep_total_reward = self.current_episode_stats["total_reward"][self._agent_env_slice]
                self.running_episode_stats["gail_reward"][self._agent_env_slice] += current_ep_gail_reward.where(
                    done_masks[i], current_ep_gail_reward.new_zeros(())
                )  # type: ignore
                self.running_episode_stats["total_reward"][self._agent_env_slice] += current_ep_total_reward.where(
                    done_masks[i], current_ep_total_reward.new_zeros(()))  # type: ignore
                self.current_episode_stats["gail_reward"][self._agent_env_slice].masked_fill_(done_masks[i], 0.0)
                self.current_episode_stats["total_reward"][self._agent_env_slice].masked_fill_(done_masks[i], 0.0)

    def _update_discriminator(self):
        return self.discriminator.update(self.rollouts)

    @profiling_wrapper.RangeContext("_update_agent")
    def _update_agent(self):
        ppo_cfg = self.config.RL.PPO
        t_update_model = time.time()

        # First compute gail rewards given by the discriminator
        self._compute_gail_rewards_for_rollouts()

        with torch.no_grad():
            step_batch = self.rollouts.buffers[
                self.rollouts.current_rollout_step_idx
            ]

            # if self.config.MODEL.USE_PRED_SEMANTICS:
            #     step_batch["observations"][
            #         "semantic"] = self.semantic_predictor(
            #         step_batch["observations"]["rgb"],
            #         step_batch["observations"]["depth"]
            #     )

            next_value = self.actor_critic.get_value(
                step_batch["observations"],
                step_batch["recurrent_hidden_states"],
                step_batch["prev_actions"],
                step_batch["masks"],
            )

        self.rollouts.compute_returns(
            next_value, ppo_cfg.use_gae, ppo_cfg.gamma, ppo_cfg.tau
        )

        if not (hasattr(self.config.GAIL,
                        "tune_discriminator") and self.config.GAIL.tune_discriminator):
            self.agent.train()

            value_loss, action_loss, dist_entropy = self.agent.update(
                self.rollouts
            )
        else:
            value_loss, action_loss, dist_entropy = 0., 0., 0.

        self.rollouts.after_update()
        self.pth_time += time.time() - t_update_model

        return (
            value_loss,
            action_loss,
            dist_entropy,
        )

    def _coalesce_post_step(
        self, losses: Dict[str, float], count_steps_delta: int
    ) -> Dict[str, float]:
        stats_ordering = sorted(self.running_episode_stats.keys())
        stats = torch.stack(
            [self.running_episode_stats[k] for k in stats_ordering], 0
        )

        stats = self._all_reduce(stats)

        for i, k in enumerate(stats_ordering):
            self.window_episode_stats[k].append(stats[i])

        if self._is_distributed:
            loss_name_ordering = sorted(losses.keys())
            stats = torch.tensor(
                [losses[k] for k in loss_name_ordering] + [count_steps_delta],
                device="cpu",
                dtype=torch.float32,
            )
            stats = self._all_reduce(stats)
            count_steps_delta = int(stats[-1].item())
            stats /= torch.distributed.get_world_size()

            losses = {
                k: stats[i].item() for i, k in enumerate(loss_name_ordering)
            }

        if self._is_distributed and rank0_only():
            self.num_rollouts_done_store.set("num_done", "0")

        self.num_steps_done += count_steps_delta

        return losses

    @rank0_only
    def _training_log(
        self, writer, losses: Dict[str, float], prev_time: int = 0
    ):
        # deltas = {
        #     k: (
        #         (v[-1] - v[0]).sum().item()
        #         if len(v) > 1
        #         else v[0].sum().item()
        #     )
        #     for k, v in self.window_episode_stats.items()
        # }

        # Yunhai's Notes: use `self._agent_env_slice` to select
        # the envs where the agent is trained online, so that
        # the envs used for collecting demonstrations are filtered
        # out
        deltas = {
            k: (
                (v[-1][self._agent_env_slice] - v[0][self._agent_env_slice]).sum().item()
                if len(v) > 1
                else v[0][self._agent_env_slice].sum().item()
            )
            for k, v in self.window_episode_stats.items()
        }
        deltas["count"] = max(deltas["count"], 1.0)

        for r in ["task_reward", "gail_reward", "total_reward"]:
            writer.add_scalar(
                f"reward/{r}",
                deltas[r] / deltas["count"],
                self.num_steps_done,
            )

        # Check to see if there are any metrics
        # that haven't been logged yet
        metrics = {
            k: v / deltas["count"]
            for k, v in deltas.items()
            if k not in {"task_reward", "gail_reward", "total_reward", "count"}
        }

        for k, v in metrics.items():
            writer.add_scalar(f"metrics/{k}", v, self.num_steps_done)
        for k, v in losses.items():
            writer.add_scalar(f"losses/{k}", v, self.num_steps_done)

        # also log demo metrics
        demo_deltas = {
            k: (
                (v[-1][self._demo_env_slice] - v[0][self._demo_env_slice]).sum().item()
                if len(v) > 1
                else v[0][self._demo_env_slice].sum().item()
            )
            for k, v in self.window_episode_stats.items()
        }
        demo_deltas["count"] = max(demo_deltas["count"], 1.0)
        demo_metrics = {
            k: v / demo_deltas["count"]
            for k, v in demo_deltas.items()
            if k not in {"task_reward", "gail_reward", "total_reward", "count"}
        }
        for k, v in demo_metrics.items():
            writer.add_scalar(f"demo_metrics/{k}", v, self.num_steps_done)
        # end of demo metrics logging


        # log stats
        if self.num_updates_done % self.config.LOG_INTERVAL == 0:
            logger.info(
                "update: {}\tfps: {:.3f}\t".format(
                    self.num_updates_done,
                    self.num_steps_done
                    / ((time.time() - self.t_start) + prev_time),
                )
            )

            logger.info(
                "update: {}\tenv-time: {:.3f}s\tpth-time: {:.3f}s\t"
                "frames: {}".format(
                    self.num_updates_done,
                    self.env_time,
                    self.pth_time,
                    self.num_steps_done,
                )
            )

            logger.info(
                "Average window size: {}  {}".format(
                    len(self.window_episode_stats["count"]),
                    "  ".join(
                        "{}: {:.3f}".format(k, v / deltas["count"])
                        for k, v in deltas.items()
                        if k != "count"
                    ),
                )
            )

    def should_end_early(self, rollout_step) -> bool:
        if not self._is_distributed:
            return False
        # This is where the preemption of workers happens.  If a
        # worker detects it will be a straggler, it preempts itself!
        return (
            rollout_step
            >= self.config.RL.PPO.num_steps * self.SHORT_ROLLOUT_THRESHOLD
        ) and int(self.num_rollouts_done_store.get("num_done")) >= (
            self.config.RL.DDPPO.sync_frac * torch.distributed.get_world_size()
        )

    @profiling_wrapper.RangeContext("train")
    def train(self) -> None:
        r"""Main method for training GAIL.

        Returns:
            None
        """

        self._init_train()

        count_checkpoints = 0
        prev_time = 0

        if not self.finetune:
            agent_lr_scheduler = LambdaLR(
                optimizer=self.agent.optimizer,
                lr_lambda=lambda x: 1 - self.percent_done(),
            )
        else:
            agent_lr_scheduler = LambdaLR(
                optimizer=self.agent.optimizer,
                lr_lambda=[
                    # actor_critic.critic
                    lambda x: self.critic_linear_decay(
                        epoch=x,
                        start_update=self.start_critic_warmup_at,
                        max_updates=self.critic_lr_decay_update,
                        start_lr=self.config.RL.PPO.lr,
                        end_lr=self.config.RL.Finetune.policy_ft_lr
                    ),
                    # actor_critic.net.state_encoder
                    lambda x: self.linear_warmup(
                        epoch=x,
                        start_update=self.actor_finetuning_update,
                        max_updates=self.actor_lr_warmup_update,
                        start_lr=0.0,
                        end_lr=self.config.RL.Finetune.policy_ft_lr
                    ),
                    # actor_critic.action_distribution
                    lambda x: self.linear_warmup(
                        epoch=x,
                        start_update=self.actor_finetuning_update,
                        max_updates=self.actor_lr_warmup_update,
                        start_lr=0.0,
                        end_lr=self.config.RL.Finetune.policy_ft_lr
                    ),
                ]
            )

        discrim_lr_scheduler = LambdaLR(
            optimizer=self.discriminator.optimizer,
            lr_lambda=lambda x: 1 - self.percent_done(),
        )

        resume_state = load_resume_state(self.config)
        if resume_state is not None:
            self.agent.load_state_dict(resume_state["agent_state_dict"])
            self.agent.optimizer.load_state_dict(resume_state["agent_optim_state"])
            agent_lr_scheduler.load_state_dict(resume_state["agent_lr_sched_state"])

            self.discriminator.load_state_dict(resume_state["discrim_state_dict"])
            self.discriminator.optimizer.load_state_dict(resume_state["discrim_optim_state"])
            discrim_lr_scheduler.load_state_dict(resume_state["discrim_lr_sched_state"])

            requeue_stats = resume_state["requeue_stats"]
            self.env_time = requeue_stats["env_time"]
            self.pth_time = requeue_stats["pth_time"]
            self.num_steps_done = requeue_stats["num_steps_done"]
            self.num_updates_done = requeue_stats["num_updates_done"]
            self._last_checkpoint_percent = requeue_stats[
                "_last_checkpoint_percent"
            ]
            count_checkpoints = requeue_stats["count_checkpoints"]
            prev_time = requeue_stats["prev_time"]

            self.running_episode_stats = requeue_stats["running_episode_stats"]
            self.window_episode_stats.update(
                requeue_stats["window_episode_stats"]
            )

        ppo_cfg = self.config.RL.PPO

        with (
            TensorboardWriter(  # type: ignore
                self.config.TENSORBOARD_DIR, flush_secs=self.flush_secs
            )
            if rank0_only()
            else contextlib.suppress()
        ) as writer:
            while not self.is_done():
                profiling_wrapper.on_start_step()
                profiling_wrapper.range_push("train update")

                if ppo_cfg.use_linear_clip_decay:
                    self.agent.clip_param = ppo_cfg.clip_param * (
                        1 - self.percent_done()
                    )

                if rank0_only() and self._should_save_resume_state():
                    requeue_stats = dict(
                        env_time=self.env_time,
                        pth_time=self.pth_time,
                        count_checkpoints=count_checkpoints,
                        num_steps_done=self.num_steps_done,
                        num_updates_done=self.num_updates_done,
                        _last_checkpoint_percent=self._last_checkpoint_percent,
                        prev_time=(time.time() - self.t_start) + prev_time,
                        running_episode_stats=self.running_episode_stats,
                        window_episode_stats=dict(self.window_episode_stats),
                    )

                    save_resume_state(
                        dict(
                            agent_state_dict=self.agent.state_dict(),
                            agent_optim_state=self.agent.optimizer.state_dict(),
                            agent_lr_sched_state=agent_lr_scheduler.state_dict(),
                            discrim_state_dict=self.discriminator.state_dict(),
                            discrim_optim_state=self.discriminator.optimizer.state_dict(),
                            discrim_lr_sched_state=discrim_lr_scheduler.state_dict(),
                            config=self.config,
                            requeue_stats=requeue_stats,
                        ),
                        self.config,
                    )

                if EXIT.is_set():
                    profiling_wrapper.range_pop()  # train update

                    self.envs.close()

                    requeue_job()

                    return

                if self.num_updates_done % 10 == 0:
                    logger.info(
                        "update: {}\tLR: {}\tPG_LR: {}".format(
                            self.num_updates_done,
                            agent_lr_scheduler.get_lr(),
                            [param_group["lr"] for param_group in
                             self.agent.optimizer.param_groups],
                        )
                    )

                if self.finetune:
                    # for param in self.actor_critic.action_distribution.parameters():
                        # print(param.requires_grad)
                    # Enable actor finetuning at update actor_finetuning_update
                    if self.num_updates_done == self.actor_finetuning_update:
                        for param in self.actor_critic.action_distribution.parameters():
                            param.requires_grad_(True)
                        for param in self.actor_critic.net.state_encoder.parameters():
                            param.requires_grad_(True)
                        for i, param_group in enumerate(self.agent.optimizer.param_groups):
                            param_group["eps"] = self.config.RL.PPO.eps
                            agent_lr_scheduler.base_lrs[i] = 1.0
                        logger.info("Start actor finetuning at: {}".format(self.num_updates_done))

                        logger.info(
                            "updated agent number of parameters: {}".format(
                                sum(param.numel() if param.requires_grad else 0 for param in self.agent.parameters())
                            )
                        )
                    if self.num_updates_done == self.start_critic_warmup_at:
                        self.agent.optimizer.param_groups[0]["eps"] = self.config.RL.PPO.eps
                        agent_lr_scheduler.base_lrs[0] = 1.0
                        logger.info("Set critic LR at: {}".format(self.num_updates_done))

                    # if self.num_updates_done > self.actor_finetuning_update:
                    agent_lr_scheduler.step()

                self.agent.eval()
                count_steps_delta = 0
                profiling_wrapper.range_push("rollouts loop")

                profiling_wrapper.range_push("_collect_rollout_step")
                for buffer_index in range(self._nbuffers):
                    self._compute_actions_and_step_envs(buffer_index)

                for step in range(ppo_cfg.num_steps):
                    is_last_step = (
                        self.should_end_early(step + 1)
                        or (step + 1) == ppo_cfg.num_steps
                    )

                    for buffer_index in range(self._nbuffers):
                        count_steps_delta += self._collect_environment_result(
                            buffer_index
                        )

                        if (buffer_index + 1) == self._nbuffers:
                            profiling_wrapper.range_pop()  # _collect_rollout_step

                        if not is_last_step:
                            if (buffer_index + 1) == self._nbuffers:
                                profiling_wrapper.range_push(
                                    "_collect_rollout_step"
                                )

                            self._compute_actions_and_step_envs(buffer_index)

                    if is_last_step:
                        break

                profiling_wrapper.range_pop()  # rollouts loop

                if self._is_distributed:
                    self.num_rollouts_done_store.add("num_done", 1)

                # update discriminator
                discrim_loss_agent, discrim_loss_demo, discrim_total_loss = self._update_discriminator()

                # update agent
                value_loss, action_loss, dist_entropy = self._update_agent()

                if ppo_cfg.use_linear_lr_decay:
                    agent_lr_scheduler.step()  # type: ignore
                if self.config.GAIL.DISCRIMINATOR.use_linear_lr_decay:
                    discrim_lr_scheduler.step()

                self.num_updates_done += 1
                losses = self._coalesce_post_step(
                    dict(
                        value_loss=value_loss,
                        action_loss=action_loss,
                        entropy=dist_entropy,
                        discrim_loss_agent=discrim_loss_agent,
                        discrim_loss_demo=discrim_loss_demo,
                        discrim_total_loss=discrim_total_loss,
                    ),
                    count_steps_delta,
                )

                self._training_log(writer, losses, prev_time)

                # checkpoint model
                if rank0_only() and self.should_checkpoint():
                    self.save_checkpoint(
                        f"ckpt.{count_checkpoints}.{self.num_steps_done}.pth",
                        dict(
                            step=self.num_steps_done,
                            wall_time=(time.time() - self.t_start) + prev_time,
                        ),
                    )
                    count_checkpoints += 1

                profiling_wrapper.range_pop()  # train update

            self.envs.close()

    def _eval_checkpoint(
        self,
        checkpoint_path: str,
        writer: TensorboardWriter,
        checkpoint_index: int = 0,
    ) -> None:
        r"""Evaluates a single checkpoint.

        Args:
            checkpoint_path: path of checkpoint
            writer: tensorboard writer object for logging to tensorboard
            checkpoint_index: index of cur checkpoint for logging

        Returns:
            None
        """
        if self._is_distributed:
            raise RuntimeError("Evaluation does not support distributed mode")

        # Map location CPU is almost always better than mapping to a CUDA device.
        ckpt_dict = self.load_checkpoint(checkpoint_path, map_location="cpu")

        if self.config.EVAL.USE_CKPT_CONFIG:
            config = self._setup_eval_config(ckpt_dict["config"])
        else:
            config = self.config.clone()

        config.defrost()
        config.GAIL.is_demonstration_env = False
        # config.TASK_CONFIG.DATASET.SPLIT = config.EVAL.SPLIT
        # config.TASK_CONFIG.DATASET.SPLIT = "val_mini"
        # config.NUM_ENVIRONMENTS = 2
        config.freeze()

        # self.config.defrost()
        # self.config.NUM_ENVIRONMENTS = 2
        # self.config.freeze()

        ppo_cfg = config.RL.PPO

        if len(self.config.VIDEO_OPTION) > 0:
            config.defrost()
            config.TASK_CONFIG.TASK.MEASUREMENTS.append("TOP_DOWN_MAP")
            config.TASK_CONFIG.TASK.MEASUREMENTS.append("COLLISIONS")
            config.freeze()

        # if config.VERBOSE:
        logger.info(f"env config: {config}")

        self._init_eval_envs(config)

        if self.using_velocity_ctrl:
            self.policy_action_space = self.envs.action_spaces[0][
                "VELOCITY_CONTROL"
            ]
            action_shape = (2,)
            action_type = torch.float
        else:
            self.policy_action_space = self.envs.action_spaces[0]
            action_shape = (1,)
            action_type = torch.long

        # Load RedNet for semantics prediction
        if self.config.MODEL.USE_PRED_SEMANTICS:
            from gail.models.rednet import load_rednet
            self.semantic_predictor = load_rednet(
                self.device,
                ckpt=self.config.MODEL.SEMANTIC_ENCODER.rednet_ckpt,
                resize=True, # since we train on half-vision
                num_classes=self.config.MODEL.SEMANTIC_ENCODER.num_classes
            )
            self.semantic_predictor.eval()

        self._setup_actor_critic_agent(ppo_cfg)

        self.agent.load_state_dict(ckpt_dict["agent_state_dict"])
        self.actor_critic = self.agent.actor_critic

        print("number of episodes in envs:", self.envs.number_of_episodes)
        observations = self.envs.reset()
        batch = batch_obs(
            observations, device=self.device, cache=self._obs_batching_cache
        )
        batch = apply_obs_transforms_batch(batch, self.obs_transforms)  # type: ignore

        current_episode_reward = torch.zeros(
            self.envs.num_envs, 1, device="cpu"
        )
        print("self.config.NUM_ENVIRONMENTS", self.config.NUM_ENVIRONMENTS)

        test_recurrent_hidden_states = torch.zeros(
            self.config.NUM_ENVIRONMENTS,
            self.actor_critic.net.num_recurrent_layers,
            self.config.MODEL.STATE_ENCODER.hidden_size, #ppo_cfg.hidden_size,
            device=self.device,
        )
        prev_actions = torch.zeros(
            self.config.NUM_ENVIRONMENTS,
            *action_shape,
            device=self.device,
            dtype=action_type,
        )
        not_done_masks = torch.zeros(
            self.config.NUM_ENVIRONMENTS,
            1,
            device=self.device,
            dtype=torch.bool,
        )
        stats_episodes: Dict[
            Any, Any
        ] = {}  # dict of dicts that stores stats per episode

        rgb_frames = [
            [] for _ in range(self.config.NUM_ENVIRONMENTS)
        ]  # type: List[List[np.ndarray]]
        if len(self.config.VIDEO_OPTION) > 0:
            os.makedirs(self.config.VIDEO_DIR, exist_ok=True)

        number_of_eval_episodes = self.config.TEST_EPISODE_COUNT
        if number_of_eval_episodes == -1:
            number_of_eval_episodes = sum(self.envs.number_of_episodes)
        else:
            total_num_eps = sum(self.envs.number_of_episodes)
            if total_num_eps < number_of_eval_episodes:
                logger.warn(
                    f"Config specified {number_of_eval_episodes} eval episodes"
                    ", dataset only has {total_num_eps}."
                )
                logger.warn(f"Evaluating with {total_num_eps} instead.")
                number_of_eval_episodes = total_num_eps

        pbar = tqdm.tqdm(total=number_of_eval_episodes)
        self.actor_critic.eval()
        while (
            len(stats_episodes) < number_of_eval_episodes
            and self.envs.num_envs > 0
        ):
            current_episodes = self.envs.current_episodes()

            if self.semantic_predictor is not None:
                batch["semantic"] = self.semantic_predictor(batch["rgb"],
                                                            batch["depth"])
                if self.config.MODEL.SEMANTIC_ENCODER.is_thda:
                    batch["semantic"] = batch["semantic"] - 1

            with torch.no_grad():
                (
                    _,
                    actions,
                    _,
                    test_recurrent_hidden_states,
                ) = self.actor_critic.act(
                    batch,
                    test_recurrent_hidden_states,
                    prev_actions,
                    not_done_masks,
                    deterministic=False,
                )

                prev_actions.copy_(actions)  # type: ignore
            # NB: Move actions to CPU.  If CUDA tensors are
            # sent in to env.step(), that will create CUDA contexts
            # in the subprocesses.
            # For backwards compatibility, we also call .item() to convert to
            # an int
            if self.using_velocity_ctrl:
                step_data = [
                    action_to_velocity_control(a)
                    for a in actions.to(device="cpu")
                ]
            else:
                step_data = [a.item() for a in actions.to(device="cpu")]

            outputs = self.envs.step(step_data)

            observations, rewards_l, dones, infos = [
                list(x) for x in zip(*outputs)
            ]
            batch = batch_obs(  # type: ignore
                observations,
                device=self.device,
                cache=self._obs_batching_cache,
            )
            batch = apply_obs_transforms_batch(batch, self.obs_transforms)  # type: ignore

            not_done_masks = torch.tensor(
                [[not done] for done in dones],
                dtype=torch.bool,
                device="cpu",
            )

            rewards = torch.tensor(
                rewards_l, dtype=torch.float, device="cpu"
            ).unsqueeze(1)
            current_episode_reward += rewards
            next_episodes = self.envs.current_episodes()
            envs_to_pause = []
            n_envs = self.envs.num_envs
            for i in range(n_envs):
                if (
                    next_episodes[i].scene_id,
                    next_episodes[i].episode_id,
                ) in stats_episodes:
                    envs_to_pause.append(i)

                # episode ended
                if not not_done_masks[i].item():
                    pbar.update()
                    episode_stats = {
                        "task_reward": current_episode_reward[i].item()
                    }
                    episode_stats.update(
                        self._extract_scalars_from_info(infos[i])
                    )
                    current_episode_reward[i] = 0
                    # use scene_id + episode_id as unique id for storing stats
                    stats_episodes[
                        (
                            current_episodes[i].scene_id,
                            current_episodes[i].episode_id,
                        )
                    ] = episode_stats

                    if len(self.config.VIDEO_OPTION) > 0:
                        generate_video(
                            video_option=self.config.VIDEO_OPTION,
                            video_dir=self.config.VIDEO_DIR,
                            images=rgb_frames[i],
                            episode_id=current_episodes[i].episode_id,
                            checkpoint_idx=checkpoint_index,
                            metrics=self._extract_scalars_from_info(infos[i]),
                            tb_writer=writer,
                        )

                        rgb_frames[i] = []

                # episode continues
                elif len(self.config.VIDEO_OPTION) > 0:
                    # TODO move normalization / channel changing out of the policy and undo it here
                    frame = observations_to_image(
                        {k: v[i] for k, v in batch.items()}, infos[i]
                    )
                    rgb_frames[i].append(frame)

            not_done_masks = not_done_masks.to(device=self.device)
            (
                self.envs,
                test_recurrent_hidden_states,
                not_done_masks,
                current_episode_reward,
                prev_actions,
                batch,
                rgb_frames,
            ) = self._pause_envs(
                envs_to_pause,
                self.envs,
                test_recurrent_hidden_states,
                not_done_masks,
                current_episode_reward,
                prev_actions,
                batch,
                rgb_frames,
            )

        num_episodes = len(stats_episodes)
        aggregated_stats = {}
        for stat_key in next(iter(stats_episodes.values())).keys():
            aggregated_stats[stat_key] = (
                sum(v[stat_key] for v in stats_episodes.values())
                / num_episodes
            )

        for k, v in aggregated_stats.items():
            logger.info(f"Average episode {k}: {v:.4f}")

        step_id = checkpoint_index
        if "extra_state" in ckpt_dict and "step" in ckpt_dict["extra_state"]:
            step_id = ckpt_dict["extra_state"]["step"]

        writer.add_scalar(
            "eval_reward/average_task_reward", aggregated_stats["task_reward"], step_id
        )

        metrics = {k: v for k, v in aggregated_stats.items() if k != "task_reward"}
        for k, v in metrics.items():
            writer.add_scalar(f"eval_metrics/{k}", v, step_id)

        self.envs.close()

    @staticmethod
    def linear_warmup(epoch: int, start_update: int, max_updates: int,
                      start_lr: int, end_lr: int) -> float:
        r"""Returns a multiplicative factor for linear value decay

        Args:
            epoch: current epoch number
            total_num_updates: total number of

        Returns:
            multiplicative factor that decreases param value linearly
        """
        if epoch < start_update:
            return 1.0

        if epoch > max_updates:
            return end_lr

        pct_step = (epoch - start_update) / (max_updates - start_update)
        step_lr = (end_lr - start_lr) * pct_step + start_lr
        if step_lr > end_lr:
            step_lr = end_lr
        # logger.info("{}, {}, {}, {}, {}, {}".format(epoch, start_update, max_updates, start_lr, end_lr, step_lr))
        return step_lr

    @staticmethod
    def critic_linear_decay(epoch: int, start_update: int, max_updates: int,
                            start_lr: int, end_lr: int) -> float:
        r"""Returns a multiplicative factor for linear value decay

        Args:
            epoch: current epoch number
            total_num_updates: total number of

        Returns:
            multiplicative factor that decreases param value linearly
        """
        if epoch <= start_update:
            return 1

        if epoch > max_updates:
            return end_lr

        pct_step = (epoch - start_update) / (max_updates - start_update)
        step_lr = start_lr - (start_lr - end_lr) * pct_step
        if step_lr < end_lr:
            step_lr = end_lr
        # logger.info(
        #     "{}, {}, {}, {}, {}, {}".format(epoch, start_update, max_updates,
        #                                     start_lr, end_lr, step_lr))
        return step_lr
