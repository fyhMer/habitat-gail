#!/usr/bin/env python3

# Created by Yunhai Feng at 4:04 pm, 2022/4/30.


from typing import Optional, Type, List, Union


import numpy as np
import random
import torch
import numpy as np
import habitat
from habitat import Config, Dataset, Env, RLEnv, VectorEnv, make_dataset, logger
from habitat_baselines.common.baseline_registry import baseline_registry
from habitat_baselines.utils.env_utils import make_env_fn
from habitat import get_config as get_task_config
from habitat.sims.habitat_simulator.actions import HabitatSimActions

from habitat.tasks.nav.object_nav_task import ObjectGoalSensor

task_cat2mpcat40 = [
    3,  # ('chair', 2, 0)
    5,  # ('table', 4, 1)
    6,  # ('picture', 5, 2)
    7,  # ('cabinet', 6, 3)
    8,  # ('cushion', 7, 4)
    10,  # ('sofa', 9, 5),
    11,  # ('bed', 10, 6)
    13,  # ('chest_of_drawers', 12, 7),
    14,  # ('plant', 13, 8)
    15,  # ('sink', 14, 9)
    18,  # ('toilet', 17, 10),
    19,  # ('stool', 18, 11),
    20,  # ('towel', 19, 12)
    22,  # ('tv_monitor', 21, 13)
    23,  # ('shower', 22, 14)
    25,  # ('bathtub', 24, 15)
    26,  # ('counter', 25, 16),
    27,  # ('fireplace', 26, 17),
    33,  # ('gym_equipment', 32, 18),
    34,  # ('seating', 33, 19),
    38,  # ('clothes', 37, 20),
    43,  # ('foodstuff', 42, 21),
    44,  # ('stationery', 43, 22),
    45,  # ('fruit', 44, 23),
    46,  # ('plaything', 45, 24),
    47,  # ('hand_tool', 46, 25),
    48,  # ('game_equipment', 47, 26),
    49,  # ('kitchenware', 48, 27)
]

task_cat2hm3dcat40 = [
    3,  # ('chair', 2, 0)
    11,  # ('bed', 10, 6)
    14,  # ('plant', 13, 8)
    18,  # ('toilet', 17, 10),
    22,  # ('tv_monitor', 21, 13)
    10,  # ('sofa', 9, 5),
]

mapping_mpcat40_to_goal21 = {
    3: 1,
    5: 2,
    6: 3,
    7: 4,
    8: 5,
    10: 6,
    11: 7,
    13: 8,
    14: 9,
    15: 10,
    18: 11,
    19: 12,
    20: 13,
    22: 14,
    23: 15,
    25: 16,
    26: 17,
    27: 18,
    33: 19,
    34: 20,
    38: 21,
    43: 22,  #  ('foodstuff', 42, task_cat: 21)
    44: 28,  #  ('stationery', 43, task_cat: 22)
    45: 26,  #  ('fruit', 44, task_cat: 23)
    46: 25,  #  ('plaything', 45, task_cat: 24)
    47: 24,  # ('hand_tool', 46, task_cat: 25)
    48: 23,  # ('game_equipment', 47, task_cat: 26)
    49: 27,  # ('kitchenware', 48, task_cat: 27)
}

@baseline_registry.register_env(name="NavGAILEnv")
class NavGAILEnv(habitat.RLEnv):
    def __init__(self, config: Config, dataset:
                 Optional[Dataset] = None,
                 # device: Optional[torch.device] = torch.device("cpu")
                 ):
        self._rl_config = config.RL
        self._gail_config = config.GAIL
        self._core_env_config = config.TASK_CONFIG
        self._reward_measure_name = self._rl_config.REWARD_MEASURE
        self._success_measure_name = self._rl_config.SUCCESS_MEASURE

        self._is_demonstration_env = config.GAIL.is_demonstration_env
        self._demonstration_timestep = 1

        self._previous_measure = None
        self._previous_action = None
        print(">>>>>>>> init Env")
        super().__init__(self._core_env_config, dataset)
        print("<<<<<<< finish init Env")

        # self._device = device
        if int(self.habitat_env.observation_space[ObjectGoalSensor.cls_uuid].high[0]) + 1 == 6:
            self._object_mapping = task_cat2hm3dcat40
        else:
            self._object_mapping = task_cat2mpcat40

    def get_demonstration_action(self):
        assert self._is_demonstration_env
        demonstrations = self.habitat_env.current_episode.demo_sequence
        if self._demonstration_timestep < len(demonstrations):
            action_name = demonstrations[self._demonstration_timestep]
            action = self._get_action_by_name(action_name)
        else:
            action = 0

        self._demonstration_timestep += 1
        return action

    def reset(self):
        if self._is_demonstration_env:
            self._demonstration_timestep = 1    # TODO: check 0 or 1?
        self._previous_action = None
        observations = super().reset()
        observations = self._map_objectgoal_in_obs(observations)
        self._previous_measure = self._env.get_metrics()[
            self._reward_measure_name
        ]
        return observations

    def step(self, *args, **kwargs):
        demo_action = None
        if self._is_demonstration_env:
            # overwrite `action` by demonstration action
            demo_action = self.get_demonstration_action()
            kwargs["action"] = {"action": demo_action} # TODO: check dict format
        self._previous_action = kwargs["action"]
        observations = self._env.step(*args, **kwargs)
        observations = self._map_objectgoal_in_obs(observations)
        # reward = self.get_reward_s_a(observations, kwargs["action"])
        reward = self.get_reward(observations)
        done = self.get_done(observations)
        info = self.get_info(observations)
        info["demo_action"] = demo_action

        return observations, reward, done, info

    def get_reward_range(self):
        # TODO: consider GAIL reward
        return (
            self._rl_config.SLACK_REWARD - 1.0,
            self._rl_config.SUCCESS_REWARD + 1.0,
        )

    # def get_reward(self, observations):
    #     raise NotImplementedError

    def get_reward(self, observations):
        reward = self._rl_config.SLACK_REWARD

        current_measure = self._env.get_metrics()[self._reward_measure_name]

        reward += self._previous_measure - current_measure
        self._previous_measure = current_measure

        if self._episode_success():
            reward += self._rl_config.SUCCESS_REWARD

        return reward

    def get_reward_s_a(self, observations, action):
        gail_reward = ...   # TODO: compute GAIL reward

        task_reward = self._rl_config.SLACK_REWARD

        current_measure = self._env.get_metrics()[self._reward_measure_name]

        task_reward += self._previous_measure - current_measure
        self._previous_measure = current_measure

        if self._episode_success():
            task_reward += self._rl_config.SUCCESS_REWARD

        return gail_reward * self._gail_config.gail_reward_coef \
             + task_reward * (1 - self._gail_config.gail_reward_coef)

    def _episode_success(self):
        return self._env.get_metrics()[self._success_measure_name]

    def get_done(self, observations):
        done = False
        if self._env.episode_over or self._episode_success():
            done = True
        return done

    def get_info(self, observations):
        return self.habitat_env.get_metrics()

    @staticmethod
    def _get_action_by_name(action):
        if action == "TURN_RIGHT":
            return HabitatSimActions.TURN_RIGHT
        elif action == "TURN_LEFT":
            return HabitatSimActions.TURN_LEFT
        elif action == "MOVE_FORWARD":
            return HabitatSimActions.MOVE_FORWARD
        elif action == "MOVE_BACKWARD":
            return HabitatSimActions.MOVE_BACKWARD
        elif action == "LOOK_UP":
            return HabitatSimActions.LOOK_UP
        elif action == "LOOK_DOWN":
            return HabitatSimActions.LOOK_DOWN
        return HabitatSimActions.STOP

    def _map_to_uniform_objectgoal(self, object_idx):
        if isinstance(object_idx, np.ndarray):
            assert object_idx.shape == (1,)
            object_idx = object_idx.item()
            new_object_idx = mapping_mpcat40_to_goal21[self._object_mapping[object_idx]]
            return np.array([new_object_idx])
        else:
            assert isinstance(object_idx, int), f"expected type `int`, got {type(object_idx)}"
            return mapping_mpcat40_to_goal21[self._object_mapping[object_idx]]

    def _map_objectgoal_in_obs(self, observations):
        # print("1" * 40, "_map_objectgoal_in_obs")
        objectgoal_idx = observations[ObjectGoalSensor.cls_uuid]
        # print("2" * 40, ">", type(objectgoal_idx))
        observations[ObjectGoalSensor.cls_uuid] = self._map_to_uniform_objectgoal(objectgoal_idx)
        # print("3" * 40, ">", observations[ObjectGoalSensor.cls_uuid])
        return observations


def construct_multi_configs(config: Config, num_envs: int) -> List[Config]:
    configs = []
    logger.info(">>>>> making dataset to get scenes info >>>>>")
    task_dataset = make_dataset(config.TASK_CONFIG.DATASET.TYPE)
    scenes = config.TASK_CONFIG.DATASET.CONTENT_SCENES
    if "*" in config.TASK_CONFIG.DATASET.CONTENT_SCENES:
        scenes = task_dataset.get_scenes_to_load(config.TASK_CONFIG.DATASET)
    logger.info("<<<<< finish getting scenes info <<<<<")

    if num_envs > 1:
        if len(scenes) == 0:
            raise RuntimeError(
                "No scenes to load, multiple process logic relies on being "
                "able to split scenes uniquely between processes "
            )

        if len(scenes) < num_envs:
            raise RuntimeError(
                "reduce the number of environments as there "
                "aren't enough number of scenes.\n"
                "num_environments: {}\tnum_scenes: {}".format(
                    num_envs, len(scenes)
                )
            )

        random.shuffle(scenes)

    scene_splits: List[List[str]] = [[] for _ in range(num_envs)]
    for idx, scene in enumerate(scenes):
        scene_splits[idx % len(scene_splits)].append(scene)

    assert sum(map(len, scene_splits)) == len(scenes)

    for i in range(num_envs):
        proc_config = config.clone()
        proc_config.defrost()

        task_config = proc_config.TASK_CONFIG
        task_config.SEED = task_config.SEED + i
        if len(scenes) > 0:
            task_config.DATASET.CONTENT_SCENES = scene_splits[i]

        task_config.SIMULATOR.HABITAT_SIM_V0.GPU_DEVICE_ID = (
            config.SIMULATOR_GPU_ID
        )

        task_config.SIMULATOR.AGENT_0.SENSORS = config.SENSORS

        proc_config.freeze()
        configs.append(proc_config)

    return configs


def construct_gail_envs(
    config: Config,
    env_class: Union[Type[Env], Type[RLEnv]],
    workers_ignore_signals: bool = False,
) -> VectorEnv:
    r"""Create VectorEnv object with specified config and env class type.
    To allow better performance, dataset are split into small ones for
    each individual env, grouped by scenes.

    :param config: configs that contain num_environments as well as information
    :param necessary to create individual environments.
    :param env_class: class type of the envs to be created.
    :param workers_ignore_signals: Passed to :ref:`habitat.VectorEnv`'s constructor

    :return: VectorEnv object created according to specification.
    """

    num_environments = config.NUM_ENVIRONMENTS
    assert num_environments % 2 == 0, "Number of environments should be " \
                                      "dividable by 2. "
    num_homo_environments = int(num_environments / 2)
    env_classes = [env_class for _ in range(num_environments)]
    configs = construct_multi_configs(config, num_homo_environments)

    demo_task_config = get_task_config(config.DEMO_TASK_CONFIG_PATH)
    config.defrost()
    config.TASK_CONFIG = demo_task_config
    config.GAIL.is_demonstration_env = True
    config.freeze()
    demo_configs = construct_multi_configs(config, num_homo_environments)

    assert len(configs + demo_configs) == len(env_classes)

    envs = habitat.VectorEnv(
        make_env_fn=make_env_fn,
        env_fn_args=tuple(zip(configs + demo_configs, env_classes)),
        workers_ignore_signals=workers_ignore_signals,
    )
    return envs
