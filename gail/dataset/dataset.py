#!/usr/bin/env python3

# Created by Yunhai Feng at 9:51 pm, 2022/4/30.

import json
import os
from typing import Any, Dict, List, Optional, Sequence

import attr
from habitat.config import Config

from habitat.core.registry import registry
from habitat.core.simulator import AgentState, ShortestPathPoint
from habitat.core.utils import DatasetFloatJSONEncoder, not_none_validator
from habitat.datasets.pointnav.pointnav_dataset import (
    CONTENT_SCENES_PATH_FIELD,
    DEFAULT_SCENE_PATH_PREFIX,
    PointNavDatasetV1,
)
from habitat.tasks.nav.nav import NavigationEpisode
from habitat.tasks.nav.object_nav_task import (
    ObjectGoal,
    # ObjectGoalNavEpisode,
    ObjectViewLocation,
)


@attr.s(auto_attribs=True, kw_only=True)
class ObjectGoalNavEpisode(NavigationEpisode):
    r"""ObjectGoal Navigation Episode

    :param object_category: Category of the obect
    """
    object_category: Optional[str] = None
    demo_sequence: Optional[List[str]] = None

    @property
    def goals_key(self) -> str:
        r"""The key to retrieve the goals"""
        return f"{os.path.basename(self.scene_id)}_{self.object_category}"


# @attr.s(auto_attribs=True, kw_only=True)
# class ObjectInScene:
#     object_id: int = attr.ib(default=None, validator=not_none_validator)
#     semantic_category_id: int = attr.ib(default=None)
#     object_template: str = attr.ib(default=None, validator=not_none_validator)
#     scale: float = attr.ib(default=None)
#     position: List[float] = attr.ib(default=None)
#     rotation: List[float] = attr.ib(default=None)
#
#
# @attr.s(auto_attribs=True, kw_only=True)
# class SceneState:
#     objects: List[ObjectInScene] = attr.ib(default=None)
#
#
# @attr.s(auto_attribs=True, kw_only=True)
# class AgentStateSpec:
#     r"""Agent data specifications that capture states of agent and sensor in replay state.
#     """
#     position: Optional[List[float]] = attr.ib(default=None)
#     rotation: Optional[List[float]] = attr.ib(default=None)
#     sensor_data: Optional[dict] = attr.ib(default=None)
#
#
# @attr.s(auto_attribs=True, kw_only=True)
# class ReplayActionSpec:
#     r"""Replay specifications that capture metadata associated with action.
#     """
#     action: str = attr.ib(default=None, validator=not_none_validator)
#     agent_state: Optional[AgentStateSpec] = attr.ib(default=None)
#
#
# @attr.s(auto_attribs=True, kw_only=True)
# class ObjectGoalNavEpisode(NavigationEpisode):
#     r"""ObjectGoal Navigation Episode
#
#     :param object_category: Category of the obect
#     """
#     object_category: Optional[str] = None
#     reference_replay: Optional[List[ReplayActionSpec]] = None
#     scene_state: Optional[List[SceneState]] = None
#     is_thda: Optional[bool] = False
#     scene_dataset: Optional[str] = "mp3d"
#     scene_dataset_config: Optional[str] = ""
#     additional_obj_config_paths: Optional[List] = []
#     attempts: Optional[int] = 1
#
#     @property
#     def goals_key(self) -> str:
#         r"""The key to retrieve the goals"""
#         return f"{os.path.basename(self.scene_id)}_{self.object_category}"

#
# @registry.register_dataset(name="ObjectNav-demo")
# class ObjectNavDemoDataset(PointNavDatasetV1):
#     r"""Class inherited from PointNavDataset that loads Object Navigation
#     dataset containing demonstrations. """
#     category_to_task_category_id: Dict[str, int]
#     category_to_scene_annotation_category_id: Dict[str, int]
#     episodes: List[ObjectGoalNavEpisode] = []  # type: ignore
#     content_scenes_path: str = "{data_path}/content/{scene}.json.gz"
#     goals_by_category: Dict[str, Sequence[ObjectGoal]]
#     gibson_to_mp3d_category_map: Dict[str, str] = {'couch': 'sofa',
#                                                    'toilet': 'toilet',
#                                                    'bed': 'bed',
#                                                    'tv': 'tv_monitor',
#                                                    'potted plant': 'plant',
#                                                    'chair': 'chair'}
#     max_episode_steps: int = 500
#
#     @staticmethod
#     def dedup_goals(dataset: Dict[str, Any]) -> Dict[str, Any]:
#         if len(dataset["episodes"]) == 0:
#             return dataset
#
#         goals_by_category = {}
#         for i, ep in enumerate(dataset["episodes"]):
#             dataset["episodes"][i]["object_category"] = ep["goals"][0][
#                 "object_category"
#             ]
#             ep = ObjectGoalNavEpisode(**ep)
#
#             goals_key = ep.goals_key
#             if goals_key not in goals_by_category:
#                 goals_by_category[goals_key] = ep.goals
#
#             dataset["episodes"][i]["goals"] = []
#
#         dataset["goals_by_category"] = goals_by_category
#
#         return dataset
#
#     def to_json(self) -> str:
#         for i in range(len(self.episodes)):
#             self.episodes[i].goals = []
#
#         result = DatasetFloatJSONEncoder().encode(self)
#
#         for i in range(len(self.episodes)):
#             goals = self.goals_by_category[self.episodes[i].goals_key]
#             if not isinstance(goals, list):
#                 goals = list(goals)
#             self.episodes[i].goals = goals
#
#         return result
#
#     def __init__(self, config: Optional[Config] = None) -> None:
#         self.goals_by_category = {}
#         if config is not None:
#             self.max_episode_steps = config.MAX_EPISODE_STEPS
#         super().__init__(config)
#         self.episodes = list(self.episodes)
#
#     @staticmethod
#     def __deserialize_goal(serialized_goal: Dict[str, Any]) -> ObjectGoal:
#         g = ObjectGoal(**serialized_goal)
#
#         for vidx, view in enumerate(g.view_points):
#             view_location = ObjectViewLocation(**view)  # type: ignore
#             view_location.agent_state = AgentState(
#                 **view_location.agent_state)  # type: ignore
#             g.view_points[vidx] = view_location
#
#         return g
#
#     def from_json(
#         self, json_str: str, scenes_dir: Optional[str] = None
#     ) -> None:
#         deserialized = json.loads(json_str)
#         if CONTENT_SCENES_PATH_FIELD in deserialized:
#             self.content_scenes_path = deserialized[CONTENT_SCENES_PATH_FIELD]
#
#         if "category_to_task_category_id" in deserialized:
#             self.category_to_task_category_id = deserialized[
#                 "category_to_task_category_id"
#             ]
#
#         if "category_to_scene_annotation_category_id" in deserialized:
#             self.category_to_scene_annotation_category_id = deserialized[
#                 "category_to_scene_annotation_category_id"
#             ]
#
#         if "category_to_mp3d_category_id" in deserialized:
#             self.category_to_scene_annotation_category_id = deserialized[
#                 "category_to_mp3d_category_id"
#             ]
#
#         assert len(self.category_to_task_category_id) == len(
#             self.category_to_scene_annotation_category_id
#         )
#
#         assert set(self.category_to_task_category_id.keys()) == set(
#             self.category_to_scene_annotation_category_id.keys()
#         ), "category_to_task and category_to_mp3d must have the same keys"
#
#         if len(deserialized["episodes"]) == 0:
#             return
#
#         if "goals_by_category" not in deserialized:
#             deserialized = self.dedup_goals(deserialized)
#
#         for k, v in deserialized["goals_by_category"].items():
#             self.goals_by_category[k] = [self.__deserialize_goal(g) for g in v]
#
#         for i, episode in enumerate(deserialized["episodes"]):
#             if "_shortest_path_cache" in episode:
#                 del episode["_shortest_path_cache"]
#
#             if "gibson" in episode["scene_id"]:
#                 episode["scene_id"] = "gibson_semantic/{}".format(
#                     episode["scene_id"].split("/")[-1])
#
#             episode = ObjectGoalNavEpisode(**episode)
#             # episode.episode_id = str(i)
#
#             if scenes_dir is not None:
#                 if episode.scene_id.startswith(DEFAULT_SCENE_PATH_PREFIX):
#                     episode.scene_id = episode.scene_id[
#                                        len(DEFAULT_SCENE_PATH_PREFIX):
#                                        ]
#
#                 episode.scene_id = os.path.join(scenes_dir, episode.scene_id)
#
#             if not episode.is_thda:
#                 episode.goals = self.goals_by_category[episode.goals_key]
#                 if episode.scene_dataset == "gibson":
#                     print("Should not contain Gibson")
#                     episode.object_category = self.gibson_to_mp3d_category_map[
#                         episode.object_category]
#             else:
#                 # print(f"Should not contain THDA! scene dataset {episode.scene_dataset}, scenes dir: {scenes_dir}, scene_id: {episode.scene_id}") #
#                 continue
#                 goals = []
#                 for g in episode.goals:
#                     g = ObjectGoal(**g)
#                     for vidx, view in enumerate(g.view_points):
#                         view_location = ObjectViewLocation(
#                             **view)  # type: ignore
#                         view_location.agent_state = AgentState(
#                             **view_location.agent_state)  # type: ignore
#                         g.view_points[vidx] = view_location
#                     goals.append(g)
#                 episode.goals = goals
#
#                 objects = [ObjectInScene(**o) for o in
#                            episode.scene_state["objects"]]
#                 scene_state = [SceneState(objects=objects).__dict__]
#                 episode.scene_state = scene_state
#
#             if episode.reference_replay is not None:
#                 for i, replay_step in enumerate(episode.reference_replay):
#                     # replay_step["agent_state"] = AgentStateSpec(**replay_step["agent_state"])
#                     replay_step["agent_state"] = None
#                     episode.reference_replay[i] = ReplayActionSpec(
#                         **replay_step)
#
#             if episode.shortest_paths is not None:
#                 for path in episode.shortest_paths:
#                     for p_index, point in enumerate(path):
#                         if point is None or isinstance(point, (int, str)):
#                             point = {
#                                 "action": point,
#                                 "rotation": None,
#                                 "position": None,
#                             }
#
#                         path[p_index] = ShortestPathPoint(**point)
#
#             if len(episode.reference_replay) > self.max_episode_steps:
#                 continue
#
#             self.episodes.append(episode)  # type: ignore [attr-defined]
#
#
# from habitat.datasets.object_nav.object_nav_dataset import ObjectNavDatasetV1
# @registry.register_dataset(name="ObjectNav-v7")
# class ObjectNavDatasetV7(ObjectNavDatasetV1):
#     episodes: List[ObjectGoalNavEpisode] = []  # type: ignore
#     def from_json(
#         self, json_str: str, scenes_dir: Optional[str] = None
#     ) -> None:
#         deserialized = json.loads(json_str)
#         if CONTENT_SCENES_PATH_FIELD in deserialized:
#             self.content_scenes_path = deserialized[CONTENT_SCENES_PATH_FIELD]
#
#         if "category_to_task_category_id" in deserialized:
#             self.category_to_task_category_id = deserialized[
#                 "category_to_task_category_id"
#             ]
#
#         if "category_to_scene_annotation_category_id" in deserialized:
#             self.category_to_scene_annotation_category_id = deserialized[
#                 "category_to_scene_annotation_category_id"
#             ]
#
#         if "category_to_mp3d_category_id" in deserialized:
#             self.category_to_scene_annotation_category_id = deserialized[
#                 "category_to_mp3d_category_id"
#             ]
#
#         assert len(self.category_to_task_category_id) == len(
#             self.category_to_scene_annotation_category_id
#         )
#
#         assert set(self.category_to_task_category_id.keys()) == set(
#             self.category_to_scene_annotation_category_id.keys()
#         ), "category_to_task and category_to_mp3d must have the same keys"
#
#         if len(deserialized["episodes"]) == 0:
#             return
#
#         if "goals_by_category" not in deserialized:
#             deserialized = self.dedup_goals(deserialized)
#
#         for k, v in deserialized["goals_by_category"].items():
#             self.goals_by_category[k] = [self.__deserialize_goal(g) for g in v]
#
#         for i, episode in enumerate(deserialized["episodes"]):
#             if "_shortest_path_cache" in episode:
#                 del episode["_shortest_path_cache"]
#
#             if "gibson" in episode["scene_id"]:
#                 print("Should not contain Gibson scene_id")
#                 episode["scene_id"] = "gibson_semantic/{}".format(
#                     episode["scene_id"].split("/")[-1])
#
#             episode = ObjectGoalNavEpisode(**episode)
#             # episode.episode_id = str(i)
#
#             if scenes_dir is not None:
#                 if episode.scene_id.startswith(DEFAULT_SCENE_PATH_PREFIX):
#                     episode.scene_id = episode.scene_id[
#                                        len(DEFAULT_SCENE_PATH_PREFIX):
#                                        ]
#
#                 episode.scene_id = os.path.join(scenes_dir, episode.scene_id)
#
#             if not episode.is_thda:
#                 episode.goals = self.goals_by_category[episode.goals_key]
#                 if episode.scene_dataset == "gibson":
#                     print("Should not contain Gibson")
#                     episode.object_category = self.gibson_to_mp3d_category_map[
#                         episode.object_category]
#             else:
#                 # print(f"Should not contain THDA! scene dataset {episode.scene_dataset}, scenes dir: {scenes_dir}, scene_id: {episode.scene_id}") #
#                 continue
#                 goals = []
#                 for g in episode.goals:
#                     g = ObjectGoal(**g)
#                     for vidx, view in enumerate(g.view_points):
#                         view_location = ObjectViewLocation(
#                             **view)  # type: ignore
#                         view_location.agent_state = AgentState(
#                             **view_location.agent_state)  # type: ignore
#                         g.view_points[vidx] = view_location
#                     goals.append(g)
#                 episode.goals = goals
#
#                 objects = [ObjectInScene(**o) for o in
#                            episode.scene_state["objects"]]
#                 scene_state = [SceneState(objects=objects).__dict__]
#                 episode.scene_state = scene_state
#
#             if episode.reference_replay is not None:
#                 for i, replay_step in enumerate(episode.reference_replay):
#                     # replay_step["agent_state"] = AgentStateSpec(**replay_step["agent_state"])
#                     replay_step["agent_state"] = None
#                     episode.reference_replay[i] = ReplayActionSpec(
#                         **replay_step)
#
#             if episode.shortest_paths is not None:
#                 for path in episode.shortest_paths:
#                     for p_index, point in enumerate(path):
#                         if point is None or isinstance(point, (int, str)):
#                             point = {
#                                 "action": point,
#                                 "rotation": None,
#                                 "position": None,
#                             }
#
#                         path[p_index] = ShortestPathPoint(**point)
#
#             if len(episode.reference_replay) > self.max_episode_steps:
#                 continue
#
#             self.episodes.append(episode)  # type: ignore [attr-defined]


@registry.register_dataset(name="ObjectNav-demo")
class ObjectNavDemoDataset(PointNavDatasetV1):
    r"""Class inherited from PointNavDataset that loads Object Navigation dataset."""
    category_to_task_category_id: Dict[str, int]
    category_to_scene_annotation_category_id: Dict[str, int]
    episodes: List[ObjectGoalNavEpisode] = []  # type: ignore
    content_scenes_path: str = "{data_path}/content/{scene}.json.gz"
    goals_by_category: Dict[str, Sequence[ObjectGoal]]

    @staticmethod
    def dedup_goals(dataset: Dict[str, Any]) -> Dict[str, Any]:
        if len(dataset["episodes"]) == 0:
            return dataset

        goals_by_category = {}
        for i, ep in enumerate(dataset["episodes"]):
            dataset["episodes"][i]["object_category"] = ep["goals"][0][
                "object_category"
            ]
            ep = ObjectGoalNavEpisode(**ep)

            goals_key = ep.goals_key
            if goals_key not in goals_by_category:
                goals_by_category[goals_key] = ep.goals

            dataset["episodes"][i]["goals"] = []

        dataset["goals_by_category"] = goals_by_category

        return dataset

    def to_json(self) -> str:
        for i in range(len(self.episodes)):
            self.episodes[i].goals = []

        result = DatasetFloatJSONEncoder().encode(self)

        for i in range(len(self.episodes)):
            goals = self.goals_by_category[self.episodes[i].goals_key]
            if not isinstance(goals, list):
                goals = list(goals)
            self.episodes[i].goals = goals

        return result

    def __init__(self, config: Optional[Config] = None) -> None:
        self.goals_by_category = {}
        super().__init__(config)
        self.episodes = list(self.episodes)

    @staticmethod
    def __deserialize_goal(serialized_goal: Dict[str, Any]) -> ObjectGoal:
        g = ObjectGoal(**serialized_goal)

        for vidx, view in enumerate(g.view_points):
            view_location = ObjectViewLocation(**view)  # type: ignore
            view_location.agent_state = AgentState(
                **view_location.agent_state)  # type: ignore
            g.view_points[vidx] = view_location

        return g

    def from_json(
        self, json_str: str, scenes_dir: Optional[str] = None
    ) -> None:
        deserialized = json.loads(json_str)
        if CONTENT_SCENES_PATH_FIELD in deserialized:
            self.content_scenes_path = deserialized[CONTENT_SCENES_PATH_FIELD]

        if "category_to_task_category_id" in deserialized:
            self.category_to_task_category_id = deserialized[
                "category_to_task_category_id"
            ]

        if "category_to_scene_annotation_category_id" in deserialized:
            self.category_to_scene_annotation_category_id = deserialized[
                "category_to_scene_annotation_category_id"
            ]

        if "category_to_mp3d_category_id" in deserialized:
            self.category_to_scene_annotation_category_id = deserialized[
                "category_to_mp3d_category_id"
            ]

        assert len(self.category_to_task_category_id) == len(
            self.category_to_scene_annotation_category_id
        )

        assert set(self.category_to_task_category_id.keys()) == set(
            self.category_to_scene_annotation_category_id.keys()
        ), "category_to_task and category_to_mp3d must have the same keys"

        if len(deserialized["episodes"]) == 0:
            return

        if "goals_by_category" not in deserialized:
            deserialized = self.dedup_goals(deserialized)

        for k, v in deserialized["goals_by_category"].items():
            self.goals_by_category[k] = [self.__deserialize_goal(g) for g in v]

        for i, episode in enumerate(deserialized["episodes"]):

            # filter out undesired episodes
            if "is_thda" in episode and episode["is_thda"] is not None \
              or "scene_dataset" in episode and episode["scene_dataset"] == "gibson":
              # or "reference_replay" in episode and len(episode["reference_replay"]) > 500:
                continue

            # reformat demonstration sequence
            demo_sequence = []
            for transition in episode["reference_replay"]:
                demo_sequence.append(transition["action"])
            episode["demo_sequence"] = demo_sequence

            # filter out unnecessary fields
            for k in ["reference_replay",
                      "scene_state",
                      "is_thda",
                      "scene_dataset",
                      "scene_dataset_config",
                      "additional_obj_config_paths",
                      "attempts"]:
                if k in episode:
                    del episode[k]

            episode = ObjectGoalNavEpisode(**episode)
            episode.episode_id = str(i)

            if scenes_dir is not None:
                if episode.scene_id.startswith(DEFAULT_SCENE_PATH_PREFIX):
                    episode.scene_id = episode.scene_id[
                                       len(DEFAULT_SCENE_PATH_PREFIX):
                                       ]

                episode.scene_id = os.path.join(scenes_dir, episode.scene_id)

            episode.goals = self.goals_by_category[episode.goals_key]

            if episode.shortest_paths is not None:
                for path in episode.shortest_paths:
                    for p_index, point in enumerate(path):
                        if point is None or isinstance(point, (int, str)):
                            point = {
                                "action": point,
                                "rotation": None,
                                "position": None,
                            }

                        path[p_index] = ShortestPathPoint(**point)

            self.episodes.append(episode)  # type: ignore [attr-defined]
