#!/usr/bin/env python3

# Created by Yunhai Feng at 12:19 下午, 2022/5/5.
from typing import Dict, Any
import numpy as np
import argparse
from tqdm import tqdm
import habitat
from habitat.utils.visualizations.utils import observations_to_image, images_to_video, append_text_to_image
from gail.dataset.dataset import ObjectNavDemoDataset # for registration
from habitat_baselines.utils.common import generate_video

INFO_BLACKLIST = {"demo_action", "top_down_map", "collisions.is_collision"}

def extract_scalars_from_info(info: Dict[str, Any]
) -> Dict[str, float]:
    result = {}
    for k, v in info.items():
        if k in INFO_BLACKLIST:
            continue

        if isinstance(v, dict):
            result.update(
                {
                    k + "." + subk: subv
                    for subk, subv in extract_scalars_from_info(
                    v
                ).items()
                    if (k + "." + subk) not in INFO_BLACKLIST
                }
            )
        # Things that are scalar-like will have an np.size of 1.
        # Strings also have an np.size of 1, so explicitly ban those
        elif np.size(v) == 1 and not isinstance(v, str):
            result[k] = float(v)

    return result

def make_videos(observations_list, output_prefix, ep_id):
    prefix = output_prefix + "_{}".format(ep_id)
    images_to_video(observations_list[0], output_dir="demos", video_name=prefix)

def get_action_by_name(action):
    from habitat.sims.habitat_simulator.actions import HabitatSimActions
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

def run_reference_replay(
    cfg, num_episodes=None, output_prefix=None
):
    possible_actions = cfg.TASK.POSSIBLE_ACTIONS

    dataset = habitat.make_dataset(
        cfg.DATASET.TYPE, config=cfg.DATASET
    )

    with habitat.Env(cfg, dataset) as env:
        total_success = 0
        spl = 0

        num_episodes = min(num_episodes, len(env.episodes))
        print("Replaying {}/{} episodes".format(num_episodes, len(env.episodes)))
        for ep_id in range(num_episodes):
            observation_list = []
            env.reset()

            step_index = 1
            total_reward = 0.0
            episode = env.current_episode

            for _, action_name in tqdm(enumerate(env.current_episode.demo_sequence[step_index:])):
                # action = possible_actions.index(data.action)
                # action_name = env.task.get_action_name(
                #     action
                # )
                print(action_name)
                # action = ["STOP", "MOVE_BACKWARD", "TURN_RIGHT", "TURN_LEFT",
                #            "LOOK_UP", "LOOK_DOWN"].index(action_name)
                action = get_action_by_name(action_name)
                print(action)

                observations = env.step(action=action)

                info = env.get_metrics()
                frame = observations_to_image({"rgb": observations["rgb"]}, info)
                frame = append_text_to_image(frame, "Find and go to {}".format(episode.object_category))

                observation_list.append(frame)
                if action_name == "STOP":
                    break
            # make_videos([observation_list], output_prefix, ep_id)

            #####
            generate_video(
                video_option=["disk"],
                video_dir=output_prefix,
                images=observation_list,
                episode_id=episode.episode_id,
                checkpoint_idx=-1,
                metrics=extract_scalars_from_info(info),
                tb_writer=None,
            )

            print("Total reward for trajectory: {}".format(total_reward))

            print("Episode length:", len(episode.demo_sequence))
            if len(episode.demo_sequence) <= 500:
                print("info:", {k: v for k, v in info.items() if k in {"success", "distance_to_goal", "success", "spl", "softspl"}})
                total_success += info["success"]
                spl += info["spl"]

        print("SPL: {}, {}, {}".format(spl/num_episodes, spl, num_episodes))
        print("Success: {}, {}, {}".format(total_success/num_episodes, total_success, num_episodes))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--path", type=str, default="replays/demo_1.json.gz"
    )
    parser.add_argument(
        "--output-prefix", type=str, default="videos"
    )
    parser.add_argument(
        "--num-episodes", type=int, default=10
    )
    args = parser.parse_args()
    # config = habitat.get_config(
    #     "configs/objectnav_expert_mp3d_single_episode_seed7.yaml")
    config = habitat.get_config(
        "configs/objectnav_expert_mp3d.yaml")
    cfg = config
    cfg.defrost()
    cfg.DATASET.DATA_PATH = args.path

    # For video
    config.TASK.MEASUREMENTS.append("TOP_DOWN_MAP")
    config.TASK.MEASUREMENTS.append("COLLISIONS")

    # Set to a high limit to allow replaying episodes with
    # number of steps greater than ObjectNav episode step
    # limit.
    cfg.ENVIRONMENT.MAX_EPISODE_STEPS = 5000

    # Set to a high limit to allow loading episodes with
    # number of steps greater than ObjectNav episode step
    # limit in the replay buffer.
    cfg.DATASET.MAX_REPLAY_STEPS = 5000
    cfg.freeze()

    run_reference_replay(
        cfg,
        num_episodes=args.num_episodes,
        output_prefix=args.output_prefix
    )


if __name__ == "__main__":
    main()

# python3 demo_replay.py --path habitat-challenge-data/data/datasets/objectnav/objectnav_mp3d_70k/train_1x1_seed7/train_1x1_seed7.json.gz --num-episodes 1 --output-prefix videos/demo/train_1x1_seed7
