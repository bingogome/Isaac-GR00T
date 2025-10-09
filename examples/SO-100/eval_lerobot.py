# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""
This is the new Gr00T policy eval script with so100, so101 robot arm. Based on:
https://github.com/huggingface/lerobot/pull/777

Example command:

```shell

python eval_gr00t_so100.py \
    --robot.type=so100_follower \
    --robot.port=/dev/ttyACM0 \
    --robot.id=lil_guy \
    --robot.cameras="{ wrist: {type: opencv, index_or_path: 9, width: 640, height: 480, fps: 30}, front: {type: opencv, index_or_path: 15, width: 640, height: 480, fps: 30}}" \
    --policy_host=10.112.209.136 \
    --lang_instruction="Grab markers and place into pen holder."
```


First replay to ensure the robot is working:
```shell
python -m lerobot.replay \
    --robot.type=so100_follower \
    --robot.port=/dev/ttyACM0 \
    --robot.id=lil_guy \
    --dataset.repo_id=youliangtan/so100-table-cleanup \
    --dataset.episode=2
```
"""

import json
import logging
import time
from dataclasses import asdict, dataclass
from pathlib import Path
from pprint import pformat
from typing import Any

import draccus
import matplotlib.pyplot as plt
import numpy as np
from lerobot.cameras.opencv.configuration_opencv import (  # noqa: F401
    OpenCVCameraConfig,
)
from lerobot.robots import (  # noqa: F401
    Robot,
    RobotConfig,
    koch_follower,
    make_robot_from_config,
    so100_follower,
    so101_follower,
    bi_so101_follower
)
from lerobot.utils.utils import (
    init_logging,
    log_say,
)

# NOTE:
# Sometimes we would like to abstract different env, or run this on a separate machine
# User can just move this single python class method gr00t/eval/service.py
# to their code or do the following line below
# sys.path.append(os.path.expanduser("~/Isaac-GR00T/gr00t/eval/"))
# from service import ExternalRobotInferenceClient

from gr00t.eval.service import ExternalRobotInferenceClient

#################################################################################


def _sorted_slices(modality_section: dict[str, Any]) -> list[tuple[str, slice]]:
    if modality_section is None:
        return []
    sorted_modalities = sorted(
        modality_section.items(), key=lambda item: item[1].get("start", 0)
    )
    return [
        (name, slice(entry["start"], entry["end"])) for name, entry in sorted_modalities
    ]


class Gr00tRobotInferenceClient:
    """The exact keys used is defined in modality.json.

    Provide a modality.json that matches the robot configuration to map
    observation and action tensors to the lerobot motor keys.
    """

    def __init__(
        self,
        host="localhost",
        port=5555,
        camera_keys: list[str] | None = None,
        robot_state_keys: list[str] | None = None,
        modality_config: dict[str, Any] | None = None,
        show_images=False,
    ):
        self.policy = ExternalRobotInferenceClient(host=host, port=port)
        self.camera_keys = camera_keys or []
        self.robot_state_keys = list(robot_state_keys or [])
        if not modality_config:
            raise ValueError("A modality configuration is required for policy inference")
        self.state_modalities = _sorted_slices(modality_config.get("state"))
        self.action_modalities = _sorted_slices(modality_config.get("action"))
        self.modality_keys = [name for name, _ in self.action_modalities]
        total_expected = max((sl.stop for _, sl in self.action_modalities), default=0)
        if len(self.robot_state_keys) < total_expected:
            raise ValueError(
                "robot_state_keys smaller than modality definition. "
                "Check modality configuration matches robot."
            )
        self._action_key_mapping = {
            name: self.robot_state_keys[slice_obj] for name, slice_obj in self.action_modalities
        }
        self.show_images = show_images
        if not self.state_modalities:
            raise ValueError("State modalities missing from modality configuration")

    def get_action(self, observation_dict, lang: str):
        # first add the images
        obs_dict = {f"video.{key}": observation_dict[key] for key in self.camera_keys}

        # show images
        if self.show_images:
            view_img(obs_dict)

        # Make all single float value of dict[str, float] state into a single array
        state = np.array([observation_dict[k] for k in self.robot_state_keys])
        for name, slice_obj in self.state_modalities:
            obs_dict[f"state.{name}"] = state[slice_obj].astype(np.float64)
        obs_dict["annotation.human.task_description"] = lang

        # then add a dummy dimension of np.array([1, ...]) to all the keys (assume history is 1)
        for k in obs_dict:
            if isinstance(obs_dict[k], np.ndarray):
                obs_dict[k] = obs_dict[k][np.newaxis, ...]
            else:
                obs_dict[k] = [obs_dict[k]]

        # get the action chunk via the policy server
        # Example of obs_dict for single camera task:
        # obs_dict = {
        #     "video.front": np.zeros((1, 480, 640, 3), dtype=np.uint8),
        #     "video.wrist": np.zeros((1, 480, 640, 3), dtype=np.uint8),
        #     "state.single_arm": np.zeros((1, 5)),
        #     "state.gripper": np.zeros((1, 1)),
        #     "annotation.human.action.task_description": [self.language_instruction],
        # }
        action_chunk = self.policy.get_action(obs_dict)

        # convert the action chunk to a list of dict[str, float]
        lerobot_actions = []
        action_horizon = action_chunk[f"action.{self.modality_keys[0]}"].shape[0]
        for i in range(action_horizon):
            action_dict = self._convert_to_lerobot_action(action_chunk, i)
            lerobot_actions.append(action_dict)
        return lerobot_actions

    def _convert_to_lerobot_action(
        self, action_chunk: dict[str, np.array], idx: int
    ) -> dict[str, float]:
        """
        This is a magic function that converts the action chunk to a dict[str, float]
        This is because the action chunk is a dict[str, np.array]
        and we want to convert it to a dict[str, float]
        so that we can send it to the robot
        """
        action_dict: dict[str, float] = {}
        for name in self.modality_keys:
            action_values = np.atleast_1d(action_chunk[f"action.{name}"][idx])
            joint_keys = self._action_key_mapping[name]
            if len(action_values) != len(joint_keys):
                raise ValueError(
                    f"Action modality '{name}' expected {len(joint_keys)} values, "
                    f"but received {len(action_values)}"
                )
            for joint_key, joint_value in zip(joint_keys, action_values):
                action_dict[joint_key] = float(joint_value)
        return action_dict


#################################################################################


def view_img(img, overlay_img=None):
    """
    This is a matplotlib viewer since cv2.imshow can be flaky in lerobot env
    """
    if isinstance(img, dict):
        # stack the images horizontally
        img = np.concatenate([img[k] for k in img], axis=1)

    plt.imshow(img)
    plt.title("Camera View")
    plt.axis("off")
    plt.pause(0.001)  # Non-blocking show
    plt.clf()  # Clear the figure for the next frame


def print_yellow(text):
    print("\033[93m {}\033[00m".format(text))


DEFAULT_MODALITY_CONFIGS = {
    "so100_follower": Path(__file__).with_name("so100__modality.json"),
    "so101_follower": Path(__file__).with_name("so100__modality.json"),
    "bi_so101_follower": Path(__file__).with_name("bi_so101_modality.json"),
}


@dataclass
class EvalConfig:
    robot: RobotConfig  # the robot to use
    policy_host: str = "localhost"  # host of the gr00t server
    policy_port: int = 5555  # port of the gr00t server
    action_horizon: int = 8  # number of actions to execute from the action chunk
    lang_instruction: str = "Grab pens and place into pen holder."
    play_sounds: bool = False  # whether to play sounds
    timeout: int = 60  # timeout in seconds
    show_images: bool = False  # whether to show images
    modality_config_path: str | None = None  # path to modality configuration json


@draccus.wrap()
def eval(cfg: EvalConfig):
    init_logging()
    logging.info(pformat(asdict(cfg)))

    # Step 1: Initialize the robot
    robot = make_robot_from_config(cfg.robot)
    robot.connect()

    # get camera keys from RobotConfig
    camera_keys = list(cfg.robot.cameras.keys())
    print("camera_keys: ", camera_keys)

    log_say("Initializing robot", cfg.play_sounds, blocking=True)

    language_instruction = cfg.lang_instruction

    # NOTE: for so100/so101, this should be:
    # ['shoulder_pan.pos', 'shoulder_lift.pos', 'elbow_flex.pos', 'wrist_flex.pos', 'wrist_roll.pos', 'gripper.pos']
    robot_state_keys = list(robot._motors_ft.keys())
    print("robot_state_keys: ", robot_state_keys)

    # load modality configuration matching the robot
    modality_path = (
        Path(cfg.modality_config_path)
        if cfg.modality_config_path
        else DEFAULT_MODALITY_CONFIGS.get(cfg.robot.type)
    )
    if modality_path is None:
        raise ValueError(
            "Could not determine modality configuration. Provide --modality-config-path."
        )
    with open(modality_path, "r", encoding="utf-8") as f:
        modality_config = json.load(f)

    # Step 2: Initialize the policy
    policy = Gr00tRobotInferenceClient(
        host=cfg.policy_host,
        port=cfg.policy_port,
        camera_keys=camera_keys,
        robot_state_keys=robot_state_keys,
        modality_config=modality_config,
        show_images=cfg.show_images,
    )
    log_say(
        "Initializing policy client with language instruction: " + language_instruction,
        cfg.play_sounds,
        blocking=True,
    )

    # Step 3: Run the Eval Loop
    while True:
        # get the realtime image
        observation_dict = robot.get_observation()
        print("observation_dict", observation_dict.keys())
        action_chunk = policy.get_action(observation_dict, language_instruction)

        for i in range(cfg.action_horizon):
            action_dict = action_chunk[i]
            print("action_dict", action_dict.keys())
            robot.send_action(action_dict)
            time.sleep(0.02)  # Implicitly wait for the action to be executed


if __name__ == "__main__":
    eval()
