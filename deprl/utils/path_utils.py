import os
from datetime import datetime

import h5py
import numpy as np
import skvideo.io
from myosuite.utils.tensor_utils import stack_tensor_dict_list

from deprl.vendor.tonic import logger

# if "MYOSUITE" in os.environ:
#     print("MYOSUITE")
#     from myosuite.utils.tensor_utils import stack_tensor_dict_list
# else:
#     print("ROBOHIVE")
#     from robohive.utils.tensor_utils import stack_tensor_dict_list


def save_dict_to_hdf5(hf, data_dict, group_name=""):
    for key, value in data_dict.items():
        if isinstance(value, dict):
            # If the value is a dictionary, create a new group and save recursively
            subgroup_name = f"{group_name}/{key}"
            hf.create_group(subgroup_name)
            save_dict_to_hdf5(hf, value, subgroup_name)
        else:
            # Convert object/string arrays to fixed-length strings before saving
            if value.dtype == "O":
                value = np.array(value, dtype="S")
            hf[group_name].create_dataset(key, data=value)


def load_dict_from_hdf5(hf, group_name=""):
    data_dict = {}
    group = hf[group_name] if group_name else hf
    for key in group.keys():
        item = group[key]
        if isinstance(item, h5py.Group):
            # If the item is a group, load recursively
            data_dict[key] = load_dict_from_hdf5(hf, f"{group_name}/{key}")
        else:
            data_dict[key] = np.array(item)
    return data_dict


def load_paths(filename):
    paths = []
    with h5py.File(filename, "r") as hf:
        for group_name in hf.keys():
            path = load_dict_from_hdf5(hf, group_name)
            paths.append(path)
    return paths


def get_current_date_time():
    now = datetime.now()
    formatted_date_time = now.strftime("%Y%m%d_%H%M%S")
    return formatted_date_time


class PathRecorder:
    def __init__(self, file_name):
        self.start_time = get_current_date_time()
        self.file_name = f"{file_name}_{self.start_time}"
        self.folder = os.path.join("./output/", self.file_name)
        self.full_reset()

    def store(self, observations, actions, rewards, env_infos):
        self.observations.append(observations)
        self.actions.append(actions)
        self.rewards.append(rewards)
        self.env_infos.append(env_infos)

    def end_of_rollout(self):
        path = dict(
            observations=np.array(self.observations),
            actions=np.array(self.actions),
            rewards=np.array(self.rewards),
            env_infos=stack_tensor_dict_list(self.env_infos),
        )
        self.paths.append(path)
        self.episode_reset()

    def episode_reset(self):
        self.observations = []
        self.actions = []
        self.rewards = []
        self.env_infos = []

    def full_reset(self):
        self.paths = []
        self.frames = []
        self.stored_frames = False
        self.episode_reset()

    def store_frame(self, frame):
        self.frames.append(frame)
        self.stored_frames = True

    def save(self, **kwargs):
        os.makedirs(self.folder, exist_ok=True)
        file_path = os.path.join(self.folder, f"rollout_{self.start_time}.h5")
        with h5py.File(file_path, "w") as hf:
            for idx, data_dict in enumerate(self.paths):
                group_name = f"dict_{idx}"
                hf.create_group(group_name)
                save_dict_to_hdf5(hf, data_dict, group_name)
        if self.stored_frames and len(self.frames) > 0:
            self.save_video()
        elif self.stored_frames and len(self.frames) == 0:
            logger.log("Attempted to save video with 0 frames. Skipping.")
        else:
            pass
        self.write_stats_to_file(file_path, **kwargs)
        return self.paths, self.folder

    def write_stats_to_file(self, file_path, **kwargs):
        with open(
            os.path.join(
                os.path.join(*file_path.split(os.sep)[:-1]), "stats.txt"
            ),
            "w",
        ) as file:
            for key, value in kwargs.items():
                if isinstance(value, list):
                    value = np.array(value)  # Convert list to NumPy array for
                    # easy mean and std calculations
                    mean = np.mean(value)
                    std = np.std(value)
                    file.write(f"{key}: Mean = {mean}, Std = {std}\n")

    def save_video(self):
        if len(self.frames) != 0:
            # normal_speed
            skvideo.io.vwrite(
                os.path.join(self.folder, f"rollout_{self.start_time}.mp4"),
                np.asarray(self.frames),
                outputdict={"-pix_fmt": "yuv420p"},
                inputdict={"-r": "100"},
                # inputdict={"-r": "40"},
            )
            # slow-mo
            skvideo.io.vwrite(
                os.path.join(
                    self.folder, f"rollout_{self.start_time}_slowmo.mp4"
                ),
                np.asarray(self.frames),
                outputdict={"-pix_fmt": "yuv420p"},
                inputdict={"-r": "50"},
                # inputdict={"-r": "40"},
            )
