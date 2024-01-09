import argparse
import os

import numpy as np
import yaml
from matplotlib import pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages

from deprl.utils import load_paths


def get_model_info(save_folder):
    path = os.path.join(os.path.split(save_folder)[0], "model_info.yaml")
    with open(path, "r") as file:
        data = yaml.safe_load(file)
    return data


def get_vel_from_ref_name(ref_name):
    motion_file = ref_name
    vel_str = motion_file.split("IK")[0][-2:]
    vel = float(vel_str[0]) + float(vel_str[1]) * 0.1
    # transform kmh to ms
    vel = -vel * (1000 / 3600)
    return vel


def plot_muscle_lengths(page, paths, save_folder, model_info):
    nrows = paths[0]["actions"].shape[-1] - 6
    for path_id, path in enumerate(paths):
        ref_name = path["env_infos"]["active_ref"][0].decode("utf-8")
        fig, axs = plt.subplots(
            nrows // 8, 9, figsize=(32, nrows // 8 * 3.0), sharex=False
        )
        for ax_id, ax in enumerate(axs.flatten()):
            # try:
            ax.plot(
                path["env_infos"]["time"],
                path["env_infos"]["obs_dict"]["muscle_length"][:, ax_id],
            )
            ax.set_title(model_info["actuator_names"][ax_id])
            limits_lower = model_info["actuator_ranges_lower"][ax_id]
            limits_upper = model_info["actuator_ranges_upper"][ax_id]
            range_lim = limits_upper - limits_lower
            ax.set_ylim(
                [
                    limits_lower - 0.1 * range_lim,
                    limits_upper + 0.1 * range_lim,
                ]
            )
            ax.hlines(
                limits_lower,
                path["env_infos"]["time"][-1],
                path["env_infos"]["time"][0],
                colors="black",
                linestyles="--",
            )
            ax.hlines(
                limits_upper,
                path["env_infos"]["time"][-1],
                path["env_infos"]["time"][0],
                colors="black",
                linestyles="--",
            )
            if ax_id == len(model_info["actuator_names"]) - 1:
                break
        for ax_id in range(axs.shape[0]):
            axs[ax_id, 0].set_ylabel("lce")
        for ax_id in range(axs.shape[1]):
            axs[-1, ax_id].set_xlabel("time")
        # axs[-1, -1].legend()
        fig.suptitle(f"muscle lengths ep {path_id} {ref_name}", fontsize=18)
        plt.tight_layout()
        fig.subplots_adjust(top=0.95, hspace=0.5, wspace=0.2)
        page.savefig()
        plt.close()


def plot_muscle_velocities(page, paths, save_folder, model_info):
    nrows = paths[0]["actions"].shape[-1] - 6
    for path_id, path in enumerate(paths):
        ref_name = path["env_infos"]["active_ref"][0].decode("utf-8")
        fig, axs = plt.subplots(
            nrows // 8, 9, figsize=(32, nrows // 8 * 3.0), sharex=False
        )
        for ax_id, ax in enumerate(axs.flatten()):
            try:
                ax.plot(
                    path["env_infos"]["time"],
                    path["env_infos"]["obs_dict"]["muscle_velocity"][:, ax_id],
                )
                ax.set_title(model_info["actuator_names"][ax_id])
            except Exception:
                pass
        for ax_id in range(axs.shape[0]):
            axs[ax_id, 0].set_ylabel("lce")
        for ax_id in range(axs.shape[1]):
            axs[-1, ax_id].set_xlabel("time")
        # axs[-1, -1].legend()
        fig.suptitle(f"muscle vels ep {path_id} {ref_name}", fontsize=18)
        plt.tight_layout()
        fig.subplots_adjust(top=0.95, hspace=0.5, wspace=0.2)
        page.savefig()
        plt.close()


def plot_muscle_forces(page, paths, save_folder, model_info):
    nrows = paths[0]["actions"].shape[-1] - 6
    for path_id, path in enumerate(paths):
        ref_name = path["env_infos"]["active_ref"][0].decode("utf-8")
        fig, axs = plt.subplots(
            nrows // 8, 9, figsize=(32, nrows // 8 * 3.0), sharex=False
        )
        for ax_id, ax in enumerate(axs.flatten()):
            try:
                ax.plot(
                    path["env_infos"]["time"],
                    path["env_infos"]["obs_dict"]["muscle_force"][:, ax_id],
                )
                ax.set_title(model_info["actuator_names"][ax_id])
            except Exception:
                pass
        for ax_id in range(axs.shape[0]):
            axs[ax_id, 0].set_ylabel("lce")
        for ax_id in range(axs.shape[1]):
            axs[-1, ax_id].set_xlabel("time")
        # axs[-1, -1].legend()
        fig.suptitle(f"muscle forces ep {path_id} {ref_name}", fontsize=18)
        plt.tight_layout()
        fig.subplots_adjust(top=0.95, hspace=0.5, wspace=0.2)
        page.savefig()
        plt.close()


def plot_actions(page, paths, save_folder, model_info):
    nrows = paths[0]["actions"].shape[-1] - 6
    for path_id, path in enumerate(paths):
        ref_name = path["env_infos"]["active_ref"][0].decode("utf-8")
        fig, axs = plt.subplots(
            nrows // 8,
            9,
            figsize=(24, nrows // 8 * 2.0),
            sharex=False,
            sharey=False,
        )
        for ax_id, ax in enumerate(axs.flatten()):
            try:
                ax.plot(
                    path["env_infos"]["time"],
                    path["env_infos"]["obs_dict"]["act"][:, ax_id],
                    label="activation",
                )
                ax.plot(
                    path["env_infos"]["time"],
                    path["actions"][:, ax_id],
                    label="excitation",
                )
                ax.set_ylabel("act/exc")
                ax.set_xlabel("time")
                # ax.set_xlim([0, 1])
                ax.set_title(model_info["actuator_names"][ax_id])
            except Exception:
                pass
        axs[-1, -1].legend()
        fig.suptitle(f"muscle activity ep {path_id} {ref_name}", fontsize=18)
        plt.tight_layout()
        fig.subplots_adjust(top=0.95, hspace=0.5, wspace=0.2)
        page.savefig()
        plt.close()


def plot_qpos(page, paths, save_folder, model_info):
    nrows = paths[0]["env_infos"]["obs_dict"]["qpos"].shape[-1] - 6
    for path_id, path in enumerate(paths):
        ref_name = path["env_infos"]["active_ref"][0].decode("utf-8")
        fig, axs = plt.subplots(
            nrows // 8,
            9,
            figsize=(40, nrows // 8 * 3.0),
            sharex=False,
            sharey=False,
        )
        for ax_id, ax in enumerate(axs.flatten()):
            try:
                ax.plot(
                    path["env_infos"]["time"],
                    path["env_infos"]["obs_dict"]["qpos"][:, ax_id],
                )
                # ax.set_xlim([path["env_infos"]["time"][0], path["env_infos"]["time"][1]])
                ax.set_ylabel("qpos val")
                ax.set_xlabel("time")
                ax.set_title(model_info["qpos_names"][ax_id])
                if model_info["qpos_names"][ax_id] != "root":
                    limits_lower = model_info["joint_ranges_lower"][ax_id - 7]
                    limits_upper = model_info["joint_ranges_upper"][ax_id - 7]
                    range_lim = limits_upper - limits_lower
                    ax.set_ylim(
                        [
                            limits_lower - 0.1 * range_lim,
                            limits_upper + 0.1 * range_lim,
                        ]
                    )
                    ax.hlines(
                        limits_lower,
                        path["env_infos"]["time"][-1],
                        path["env_infos"]["time"][0],
                        colors="black",
                        linestyles="--",
                    )
                    ax.hlines(
                        limits_upper,
                        path["env_infos"]["time"][-1],
                        path["env_infos"]["time"][0],
                        colors="black",
                        linestyles="--",
                    )
            except Exception:
                pass
        fig.suptitle(f"qpos ep {path_id} {ref_name}", fontsize=18)
        plt.tight_layout()
        page.savefig()
        plt.close()


def plot_qvel(page, paths, save_folder, model_info):
    nrows = paths[0]["env_infos"]["obs_dict"]["qvel"].shape[-1] - 6
    for path_id, path in enumerate(paths):
        ref_name = path["env_infos"]["active_ref"][0].decode("utf-8")
        fig, axs = plt.subplots(
            nrows // 8,
            9,
            figsize=(40, nrows // 8 * 3.0),
            sharex=False,
            sharey=False,
        )
        for ax_id, ax in enumerate(axs.flatten()):
            try:
                ax.plot(
                    path["env_infos"]["time"],
                    path["env_infos"]["obs_dict"]["qvel"][:, ax_id],
                )
                ax.set_ylabel("qvel val")
                ax.set_title(model_info["qvel_names"][ax_id])
                ax.set_xlabel("time")
            except Exception:
                pass
        fig.suptitle(f"qvel ep {path_id} {ref_name}", fontsize=18)
        plt.tight_layout()
        page.savefig()
        plt.close()


def plot_muscle_activity_image(page, paths, save_folder, model_info):
    paths[0]["actions"].shape[-1] - 6
    for path_id, path in enumerate(paths):
        ref_name = path["env_infos"]["active_ref"][0].decode("utf-8")
        fig, ax = plt.subplots(1, 1, figsize=(40, 10))
        cax = ax.imshow(
            path["actions"][:, :].T,
        )
        ax.set_xlabel("time")
        ax.set_ylabel("muscle act.")
        plt.colorbar(cax)
        fig.suptitle(f"muscle activity ep {path_id} {ref_name}", fontsize=18)
        plt.tight_layout()
        page.savefig()
        plt.close()


def plot_dict(page, paths, save_folder, dict_key):
    os.makedirs(os.path.join(save_folder, dict_key), exist_ok=True)
    for path_id, path in enumerate(paths):
        for key in path["env_infos"][dict_key].keys():
            if not key == "height" and not key == "feet_rel_positions":
                shape = path["env_infos"][dict_key][key].shape
                nrows = shape[-1] if len(shape) > 1 else 1
                if nrows == 1:
                    fig, axs = plt.subplots(1, 1, figsize=(5, 1))
                    axs.plot(path["env_infos"][dict_key][key])
                else:
                    fig, axs = plt.subplots(nrows, 1, figsize=(5, nrows))
                    nrows * 1.2
                    ax_idx = 0
                    for state_id, state in enumerate(
                        path["env_infos"][dict_key][key].T
                    ):
                        axs.flatten()[ax_idx].plot(state)
                        axs.flatten()[ax_idx].set_title(f"{key}_{state_id}")
                        ax_idx += 1
                page.savefig()
                plt.close()


def plot_length_histogram(page, paths, save_folder, dict_key):
    lengths = [x["env_infos"]["time"].shape[0] for x in paths]
    fig, axs = plt.subplots(1, 1, figsize=(2, 2))
    axs.hist(lengths, bins=10)
    plt.tight_layout()
    page.savefig()(os.path.join(save_folder, "histogram.png"))
    plt.close()
    mean = np.mean(lengths)
    std = np.std(lengths)
    print("Episode lengths:")
    print(f"{mean=}")
    print(f"{std=}")
    errors = []
    for p in paths:
        tracking_error = p["env_infos"]["tracking_error"]
        non_nan_indices = np.where(1 - np.isnan(tracking_error))[0]
        errors.append(
            np.sum(tracking_error[non_nan_indices]) / len(non_nan_indices)
        )
    print(f"{np.mean(errors)=}")


def plot_reference_error_vel(page, paths, save_folder):
    if "vel" in paths[0]["env_infos"]["obs_dict"]:
        # nrows = paths[0]["env_infos"]["obs_dict"]["target_speed"].shape[1]
        for path_id, path in enumerate(paths):
            fig, axs = plt.subplots(2, 1, figsize=(4, 2.0))
            # axs = [axs]
            for ax_id, ax in enumerate(axs):
                ax.plot(
                    path["env_infos"]["time"],
                    # path["env_infos"]["obs_dict"]["vel"][:, ax_id],
                    path["env_infos"]["obs_dict"]["vel"][:, ax_id],
                    # path["env_infos"]["obs_dict"]["vel"][:, ax_id],
                    # path["env_infos"]["obs_dict"]["vel"],
                    label="achieved",
                )
                ax.plot(
                    path["env_infos"]["time"],
                    path["env_infos"]["obs_dict"]["target_speed"][:, 0, ax_id],
                    # path["env_infos"]["obs_dict"]["target_speed"][:,0],
                    label="desired",
                )
                # ax.set_ylim([-1.5,0])
                # ax.set_title()
            axs[0].legend()
            plt.tight_layout()
            page.savefig()
            plt.close()


def plot_goal_traj(page, paths, save_folder, model_info):
    os.makedirs(os.path.join(save_folder, "goal_traj"), exist_ok=True)
    len(paths) if len(paths) > 1 else 2
    for path_id, path in enumerate(paths):
        fig, axs = plt.subplots(
            len(model_info["site_names"]),
            3,
            figsize=(20, len(model_info["site_names"]) * 4 * 1.2),
        )
        ref_name = path["env_infos"]["active_ref"][0].decode("utf-8")
        robot_error = path["env_infos"]["obs_dict"]["robot_error"]
        current_state = path["env_infos"]["obs_dict"]["current_state"]
        target_state = robot_error + current_state
        time = path["env_infos"]["time"]
        name_ptr = 0
        for ax_idx, ax in enumerate(axs.flatten()):
            if ax_idx % 3 == 0:
                ax.set_title(
                    model_info["site_names"][name_ptr] + "_x", fontsize=20
                )
            elif ax_idx % 3 == 1:
                ax.set_title(
                    model_info["site_names"][name_ptr] + "_y", fontsize=20
                )
            else:
                ax.set_title(
                    model_info["site_names"][name_ptr] + "_z", fontsize=20
                )
                name_ptr += 1
            ax.plot(time, current_state[:, ax_idx], label="current")
            ax.plot(time, target_state[:, ax_idx], label="target")
            fig.suptitle(
                f" tracking error ep {path_id} {ref_name}", fontsize=18
            )
            ax.legend()

        fig.tight_layout()
        fig.subplots_adjust(top=0.95, hspace=0.5, wspace=0.2)
        page.savefig(fig)
        plt.close()


def plot_goal_traj_adjusted(page, paths, save_folder, model_info):
    os.makedirs(os.path.join(save_folder, "goal_traj"), exist_ok=True)
    save = False
    for path_id, path in enumerate(paths):
        fig, axs = plt.subplots(
            len(model_info["site_names"]),
            3,
            figsize=(20, len(model_info["site_names"]) * 4 * 1.2),
        )
        ref_name = path["env_infos"]["active_ref"][0].decode("utf-8")
        if "walk" in ref_name or "run" in ref_name:
            episode_length = path["actions"].shape[0]
            robot_error = path["env_infos"]["obs_dict"]["robot_error"]
            current_state = path["env_infos"]["obs_dict"]["current_state"]
            target_state = robot_error + current_state
            time = path["env_infos"]["time"]
            name_ptr = 0
            for ax_idx, ax in enumerate(axs.flatten()):
                if ax_idx % 3 == 0:
                    ax.set_title(
                        model_info["site_names"][name_ptr] + "_x", fontsize=20
                    )
                elif ax_idx % 3 == 1:
                    ax.set_title(
                        model_info["site_names"][name_ptr] + "_y", fontsize=20
                    )
                else:
                    ax.set_title(
                        model_info["site_names"][name_ptr] + "_z", fontsize=20
                    )
                    name_ptr += 1
                if ax_idx % 3 == 1:
                    vel = get_vel_from_ref_name(ref_name)
                    current_state[:, ax_idx] = current_state[
                        :, ax_idx
                    ] - np.linspace(
                        0, +0.01 * vel * episode_length, episode_length
                    )
                    target_state[:, ax_idx] = target_state[
                        :, ax_idx
                    ] - np.linspace(
                        0, +0.01 * vel * episode_length, episode_length
                    )
                ax.plot(time, current_state[:, ax_idx], label="current")
                ax.plot(time, target_state[:, ax_idx], label="target")
                fig.suptitle(
                    f" vel adjusted-tracking error ep {path_id} {ref_name}",
                    fontsize=18,
                )
                ax.legend()
                save = True
        fig.tight_layout()
        fig.subplots_adjust(top=0.95, hspace=0.5, wspace=0.2)
        if save:
            page.savefig(fig)
        plt.close()


def plot_grfs(page, paths, save_folder, model_info):
    for path_id, path in enumerate(paths):
        ref_name = path["env_infos"]["active_ref"][0].decode("utf-8")
        contacts = path["env_infos"]["grf"]
        fig, axs = plt.subplots(
            2, 6, figsize=(40, 10), sharex=False, sharey=False
        )
        for ax_id in range(6):
            axs[0, ax_id].plot(
                path["env_infos"]["time"], contacts[:, 0, ax_id]
            )
            axs[0, ax_id].set_title(f"left foot: {ax_id}")
            axs[1, ax_id].plot(
                path["env_infos"]["time"], contacts[:, 0, ax_id]
            )
            axs[1, ax_id].set_title(f"right foot: {ax_id}")
            axs[-1, ax_id].set_xlabel("time")
        axs[0, 0].set_ylabel("force")
        axs[1, 0].set_ylabel("force")
        fig.suptitle(f"contact force ep {path_id} {ref_name}", fontsize=18)
        plt.tight_layout()
        page.savefig()
        plt.close()


def plot_limfrc(page, paths, save_folder, model_info):
    sensor_names = model_info["limfrc_names"]
    nrows = len(sensor_names) // 2
    for path_id, path in enumerate(paths):
        ref_name = path["env_infos"]["active_ref"][0].decode("utf-8")
        fig, axs = plt.subplots(
            2, nrows, figsize=(40, 10), sharex=False, sharey=False
        )
        for ax_id, ax in enumerate(axs.flatten()):
            ax.plot(
                path["env_infos"]["time"],
                path["env_infos"]["limfrc"][:, ax_id],
            )
            ax.set_ylabel("limfrc")
            ax.set_xlabel("time")
            ax.set_title(sensor_names[ax_id])
        fig.suptitle(f"Limit force sensors {path_id} {ref_name}", fontsize=18)
        plt.tight_layout()
        page.savefig()
        plt.close()


def examine_rollout(paths, save_folder):
    model_info = get_model_info(save_folder)
    pdf_filename = os.path.join(save_folder, "data_plot.pdf")
    with PdfPages(pdf_filename) as page:
        # plot_muscle_activity_image(page, paths, save_folder, model_info)
        plot_muscle_lengths(page, paths, save_folder, model_info)
        plot_muscle_forces(page, paths, save_folder, model_info)
        plot_qpos(page, paths, save_folder, model_info)
        plot_actions(page, paths, save_folder, model_info)
        plot_qvel(page, paths, save_folder, model_info)
        plot_grfs(page, paths, save_folder, model_info)
        plot_limfrc(page, paths, save_folder, model_info)
        plot_goal_traj(page, paths, save_folder, model_info)
        plot_goal_traj_adjusted(page, paths, save_folder, model_info)

        # plot_length_histogram(page, paths, save_folder, "obs_dict")
        # plot_reference_error(page, paths, save_folder)
        # actions
        # observations and rewards
        # plot_dict(page, paths, save_folder, "obs_dict")
        # plot_dict(page, paths, save_folder, "rwd_dict")

        # reference_error
        # plot_reference_error_vel(page, paths, save_folder)
        # plot_goal_traj(page, paths, save_folder)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--path", type=str)
    args = vars(parser.parse_args())

    save_folder = f'{args["path"][:-3]}'
    os.makedirs(save_folder, exist_ok=True)
    paths = load_paths(args["path"])
    examine_rollout(paths, save_folder)
