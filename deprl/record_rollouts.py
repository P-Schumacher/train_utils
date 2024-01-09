"""Script used to play with trained agents."""

import argparse
import os
import time

import mujoco
import numpy as np
import yaml

from deprl import env_wrappers
from deprl.examine_rollouts import examine_rollout
from deprl.utils import PathRecorder, load_checkpoint
from deprl.vendor.tonic import logger

from .vendor.tonic import logger


def get_paths(path, checkpoint, checkpoint_file):
    """
    Checkpoints can be given as number e.g. <--checkpoint 1000000> or as file paths
    e.g. <--checkpoint_file path/checkpoints/step_1000000.pt'>
    This function handles this functionality.
    """
    if checkpoint_file is not None:
        path = checkpoint_file.split("checkpoints")[0]
        checkpoint = checkpoint_file.split("step_")[1].split(".")[0]
    checkpoint_path = os.path.join(path, "checkpoints")
    return path, checkpoint, checkpoint_path


def save_model_info_to_yaml(env, folder):
    # which joint type takes how many qpos fields
    qpos_type_size = [7, 4, 1, 1]
    qvel_type_size = [6, 3, 1, 1]
    file_path = folder
    info_dict = {}
    qpos_names = []
    qvel_names = []
    for jnt_idx in range(env.sim.model.njnt):
        jnt = env.sim.model.joint(jnt_idx)
        for _ in range(qpos_type_size[env.sim.model.joint(jnt_idx).type[0]]):
            qpos_names.append(jnt.name)
        for _ in range(qvel_type_size[env.sim.model.joint(jnt_idx).type[0]]):
            qvel_names.append(jnt.name)
    info_dict["qpos_names"] = qpos_names
    info_dict["qvel_names"] = qvel_names
    actuator_names = [
        env.sim.model.actuator(act_idx).name
        for act_idx in range(env.sim.model.nu)
    ]
    joint_ranges = []
    for jnt_idx in range(env.sim.model.njnt):
        if env.sim.model.joint(jnt_idx).type[0] != 0:
            joint_ranges.append(
                [
                    env.sim.model.joint(jnt_idx).range[0],
                    env.sim.model.joint(jnt_idx).range[1],
                ]
            )
    info_dict["joint_ranges_lower"] = list([float(x[0]) for x in joint_ranges])
    info_dict["joint_ranges_upper"] = list([float(x[1]) for x in joint_ranges])
    info_dict["actuator_names"] = actuator_names
    actuator_ranges = []
    for act_idx in range(env.sim.model.nu):
        actuator_ranges.append(
            [
                env.sim.model.actuator(act_idx).lengthrange[0],
                env.sim.model.actuator(act_idx).lengthrange[1],
            ]
        )
    info_dict["actuator_ranges_lower"] = list(
        [float(x[0]) for x in actuator_ranges]
    )
    info_dict["actuator_ranges_upper"] = list(
        [float(x[1]) for x in actuator_ranges]
    )
    info_dict["site_names"] = env.endeffector_sites
    limfrc_names = [
        env.sim.model.sensor(sens_idx).name
        for sens_idx in range(env.sim.model.nsensor)
        if "limfrc" in env.sim.model.sensor(sens_idx).name
    ]
    info_dict["limfrc_names"] = limfrc_names
    with open(os.path.join(file_path, "model_info.yaml"), "w") as yaml_file:
        yaml.dump(info_dict, yaml_file, default_flow_style=False)


def weaken_jetpack(env, actions):
    addresses = np.where(
        np.array(
            [
                env.sim.model.actuator(i).gaintype[0]
                for i in range(env.sim.model.nu)
            ]
        )
        == 0
    )[0]
    for adr in addresses:
        actions[adr] = np.clip(actions[adr], -0.05, 0.05)


def get_jetpack_cost(env, actions):
    addresses = np.where(
        np.array(
            [
                env.sim.model.actuator(i).gaintype[0]
                for i in range(env.sim.model.nu)
            ]
        )
        == 0
    )[0]
    return np.sum(np.abs(actions[addresses]))


def prepare_ee_target(env):
    # return env
    foot1 = env.episode_ref[:, -9:-6]
    foot2 = env.episode_ref[:, -12:-9]
    t = np.linspace(0, 6, foot1.shape[0])
    # foot1[:, :] = env.episode_ref[0, -9:-6]
    # foot2[:, :] = env.episode_ref[0, -12:-9]

    foot1[:, -3] = foot1[:, -3] + 0.5 * np.sin(t)
    env.episode_ref[:, -9:-6] = foot1
    env.episode_ref[:, -12:-9] = foot2


def remove_skin(env):
    geom_1_indices = np.where(env.sim.model.geom_group == 1)
    env.sim.model.geom_rgba[geom_1_indices, 3] = 0


def play_gym(
    agent,
    environment,
    num_episodes,
    noisy,
    render,
    save_video,
    playback,
    cam,
    target_tracking,
    no_print,
    terrain,
    spline,
):
    """Launches an agent in a Gym-based environment."""
    environment = env_wrappers.apply_wrapper(environment)
    observations = environment.reset()
    muscle_states = environment.muscle_states
    global_ref_max = 0
    global_ref_min = 1000
    global_min_reward = float("inf")
    global_max_reward = -float("inf")
    steps = 0
    episodes = 0
    tracking_error_cumul = []
    episode_lengths_cumul = []
    recorder = PathRecorder("rollouts")
    if render or save_video:
        remove_skin(environment)
    for ep in range(num_episodes):
        score = 0
        length = 0
        tracking_error = 0
        min_reward = float("inf")
        max_reward = -float("inf")
        if terrain:
            environment.terrain_curriculum.override(1.0)
        if spline:
            environment.spline_curriculum.override(1.0)
        observations = environment.reset()
        # prepare_ee_target(environment)
        total_fuel_cost_lin_xy = 0
        total_fuel_cost_lin_z = 0
        STOP_RECORDING = False
        while True:
            if not noisy:
                actions = agent.test_step(
                    observations, muscle_states=muscle_states, steps=1e6
                )
            else:
                actions = agent.noisy_test_step(
                    observations, muscle_states=muscle_states, steps=1e6
                )
            if len(actions.shape) > 1:
                actions = actions[0, :]
            weaken_jetpack(environment, actions)
            environment.get_ncons_hfield()
            # print(f'{ep=} {steps=} ncons_left={ncons[0]} ncons_right={ncons[1]}')
            if not playback:
                observations, reward, done, info = environment.step(actions)
                info["limfrc"] = environment.get_limfrc_sensor_values()
                info["grf"] = environment.get_grf_values()
            else:
                environment.playback_endeffector()
                reward, done = 0, 0
                info = dict(env_infos={})
            muscle_states = environment.muscle_states
            ref_error = environment.obs_dict["robot_error"].reshape(-1, 3)
            ref_error[:, 2] = ref_error[:, 2] * 0.5
            tracking_error += np.mean(np.square(ref_error))
            # total_fuel_cost += get_jetpack_cost(environment, actions)
            total_fuel_cost_lin_xy += np.sum(np.abs(actions[6:8]))
            total_fuel_cost_lin_z += np.sum(np.abs(actions[8]))
            if done or length >= 1000:
                STOP_RECORDING = 1
            info["tracking_error"] = (
                np.mean(np.square(ref_error)) if not STOP_RECORDING else np.nan
            )
            if render:
                # mujoco_render(environment)
                if target_tracking:
                    # ref_mot = environment.unwrapped.ref.get_reference(environment.unwrapped.time)
                    # environment.unwrapped.set_target_spheres(ref_mot.endeffector)
                    endeff = environment.unwrapped.episode_ref[
                        environment.unwrapped.steps
                    ]
                    # endef
                    environment.unwrapped.set_target_spheres(endeff)
                environment.unwrapped.mj_render()
                time.sleep(0.005)
            if save_video:
                ref_error = np.mean(
                    np.square(environment.unwrapped.obs_dict["robot_error"])
                )
                global_ref_max = max(global_ref_max, ref_error)
                global_ref_min = min(global_ref_min, ref_error)
                global_ref_max = 0.005
                environment.unwrapped.sim.model.geom("torso_geom_1").rgba[
                    1
                ] = 1 - (ref_error) / (global_ref_max)
                if target_tracking:
                    # ref_mot = environment.unwrapped.ref.get_reference(environment.unwrapped.time)
                    # environment.unwrapped.set_target_spheres(ref_mot.endeffector)
                    endeff = environment.unwrapped.episode_ref[
                        environment.unwrapped.steps
                    ]
                    environment.unwrapped.set_target_spheres(endeff)

                environment.sim.renderer.set_viewer_settings(
                    render_tendon=True, render_actuator=True
                )
                frame = environment.unwrapped.sim.renderer.render_offscreen(
                    camera_id=cam,
                    # height=1080,
                    # width=1920
                )
                environment.sim.renderer._scene_option.flags[
                    mujoco.mjtVisFlag.mjVIS_CONTACTFORCE
                ] = 1
                environment.sim.renderer._scene_option.flags[
                    mujoco.mjtVisFlag.mjVIS_CONTACTPOINT
                ] = 1
                recorder.store_frame(frame)

            steps += 1
            score += reward
            min_reward = min(min_reward, reward)
            max_reward = max(max_reward, reward)
            global_min_reward = min(global_min_reward, reward)
            global_max_reward = max(global_max_reward, reward)

            length += 1
            recorder.store(observations, actions, reward, info)
            done = 1 if environment.sim.data.site("head").xpos[2] < 0.5 else 0
            if done or length >= 1000:
                print(f"{total_fuel_cost_lin_xy=}")
                print(f"{total_fuel_cost_lin_z=}")
                tracking_error_cumul.append(tracking_error / length)
                episodes += 1
                if not no_print:
                    print()
                    print(f"Episodes: {episodes:,}")
                    print(f"Score: {score:,.3f}")
                    print(f"Length: {length:,}")
                    print(f"Terminal: {done:}")
                    print(f"Min reward: {min_reward:,.3f}")
                    print(f"Max reward: {max_reward:,.3f}")
                    print(f"Global min reward: {min_reward:,.3f}")
                    print(f"Global max reward: {max_reward:,.3f}")
                    print(f"tracking_error: {tracking_error_cumul[-1]:,.3f}")
                episode_lengths_cumul.append(length)
                recorder.end_of_rollout()
                break
    print(f"{np.mean(tracking_error_cumul)=}")
    print(f"{np.std(tracking_error_cumul)=}")
    print(f"{np.mean(episode_lengths_cumul)=}")
    print(f"{np.std(episode_lengths_cumul)=}")
    paths, folder = recorder.save(
        tracking_error=tracking_error_cumul,
        episode_length=episode_lengths_cumul,
    )
    environment.close()
    return paths, folder


def play(
    path,
    checkpoint,
    checkpoint_file,
    seed,
    header,
    agent,
    environment,
    num_episodes,
    noisy,
    render,
    save_video,
    playback,
    examine,
    cam,
    target_tracking,
    no_print,
    terrain,
    spline,
):
    """Reloads an agent and an environment from a previous experiment."""

    logger.log(f"Loading experiment from {path}")
    # Load config file and checkpoint path from folder
    path, checkpoint, checkpoint_path = get_paths(
        path, checkpoint, checkpoint_file
    )   
    config, checkpoint_path, _ = load_checkpoint(checkpoint_path, checkpoint)

    # Get important info from config
    header = header or config["tonic"]["header"]
    agent = agent or config["tonic"]["agent"]
    environment = environment or config["tonic"]["test_environment"]
    environment = environment or config["tonic"]["environment"]

    # Run the header first, e.g. to load an ML framework.
    if header:
        exec(header)

    # Build the agent.
    if not agent:
        raise ValueError("No agent specified.")
    agent = eval(agent)

    # Build the environment.
    env = environment.split(",")
    new_env = []
    for substring in env:
        if ")" in substring:
            if "episode_length" not in substring:
                new_env.append(
                    "reference_noise=0.0, episode_length=1000," + substring
                )
            else:
                new_env.append("reference_noise=0.0, episode_length=1000)")
                break
        elif "episode_length" not in substring:
            new_env.append(substring + ",")
        else:
            pass
    new_env = "".join(new_env)
    environment = eval(new_env)
    environment.seed(seed)

    # Adapt mpo specific settings
    if config and "mpo_args" in config:
        agent.set_params(**config["mpo_args"])
    if "env_args" in config:
        for k,v in config['env_args'].items():
            setattr(environment, k, v)
    # if config and "env_args" in config:
    #     environment.merge_args(config["env_args"])
    #     environment.apply_args()
    # Initialize the agent.
    agent.initialize(
        observation_space=environment.observation_space,
        action_space=environment.action_space,
        seed=seed,
    )

    # Load the weights of the agent form a checkpoint.
    if checkpoint_path:
        agent.load(checkpoint_path, only_checkpoint=True)
    paths, folder = play_gym(
        agent,
        environment,
        num_episodes,
        noisy,
        render,
        save_video,
        playback,
        cam,
        target_tracking,
        no_print,
        terrain,
        spline,
    )
    save_model_info_to_yaml(environment, folder)
    if examine:
        examine_rollout(paths, folder)


if __name__ == "__main__":
    # Argument parsing.
    parser = argparse.ArgumentParser()
    parser.add_argument("--path")
    parser.add_argument("--header")
    parser.add_argument("--agent")
    parser.add_argument("--environment", "--env")
    parser.add_argument("--checkpoint", default="last")
    parser.add_argument("--checkpoint_file", default=None)
    parser.add_argument("--cam", default="front_view")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--num_episodes", type=int, default=1)
    parser.add_argument("--render", action="store_true")
    parser.add_argument("--save_video", action="store_true")
    parser.add_argument("--target_tracking", action="store_true")
    parser.add_argument("--playback", action="store_true")
    parser.add_argument("--examine", action="store_true")
    parser.add_argument("--noisy", action="store_true")
    parser.add_argument("--no_print", action="store_true")
    parser.add_argument("--terrain", action="store_true")
    parser.add_argument("--spline", action="store_true")
    kwargs = vars(parser.parse_args())
    play(**kwargs)
