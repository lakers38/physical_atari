# Copyright 2025 Keen Technologies, Inc.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import datetime
import json
import logging
import multiprocessing
import os
import struct
import time

import numpy as np
import torch
from PIL import Image

from framework.Logger import add_file_handler_to_logger, logger


def main(args):
    logger.setLevel(getattr(logging, args.log_level))

    experiment_name = os.path.splitext(os.path.basename(args.game_config))[0]
    experiment_name = f"{experiment_name}_{datetime.datetime.now().strftime('%Y%m%d_%H%M%S')}"
    data_dir = os.path.join(args.results_dir, experiment_name)
    os.makedirs(data_dir, exist_ok=True)

    add_file_handler_to_logger(os.path.join(data_dir, experiment_name + '.log'))

    logger.info(f"Importing agent: {args.agent_type}")
    if args.agent_type == 'agent_delay_target':
        from agent_delay_target import Agent
    elif args.agent_type == 'agent_dqn':
        from agent_dqn import Agent
    elif args.agent_type == 'agent_rainbow':
        from agent_rainbow import Agent
    elif args.agent_type == 'agent_random':
        from agent_random import Agent
    else:
        raise ValueError(f"Invalid agent type={args.agent_type}")

    dev = f"cuda:{args.gpu}"
    score_detector_type = args.score_detector_type
    total_frames = args.total_frames

    atari_height = 210
    atari_width = 160
    # REVIEW: support both rgb and yuyv/grayscale
    obs_dims = (atari_width, atari_height, 3)

    lives_as_episodes = args.lives_as_episode
    logger.info(f"args={args}")
    logger.info(f"{args.agent_type}")
    logger.info(f"lives_as_episodes={lives_as_episodes}")
    # "Revisiting the ALE" recommends a max episode frames (60fps) of 18_000, which is only five minutes, which would cut short many
    # valid high performing games.
    # "Is Deep Reinforcement Learning Really Superhuman on Atari?" https://arxiv.org/pdf/1908.04683 recommends 18k limit without a reward.
    max_frames_without_reward = 18_000

    try:
        if args.use_gui > 0:
            import traceback

            from gui_physical import SharedFrameData, create_gui_process

            def process_wrapper(func, logger):
                def wrap(*args, **kwargs):
                    try:
                        func(*args, **kwargs)
                    except Exception:
                        traceback.print_exc()
                        logger.error("Exception in process", exc_info=True)

                return wrap

            shared_lock = multiprocessing.Lock()
            shared_data = SharedFrameData(obs_dim=obs_dims, lock=shared_lock)

            # used for infrequent, episode-related stats that should not be dropped
            episode_queue = multiprocessing.Queue()
            configure_event = multiprocessing.Event() if args.use_gui == 2 else None
            exit_event = multiprocessing.Event()

            gui_process = multiprocessing.Process(
                target=process_wrapper(create_gui_process, logger),
                args=(
                    args.game_config,
                    args.joystick_config,
                    args.camera_config,
                    args.detection_config,
                    args.score_detector_type,
                    obs_dims,
                    episode_queue,
                    shared_data,
                    configure_event,
                    exit_event,
                ),
            )
            gui_process.daemon = True
            gui_process.start()

            if configure_event is not None:
                logger.info("Waiting for configuration...")
                try:
                    configure_event.wait()
                except KeyboardInterrupt:
                    logger.info("KeyboardInterrupt caught during configuration wait.")
                    raise  # propagate the exception

                # parse the configuration file for the experiment run
                setup_config = '.setup.cfg.json'
                if os.path.exists(setup_config):
                    with open(setup_config) as cf:
                        config_data = json.load(cf)
                        if "num_runs" in config_data:
                            args.num_runs = int(config_data["num_runs"])
                        if "save_model" in config_data:
                            args.save_model = bool(config_data["save_model"])
                        if "load_model" in config_data:
                            args.load_model = config_data["load_model"]

                logger.info(f"Configuration complete. Game is starting and will complete {args.num_runs} runs.")
        else:
            shared_data = None
            episode_queue = None

        load_model = None
        if args.load_model is not None:
            if os.path.exists(args.load_model):
                load_model = args.load_model
            else:
                logger.warning(f"Could not find model checkpoint: {args.load_model}")

        for run_num in range(args.num_runs):
            if args.use_gui > 0 and exit_event.is_set():
                logger.info("Exit requested by GUI. Exiting run.")
                break

            run_dir = os.path.join(data_dir, f"run_{run_num}")
            os.makedirs(run_dir, exist_ok=True)

            seed = args.seed + run_num
            game = None
            try:
                # Init env model.
                from env_physical import PhysicalEnv

                env = PhysicalEnv(
                    args.game_config,
                    args.camera_config,
                    args.joystick_config,
                    args.detection_config,
                    score_detector_type,
                    device=dev,
                    obs_dims=obs_dims,
                    reduce_action_set=args.reduce_action_set,
                    data_dir=run_dir,
                )

                game = env.get_name()
                action_set = env.get_action_set()

                num_actions = len(action_set)
                logger.debug(f'{num_actions} actions: {action_set}')

                # Init a fresh model.
                if args.agent_type == 'agent_delay_target':
                    agent_args = {
                        'ring_buffer_size': 200 * 1024,
                        "use_model": 3,
                        "gpu": args.gpu,
                    }
                elif args.agent_type == 'agent_dqn':
                    agent_args = {
                        "gpu": args.gpu,
                        "buffer_size": args.dqn_buffer_size,
                        "batch_size": args.dqn_batch_size,
                        "learning_rate": args.dqn_learning_rate,
                        "gamma": args.dqn_gamma,
                        "train_start": args.dqn_train_start,
                        "train_freq": args.dqn_train_freq,
                        "target_update_freq": args.dqn_target_update_freq,
                        "epsilon_start": args.dqn_epsilon_start,
                        "epsilon_end": args.dqn_epsilon_end,
                        "epsilon_decay_frames": args.dqn_epsilon_decay_frames,
                        "stack_size": args.dqn_stack_size,
                        "obs_height": args.dqn_obs_height,
                        "obs_width": args.dqn_obs_width,
                    }
                elif args.agent_type == 'agent_rainbow':
                    agent_args = {
                        "gpu": args.gpu,
                        "buffer_size": args.rainbow_buffer_size,
                        "batch_size": args.rainbow_batch_size,
                        "learning_rate": args.rainbow_learning_rate,
                        "gamma": args.rainbow_gamma,
                        "train_start": args.rainbow_train_start,
                        "train_freq": args.rainbow_train_freq,
                        "target_update_freq": args.rainbow_target_update_freq,
                        "stack_size": args.rainbow_stack_size,
                        "obs_height": args.rainbow_obs_height,
                        "obs_width": args.rainbow_obs_width,
                        "n_step": args.rainbow_n_step,
                        "num_atoms": args.rainbow_num_atoms,
                        "v_min": args.rainbow_v_min,
                        "v_max": args.rainbow_v_max,
                        "priority_alpha": args.rainbow_priority_alpha,
                        "priority_beta": args.rainbow_priority_beta,
                        "priority_beta_increment": args.rainbow_priority_beta_increment,
                        "priority_eps": args.rainbow_priority_eps,
                        "epsilon_start": 0.0,
                        "epsilon_end": 0.0,
                    }
                else:
                    agent_args = {"gpu": args.gpu}
                if load_model is not None:
                    agent_args["load_file"] = load_model

                agent = Agent(run_dir, seed, num_actions, total_frames, **agent_args)

                last_model_save = -1
                save_incremental_model = args.save_model and args.save_model_increment > 0

                episode_avg = 0
                episode_scores = []
                episode_end = []
                environment_start = 0
                running_episode_score = 0
                experiment_start_time = environment_start_time = time.time()

                # put the average of 100 episodes in each slot, evenly divided by the total number of learning steps
                episode_graph = torch.zeros(1000, device='cpu')

                frames_without_reward = 0
                previous_lives = env.lives()

                # allow the commands to be delayed by this many 60 fps frames (useful when running ALE simulation)
                delayed_actions = [0] * args.delay_frames
                taken_action = 0

                # note that atlantis can learn to play indefinitely, so there may be no completed episodes in the window
                average_frames = 100_000  # frames to average episode scores over for episode_graph

                if args.log_score_images:
                    score_image_dir = os.path.join(run_dir, "score_images")
                    os.makedirs(score_image_dir, exist_ok=True)

                score_file = None
                if args.log_scores:
                    score_file = open(
                        os.path.join(run_dir, "scores_" + datetime.datetime.now().strftime("%Y%b%d-%H-%M-%S")) + ".log",
                        "w",
                    )

                logger.info("Starting Training")

                if args.capture_frames:
                    # raw filename expect format: name_{w}x{h}.{y or rgb}
                    capture_w, capture_h = 84, 84
                    capture_video_name = os.path.join(run_dir, f"{game}_{capture_w}x{capture_h}.rgb")
                    capture_video_file = open(capture_video_name, "wb")
                    capture_score_name = os.path.join(run_dir, "score_file.bin")
                    capture_score_file = open(capture_score_name, "ab")
                    capture_lives_name = os.path.join(run_dir, "lives_file.bin")
                    capture_lives_file = open(capture_lives_name, "ab")
                else:
                    capture_video_file = None
                    capture_score_file = None
                    capture_lives_file = None

                last_frame_time = environment_start_time

                target_fps = 60
                fps_frames = 0
                fps_start_time = time.time()
                fps = 0.0
            except Exception as e:
                logger.critical(f"Exception in run initialization: {e}", exc_info=True)
                continue

            try:
                for u in range(total_frames):
                    if args.use_gui > 0 and exit_event.is_set():
                        logger.info("Exit requested by GUI. Exiting training.")
                        break

                    if save_incremental_model and (u + 1) // args.save_model_increment != last_model_save:
                        last_model_save = (u + 1) // args.save_model_increment
                        filename = f'{run_dir}/{game}_{args.agent_type}.model'
                        logger.info('writing ' + filename)
                        agent.save_model(filename)

                    # fill in our average score graph so we get exactly 1000 points on it
                    if u * episode_graph.shape[0] // total_frames != (u + 1) * episode_graph.shape[0] // total_frames:
                        i = u * episode_graph.shape[0] // total_frames
                        count = 0
                        total = 0
                        for j in range(len(episode_scores) - 1, -1, -1):
                            if episode_end[j] < u - average_frames:
                                break
                            count += 1
                            total += episode_scores[j]
                        if count == 0:
                            # -999 placeholder for no data
                            episode_avg = -999
                        else:
                            episode_avg = total / count
                            # if no episodes were completed in the previous window, backfill with the current value
                            for j in range(i - 1, -1, -1):
                                if episode_graph[j] != -999:
                                    break
                                episode_graph[j] = episode_avg
                        episode_graph[i] = episode_avg

                    delayed_actions.append(taken_action)

                    start = time.time()
                    torch.cuda.nvtx.range_push("act")
                    cmd = delayed_actions.pop(0)
                    reward, info = env.act(action_set[cmd])
                    running_episode_score += reward
                    torch.cuda.nvtx.range_pop()
                    interframe_period = start - last_frame_time
                    last_frame_time = start

                    if reward != 0:
                        frames_without_reward = 0
                    else:
                        frames_without_reward += 1

                    end_of_episode = 0

                    if lives_as_episodes and env.lives() < previous_lives:
                        previous_lives = env.lives()
                        end_of_episode = 1  # loss of life
                    elif env.game_over():
                        end_of_episode = 2  # game over
                    elif frames_without_reward == max_frames_without_reward:
                        end_of_episode = 3  # terminated without game over
                        logger.debug(f'terminated at {frames_without_reward} frames without reward')

                    if end_of_episode > 1:
                        torch.cuda.nvtx.range_push("reset")
                        env.reset()
                        previous_lives = env.lives()
                        frames_without_reward = 0

                        frames = u - environment_start
                        episode_end.append(u)
                        environment_start = u
                        episode_scores.append(running_episode_score)
                        running_episode_score = 0

                        # calculate step speed
                        now = time.time()
                        ep_frames_per_second = frames / (now - environment_start_time)
                        environment_start_time = now

                        logger.info(
                            f'{game} frame:{u:7} {ep_frames_per_second:4.0f}/s eps {len(episode_scores) - 1},{frames:5}={int(episode_scores[-1]):5} avg {episode_avg:4.1f}'
                        )

                        if score_file:
                            score_file.write(
                                f"{game} episode: {len(episode_scores)} frame: {u} score: {episode_scores[-1]} "
                                + "time: %7.2f\n" % (time.time() - experiment_start_time)
                            )
                            score_file.flush()

                        if args.log_score_images and env.past_observation_cam:  # for validating score recognition
                            filename = score_image_dir + "/" + f"episode-{len(episode_scores):06d}.png"
                            Image.fromarray(env.past_observation_cam[0]).save(filename)

                        if episode_queue is not None:
                            episode_data = {
                                "episode": (episode_scores[-1], episode_end[-1]),
                                "episode_avg": episode_avg if episode_avg != -999 else 0,
                            }
                            episode_queue.put(episode_data)

                        torch.cuda.nvtx.range_pop()

                    torch.cuda.nvtx.range_push("env.get_observation")
                    observation_rgb8 = env.get_observation()
                    torch.cuda.nvtx.range_pop()

                    torch.cuda.nvtx.range_push("agent.frame")
                    taken_action = agent.frame(observation_rgb8, reward, end_of_episode)
                    torch.cuda.nvtx.range_pop()

                    if fps_frames == target_fps:
                        elapsed_time = time.time() - fps_start_time
                        fps = fps_frames / elapsed_time if elapsed_time > 0.0 else 0.0

                        fps_frames = 0
                        fps_start_time = time.time()
                    else:
                        fps_frames += 1

                    # logger.debug("%06d"%u, "% 8d"%int(1000*interframe_period))
                    if shared_data is not None:
                        # start_time = time.time()
                        torch.cuda.nvtx.range_push("write_to_shmem")
                        cam_frame_num, cam_frame = env.get_camera_frame()
                        rect_frame = observation_rgb8
                        frame_data = {
                            "frame": cam_frame_num,
                            "lives": env.lives(),
                            "total_lives": info["total_lives"],
                            "score": info["score"],
                            "action": action_set[taken_action].name,
                            "fps": fps,
                            "tags": info["tags"],
                            "reward_termination": (reward, (end_of_episode >= 1)),
                            "interframe_period": interframe_period,
                        }
                        shared_data.write_to_shmem(cam_frame, rect_frame, frame_data)
                        torch.cuda.nvtx.range_pop()
                        # if (time.time()-start_time) > 0.0005:
                        #    logger.debug(f"shmem write taking longer than expected: {(time.time()-start_time)*1000.0:.2f}ms")
                        # logger.debug(f"writing to shared mem={(time.time()-start_time)*1000.0:.2f}ms")

                    if capture_video_file is not None:
                        img = Image.fromarray(observation_rgb8)
                        img_resized = img.resize((84, 84), Image.Resampling.LANCZOS)
                        resized_array = np.array(img_resized)
                        capture_video_file.write(resized_array.tobytes())

                    if capture_score_file is not None:
                        capture_score_file.write(struct.pack('<f', info["score"]))

                    if capture_lives_file is not None:
                        capture_lives_file.write(struct.pack('<f', env.lives()))

            except Exception as e:
                logger.critical(f"Exception in game run: {e}", exc_info=True)
            finally:
                # avoid spurious error messages when the game and environment were not initialized successfully.
                if game is None:
                    continue

                # write results for the run
                filename = run_dir + '/' + game + '.score'
                logger.info('writing ' + filename)
                episode_graph.cpu().numpy().tofile(filename)

                plots = torch.zeros(len(episode_scores), 2)
                for i in range(len(episode_scores)):
                    plots[i][0] = episode_end[i]
                    plots[i][1] = episode_scores[i]
                filename = run_dir + '/' + game + '.scatter'
                logger.info('writing ' + filename)
                plots.cpu().numpy().tofile(filename)

                filename = run_dir + '/' + game + '.loss'
                logger.info('writing ' + filename)
                torch.tensor(agent.train_losses).cpu().numpy().tofile(filename)

                if args.save_model:
                    filename = f'{run_dir}/{game}_{args.agent_type}.model'
                    logger.info('writing ' + filename)
                    agent.save_model(filename)

                env.close()
                env = None
                agent = None

                if score_file:
                    score_file.close()

                if capture_video_file is not None:
                    capture_video_file.close()
                    capture_video_file = None

                if capture_score_file is not None:
                    capture_score_file.close()
                    capture_score_file = None

                if capture_lives_file is not None:
                    capture_lives_file.close()
                    capture_lives_file = None

                if shared_data is not None:
                    shared_data.write_to_shmem(None, None, {"run_complete": run_num})

                logger.info(f"Training complete for run {run_num + 1}/{args.num_runs}")

    finally:
        logger.info('Exiting')

        if args.use_gui > 0:
            assert shared_data is not None
            shared_data.write_to_shmem(None, None, {"shutdown": 1})
            gui_process.join()
            if gui_process.exitcode != 0:
                logger.warning(f"gui_process crashed or exited with error {gui_process.exitcode}.")
            shared_data.close()
            shared_data.shutdown()

        logger.info('Complete.')


def get_argument_parser():
    from argparse import ArgumentParser

    from framework import ScoreDetectorConfig

    parser = ArgumentParser(description="harness_physical.py arguments")
    parser.add_argument('--results_dir', type=str, default=os.path.join(os.getcwd(), 'results'))
    parser.add_argument(
        '--log_level', type=str, default="INFO", choices=["DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"]
    )
    parser.add_argument('--game_config', type=str, default="configs/games/ms_pacman.json")
    parser.add_argument('--camera_config', type=str, default="configs/cameras/camera_kiyo_pro.json")
    parser.add_argument('--joystick_config', type=str, default="configs/controllers/robotroller.json")
    parser.add_argument('--detection_config', type=str, default="configs/screen_detection/fixed.json")
    parser.add_argument('--description', type=str, default="Experiment description")
    parser.add_argument(
        '--score_detector_type',
        type=str,
        default=ScoreDetectorConfig.DEFAULT_MODEL,
        choices=ScoreDetectorConfig.ALL_MODELS,
    )
    parser.add_argument(
        '--agent_type',
        type=str,
        default="agent_delay_target",
        choices=["agent_delay_target", "agent_random", "agent_dqn", "agent_rainbow"],
    )
    parser.add_argument(
        '--reduce_action_set',
        type=int,
        default=2,
        choices=[0, 1, 2],
        help="0=legal, 1=minimal, 2=minimal w/ additional restrictions",
    )
    parser.add_argument(
        '--lives_as_episode', type=int, default=1, choices=[0, 1], help="treat loss of life as an episode end"
    )
    parser.add_argument('--num_runs', type=int, default=1)
    parser.add_argument('--seed', type=int, default=0)
    parser.add_argument('--total_frames', type=int, default=1_000_000)
    parser.add_argument(
        '--use_gui', type=int, default=2, choices=[0, 1, 2], help="0=no gui, 1=gui no config step, 2=gui w/ config step"
    )

    # DQN-specific configuration (used when --agent_type=agent_dqn)
    parser.add_argument('--dqn_buffer_size', type=int, default=100_000)
    parser.add_argument('--dqn_batch_size', type=int, default=32)
    parser.add_argument('--dqn_learning_rate', type=float, default=2.5e-4)
    parser.add_argument('--dqn_gamma', type=float, default=0.99)
    parser.add_argument('--dqn_train_start', type=int, default=50_000)
    parser.add_argument('--dqn_train_freq', type=int, default=4)
    parser.add_argument('--dqn_target_update_freq', type=int, default=10_000)
    parser.add_argument('--dqn_epsilon_start', type=float, default=1.0)
    parser.add_argument('--dqn_epsilon_end', type=float, default=0.1)
    parser.add_argument('--dqn_epsilon_decay_frames', type=int, default=1_000_000)
    parser.add_argument('--dqn_stack_size', type=int, default=4)
    parser.add_argument('--dqn_obs_height', type=int, default=84)
    parser.add_argument('--dqn_obs_width', type=int, default=84)

    # Rainbow-specific configuration
    parser.add_argument('--rainbow_buffer_size', type=int, default=500_000)
    parser.add_argument('--rainbow_batch_size', type=int, default=32)
    parser.add_argument('--rainbow_learning_rate', type=float, default=1e-4)
    parser.add_argument('--rainbow_gamma', type=float, default=0.99)
    parser.add_argument('--rainbow_train_start', type=int, default=50_000)
    parser.add_argument('--rainbow_train_freq', type=int, default=1)
    parser.add_argument('--rainbow_target_update_freq', type=int, default=2_000)
    parser.add_argument('--rainbow_stack_size', type=int, default=4)
    parser.add_argument('--rainbow_obs_height', type=int, default=84)
    parser.add_argument('--rainbow_obs_width', type=int, default=84)
    parser.add_argument('--rainbow_n_step', type=int, default=3)
    parser.add_argument('--rainbow_num_atoms', type=int, default=51)
    parser.add_argument('--rainbow_v_min', type=float, default=-10.0)
    parser.add_argument('--rainbow_v_max', type=float, default=10.0)
    parser.add_argument('--rainbow_priority_alpha', type=float, default=0.5)
    parser.add_argument('--rainbow_priority_beta', type=float, default=0.4)
    parser.add_argument('--rainbow_priority_beta_increment', type=float, default=1e-6)
    parser.add_argument('--rainbow_priority_eps', type=float, default=1e-6)

    parser.add_argument('--gpu', type=int, default=0)
    parser.add_argument('--load_model', type=str, default=None)
    parser.add_argument('--save_model', action='store_true')
    parser.add_argument(
        '--save_model_increment',
        type=int,
        default=0,
        help="when save_model=True and save_model_increment > 0, save the model every 'save_model_increment' frames.",
    )
    parser.add_argument('--log_scores', action='store_true')
    parser.add_argument('--log_score_images', action='store_true')
    parser.add_argument('--capture_frames', action='store_true', help="generate a raw movie of run")
    # REVIEW: only used for testing ale_env within the harness
    parser.add_argument('--delay_frames', type=int, default=0)
    return parser


if __name__ == '__main__':
    arg_parser = get_argument_parser()
    args = arg_parser.parse_args()

    try:
        main(args)
    except KeyboardInterrupt:
        logger.debug("KeyboardInterrupt received")

    exit(0)
