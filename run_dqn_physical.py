import argparse
import os
import time

from agent_dqn import Agent
from env_physical import PhysicalEnv
from latency_wrap.wrapper_v0_2 import LatencyModel


def build_arg_parser():
    parser = argparse.ArgumentParser(description="Run agent_dqn directly against the PhysicalEnv")
    parser.add_argument('--results_dir', type=str, default=os.path.join(os.getcwd(), 'results', 'physical_dqn'))
    parser.add_argument('--game_config', type=str, default="configs/games/ms_pacman.json")
    parser.add_argument('--camera_config', type=str, default="configs/cameras/camera_kiyo_pro.json")
    parser.add_argument('--joystick_config', type=str, default="configs/controllers/robotroller.json")
    parser.add_argument('--detection_config', type=str, default="configs/screen_detection/fixed.json")
    parser.add_argument('--score_detector_type', type=str, default="crnn_ctc")
    parser.add_argument('--total_frames', type=int, default=200_000)
    parser.add_argument('--seed', type=int, default=0)
    parser.add_argument('--load_model', type=str, required=True, help="Path to the .pt checkpoint produced in simulation")
    parser.add_argument('--delay_frames', type=int, default=6, help="Command latency queue length")
    parser.add_argument('--latency_weights', type=str, default="latency_wrap", help="Directory containing latency model weights")
    parser.add_argument('--no_latency', action='store_true', help="Disable learned latency wrapper")
    parser.add_argument('--log_interval', type=int, default=1_000, help="Frames between console status prints")
    parser.add_argument('--gpu', type=int, default=0)
    parser.add_argument('--buffer_size', type=int, default=100_000)
    parser.add_argument('--batch_size', type=int, default=32)
    parser.add_argument('--learning_rate', type=float, default=2.5e-4)
    parser.add_argument('--gamma', type=float, default=0.99)
    parser.add_argument('--train_start', type=int, default=50_000)
    parser.add_argument('--train_freq', type=int, default=4)
    parser.add_argument('--target_update_freq', type=int, default=10_000)
    parser.add_argument('--epsilon_start', type=float, default=1.0)
    parser.add_argument('--epsilon_end', type=float, default=0.1)
    parser.add_argument('--epsilon_decay_frames', type=int, default=1_000_000)
    parser.add_argument('--stack_size', type=int, default=4)
    parser.add_argument('--obs_height', type=int, default=84)
    parser.add_argument('--obs_width', type=int, default=84)
    return parser


def main():
    parser = build_arg_parser()
    args = parser.parse_args()

    os.makedirs(args.results_dir, exist_ok=True)

    env = PhysicalEnv(
        args.game_config,
        args.camera_config,
        args.joystick_config,
        args.detection_config,
        args.score_detector_type,
        reduce_action_set=0,
        device=f'cuda:{args.gpu}' if args.gpu >= 0 else 'cpu',
        data_dir=args.results_dir,
    )

    action_set = env.get_action_set()
    num_actions = len(action_set)

    agent_kwargs = {
        "gpu": args.gpu,
        "buffer_size": args.buffer_size,
        "batch_size": args.batch_size,
        "learning_rate": args.learning_rate,
        "gamma": args.gamma,
        "train_start": args.train_start,
        "train_freq": args.train_freq,
        "target_update_freq": args.target_update_freq,
        "epsilon_start": args.epsilon_start,
        "epsilon_end": args.epsilon_end,
        "epsilon_decay_frames": args.epsilon_decay_frames,
        "stack_size": args.stack_size,
        "obs_height": args.obs_height,
        "obs_width": args.obs_width,
        "load_file": args.load_model,
    }

    agent = Agent(
        data_dir=args.results_dir,
        seed=args.seed,
        num_actions=num_actions,
        total_frames=args.total_frames,
        **agent_kwargs,
    )

    latency_model = None if args.no_latency else LatencyModel(args.latency_weights)

    delayed_actions = [0] * args.delay_frames
    taken_action = 0
    episode_reward = 0.0
    previous_lives = env.lives()
    running_episode = 0
    frames_without_reward = 0
    max_frames_without_reward = 18_000

    start_time = time.time()

    try:
        for frame in range(args.total_frames):
            delayed_actions.append(taken_action)
            cmd_index = delayed_actions.pop(0)
            reward, info = env.act(action_set[cmd_index])
            observation = env.get_observation()

            episode_reward += reward

            if reward != 0:
                frames_without_reward = 0
            else:
                frames_without_reward += 1

            end_of_episode = 0
            if env.lives() < previous_lives:
                end_of_episode = 1  # lost a life
                previous_lives = env.lives()
            if env.game_over() or frames_without_reward >= max_frames_without_reward:
                end_of_episode = 2

            agent_action = agent.frame(observation, reward, end_of_episode)
            requested_action = action_set[agent_action]

            if latency_model is not None:
                hw_action = latency_model.act(requested_action)
                taken_action = action_set.index(hw_action) if hw_action in action_set else agent_action
            else:
                taken_action = agent_action

            if end_of_episode >= 2:
                running_episode += 1
                elapsed = time.time() - start_time
                fps = frame / elapsed if elapsed > 0 else 0.0
                print(
                    f"Episode {running_episode:4d} | score {episode_reward:6.0f} | "
                    f"frame {frame:7d} | fps {fps:6.1f} | epsilon {agent.epsilon:.3f}"
                )
                env.reset()
                previous_lives = env.lives()
                frames_without_reward = 0
                episode_reward = 0.0
                delayed_actions = [0] * args.delay_frames

            if args.log_interval > 0 and (frame + 1) % args.log_interval == 0:
                elapsed = time.time() - start_time
                fps = frame / elapsed if elapsed > 0 else 0.0
                print(
                    f"[Frame {frame + 1}] epsilon={agent.epsilon:.3f} loss={agent.last_loss:.4f} "
                    f"loss_ema={agent.loss_ema if agent.loss_ema is not None else 0.0:.4f} fps={fps:6.1f}"
                )

    except KeyboardInterrupt:
        print("Interrupted by user.")
    finally:
        env.close()


if __name__ == '__main__':
    main()
