import argparse
import ast
import os
import time
from importlib import import_module

from ale_py import Action, ALEInterface, LoggerMode, roms

from latency_wrap.wrapper_v0_2 import LatencyModel


SUPPORTED_AGENTS = {
    "agent_dqn": "agent_dqn",
    "agent_delay_target": "agent_delay_target",
    "agent_random": "agent_random",
    "agent_rainbow": "agent_rainbow",
}


def parse_agent_args(raw_args):
    parsed = {}
    for entry in raw_args:
        if '=' not in entry:
            raise ValueError(f"Invalid --agent_arg entry '{entry}'. Expected format key=value.")
        key, value = entry.split('=', 1)
        key = key.strip()
        value = value.strip()
        try:
            parsed[key] = ast.literal_eval(value)
        except (ValueError, SyntaxError):
            parsed[key] = value
    return parsed


def build_argument_parser():
    parser = argparse.ArgumentParser(description="Generic Atari simulator loop with latency wrapper")
    parser.add_argument('--rom', type=str, default='ms_pacman')
    parser.add_argument('--seed', type=int, default=0)
    parser.add_argument('--total_frames', type=int, default=1_000_000)
    parser.add_argument('--results_dir', type=str, default=os.path.join(os.getcwd(), 'results', 'sim_latency'))
    parser.add_argument('--delay_frames', type=int, default=6)
    parser.add_argument('--latency_weights', type=str, default='latency_wrap')
    parser.add_argument('--no_latency', action='store_true')
    parser.add_argument('--checkpoint_interval', type=int, default=200_000)
    parser.add_argument('--checkpoint_name', type=str, default='sim_agent.pt')
    parser.add_argument('--agent_type', type=str, default='agent_dqn', choices=list(SUPPORTED_AGENTS.keys()))
    parser.add_argument('--load_model', type=str, default=None)
    parser.add_argument('--agent_arg', action='append', default=[], help="Extra agent kwargs (key=value)")
    parser.add_argument('--log_interval', type=int, default=10_000)
    parser.add_argument('--no_wandb', action='store_true')
    parser.add_argument('--wandb_project', type=str, default='sim-latency')
    parser.add_argument('--wandb_entity', type=str, default=None)
    parser.add_argument('--wandb_run_name', type=str, default=None)
    parser.add_argument('--record_video', action='store_true', help="Record gameplay videos for selected episodes")
    parser.add_argument('--video_dir', type=str, default=os.path.join(os.getcwd(), 'videos', 'sim_latency'))
    parser.add_argument('--video_every', type=int, default=10, help="Record every N episodes when --record_video is set")
    parser.add_argument('--video_fps', type=int, default=60)
    return parser


def maybe_init_wandb(args, config):
    if args.no_wandb:
        return None
    try:
        import wandb  # type: ignore
    except ImportError:
        print("wandb not installed; skipping logging. Install wandb or pass --no_wandb.")
        return None

    run = wandb.init(
        project=args.wandb_project,
        entity=args.wandb_entity,
        name=args.wandb_run_name,
        config=config,
        reinit=True,
    )
    return run


def save_checkpoint(agent, results_dir, filename):
    os.makedirs(results_dir, exist_ok=True)
    path = os.path.join(results_dir, filename)
    agent.save_model(path)
    print(f"Saved checkpoint to {path}")


def build_agent(agent_type, results_dir, seed, num_actions, total_frames, load_model, extra_args):
    module_name = SUPPORTED_AGENTS[agent_type]
    module = import_module(module_name)
    AgentClass = getattr(module, "Agent")

    # default kwargs per agent type
    default_kwargs = {
        "agent_dqn": {
            "gpu": 0,
            "buffer_size": 100_000,
            "batch_size": 32,
            "learning_rate": 2.5e-4,
            "gamma": 0.99,
            "train_start": 50_000,
            "train_freq": 4,
            "target_update_freq": 10_000,
            "epsilon_start": 1.0,
            "epsilon_end": 0.1,
            "epsilon_decay_frames": 1_000_000,
            "stack_size": 4,
            "obs_height": 84,
            "obs_width": 84,
        },
        "agent_delay_target": {
            "gpu": 0,
            "ring_buffer_size": 200 * 1024,
            "use_model": 3,
        },
        "agent_random": {
            "gpu": -1,
        },
        "agent_rainbow": {
            "gpu": 0,
            "buffer_size": 500_000,
            "batch_size": 32,
            "learning_rate": 1e-4,
            "gamma": 0.99,
            "train_start": 50_000,
            "train_freq": 1,
            "target_update_freq": 2_000,
            "stack_size": 4,
            "obs_height": 84,
            "obs_width": 84,
            "n_step": 3,
            "num_atoms": 51,
            "v_min": -10.0,
            "v_max": 10.0,
            "priority_alpha": 0.5,
            "priority_beta": 0.4,
            "priority_beta_increment": 1e-6,
            "priority_eps": 1e-6,
            "epsilon_start": 0.0,
            "epsilon_end": 0.0,
        },
    }[agent_type].copy()

    default_kwargs.update(extra_args)
    if load_model is not None:
        default_kwargs["load_file"] = load_model

    agent = AgentClass(
        data_dir=results_dir,
        seed=seed,
        num_actions=num_actions,
        total_frames=total_frames,
        **default_kwargs,
    )
    return agent


def main():
    parser = build_argument_parser()
    args = parser.parse_args()

    os.makedirs(args.results_dir, exist_ok=True)

    ale = ALEInterface()
    ale.setLoggerMode(LoggerMode.Error)
    ale.setInt('random_seed', args.seed)
    rom_path = roms.get_rom_path(args.rom)
    ale.loadROM(rom_path)

    legal_actions = ale.getMinimalActionSet()
    action_set = [Action(a) for a in legal_actions]
    action_to_index = {act: idx for idx, act in enumerate(action_set)}

    agent_kwargs = parse_agent_args(args.agent_arg)
    agent = build_agent(
        args.agent_type,
        args.results_dir,
        args.seed,
        len(action_set),
        args.total_frames,
        args.load_model,
        agent_kwargs,
    )

    latency_model = None if args.no_latency else LatencyModel(args.latency_weights)

    if args.record_video:
        import imageio
        os.makedirs(args.video_dir, exist_ok=True)

    config = vars(args).copy()
    config['agent_kwargs'] = agent_kwargs
    config.pop('agent_arg', None)
    wandb_run = maybe_init_wandb(args, config)

    delayed_actions = [0] * args.delay_frames
    taken_action_index = 0
    obs = ale.getScreenRGB()
    episode_reward = 0.0
    episode = 0

    def should_record(ep_index: int) -> bool:
        return args.record_video and (ep_index % args.video_every == 0)

    next_episode_index = 1
    record_episode = should_record(next_episode_index)
    episode_frames = [] if record_episode else []

    start_time = time.time()

    for frame in range(args.total_frames):
        delayed_actions.append(taken_action_index)
        cmd_index = delayed_actions.pop(0)
        if args.record_video and record_episode:
            episode_frames.append(obs.copy())

        reward = ale.act(legal_actions[cmd_index])
        episode_reward += reward

        done = ale.game_over()
        end_flag = 2 if done else 0

        next_obs = ale.getScreenRGB()
        if args.record_video and record_episode:
            episode_frames.append(next_obs.copy())

        agent_action_index = agent.frame(obs, reward, end_flag)
        requested_action = action_set[agent_action_index]

        if latency_model is not None:
            hw_action = latency_model.act(requested_action)
            taken_action_index = action_to_index.get(hw_action, agent_action_index)
        else:
            taken_action_index = agent_action_index

        if args.checkpoint_interval > 0 and (frame + 1) % args.checkpoint_interval == 0:
            save_checkpoint(agent, args.results_dir, f"{args.rom}_{frame + 1}.pt")

        if done:
            episode += 1
            elapsed = time.time() - start_time
            fps = frame / elapsed if elapsed > 0 else 0.0
            log_dict = {
                "episode": episode,
                "episode_reward": episode_reward,
                "epsilon": getattr(agent, 'epsilon', None),
                "frames": frame,
                "fps": fps,
                "loss": getattr(agent, 'last_loss', None),
                "loss_ema": getattr(agent, 'loss_ema', None),
                "avg_q": getattr(agent, 'last_avg_q', None),
                "max_q": getattr(agent, 'last_max_q', None),
            }
            print(
                f"Episode {episode:4d} | score {episode_reward:6.0f} | frames {frame:7d} | "
                f"fps {fps:5.1f} | epsilon {log_dict['epsilon'] if log_dict['epsilon'] is not None else -1}"
            )
            if wandb_run is not None:
                wandb_run.log({k: v for k, v in log_dict.items() if v is not None}, step=frame)

            if args.record_video and record_episode and episode_frames:
                video_path = os.path.join(
                    args.video_dir,
                    f"{args.rom}_episode_{episode:04d}.mp4",
                )
                imageio.mimsave(video_path, episode_frames, fps=args.video_fps)
                print(f"Saved video {video_path}")
            episode_frames = []
            record_episode = False

            ale.reset_game()
            next_obs = ale.getScreenRGB()
            episode_reward = 0.0
            delayed_actions = [0] * args.delay_frames

            next_episode_index = episode + 1
            record_episode = should_record(next_episode_index)
            episode_frames = [] if record_episode else []

        if args.log_interval > 0 and (frame + 1) % args.log_interval == 0:
            elapsed = time.time() - start_time
            fps = frame / elapsed if elapsed > 0 else 0.0
            epsilon = getattr(agent, 'epsilon', None)
            loss = getattr(agent, 'last_loss', None)
            loss_ema = getattr(agent, 'loss_ema', None)
            print(
                f"[Frame {frame + 1}] fps={fps:6.1f} "
                + (f"epsilon={epsilon:.3f} " if epsilon is not None else "")
                + (f"loss={loss:.4f} " if loss is not None else "")
                + (f"loss_ema={loss_ema:.4f}" if loss_ema is not None else "")
            )

        obs = next_obs

    if args.checkpoint_name:
        save_checkpoint(agent, args.results_dir, args.checkpoint_name)
    if wandb_run is not None:
        wandb_run.finish()
    print("Simulation complete.")


if __name__ == '__main__':
    main()
