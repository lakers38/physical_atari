#!/usr/bin/env python3

import argparse
import multiprocessing as mp
import os
import queue
import time
from pathlib import Path
from types import SimpleNamespace

import ale_py
import gymnasium as gym
import numpy as np
import torch

from agent_meme import VectorMEMEAgent, MEMECore, PrioritisedSequenceReplay
from sim_latency_vec import build_env


gym.register_envs(ale_py)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Distributed MEME trainer (actors + learner)")
    parser.add_argument("--rom", type=str, default="MsPacman")
    parser.add_argument("--num_actors", type=int, default=4)
    parser.add_argument("--envs_per_actor", type=int, default=4)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--total_frames", type=int, default=5_000_000)
    parser.add_argument("--latency_weights", type=str, default="latency_wrap")
    parser.add_argument("--results_dir", type=str, default=os.path.join(os.getcwd(), "results", "meme_distributed"))
    parser.add_argument("--replay_capacity", type=int, default=20_000)
    parser.add_argument("--replay_seq_len", type=int, default=160)
    parser.add_argument("--replay_burn_in", type=int, default=0)
    parser.add_argument("--batch_size", type=int, default=64)
    parser.add_argument("--learner_publish_interval", type=int, default=100)
    parser.add_argument("--weights_dir", type=str, default=os.path.join(os.getcwd(), "results", "meme_distributed", "weights"))
    parser.add_argument("--learner_gpu", type=int, default=0)
    parser.add_argument("--actor_gpu", type=int, default=-1)
    parser.add_argument("--train_interval", type=int, default=1)
    parser.add_argument("--log_interval", type=int, default=50)
    parser.add_argument("--checkpoint_interval", type=int, default=10_000)
    return parser.parse_args()


class ActorReplayClient:
    def __init__(self, request_queue: mp.Queue):
        self._queue = request_queue

    def add_episode(self, episode):
        self._queue.put({"type": "add", "episode": episode})

    def sample(self, batch_size: int):
        raise RuntimeError("Actors do not sample from replay")

    def update_priorities(self, indices, priorities):
        raise RuntimeError("Actors do not update replay priorities")


class LearnerReplayClient:
    def __init__(self, request_queue: mp.Queue, sample_queue: mp.Queue):
        self._req = request_queue
        self._sample = sample_queue

    def add_episode(self, episode):
        raise RuntimeError("Learner does not push raw episodes")

    def sample(self, batch_size: int):
        self._req.put({"type": "sample", "batch_size": batch_size})
        payload = self._sample.get()
        weights = torch.tensor(payload["weights"], dtype=torch.float32)
        return payload["sequences"], payload["indices"], weights

    def update_priorities(self, indices, priorities):
        self._req.put(
            {
                "type": "update",
                "indices": list(indices),
                "priorities": [float(x) for x in priorities.detach().cpu().numpy()],
            }
        )


def run_replay_server(args: argparse.Namespace, request_queue: mp.Queue, sample_queue: mp.Queue, shutdown: mp.Event):
    buffer = PrioritisedSequenceReplay(
        capacity=args.replay_capacity,
        seq_len=args.replay_seq_len,
        burn_in=args.replay_burn_in,
        stack_size=4,
        obs_shape=(84, 84),
        alpha=0.6,
        beta_start=0.4,
        beta_increment=1e-6,
    )
    while True:
        if shutdown.is_set() and request_queue.empty():
            break
        try:
            msg = request_queue.get(timeout=0.5)
        except queue.Empty:
            continue
        op = msg.get("type")
        if op == "add":
            buffer.add_episode(msg["episode"])
        elif op == "sample":
            sequences, indices, weights = buffer.sample(msg["batch_size"])
            if isinstance(weights, torch.Tensor):
                weights_payload = weights.cpu().tolist()
            else:
                weights_payload = list(weights)
            sample_queue.put({"sequences": sequences, "indices": indices, "weights": weights_payload})
        elif op == "update":
            tensor = torch.tensor(msg["priorities"], dtype=torch.float32)
            buffer.update_priorities(msg["indices"], tensor)
        elif op == "stop":
            break


def build_actor_env(args: argparse.Namespace, seed_offset: int):
    env_args = SimpleNamespace(
        rom=args.rom,
        num_envs=args.envs_per_actor,
        seed=args.seed + seed_offset,
        latency_weights=args.latency_weights,
    )
    return build_env(env_args)


def actor_worker(
    actor_id: int,
    args: argparse.Namespace,
    request_queue: mp.Queue,
    frame_counter: mp.Value,
    shutdown: mp.Event,
    weight_version: mp.Value,
    weights_path: Path,
):
    torch.set_num_threads(1)
    np.random.seed(args.seed + actor_id)
    env = build_actor_env(args, actor_id * args.envs_per_actor)
    results_dir = Path(args.results_dir) / f"actor_{actor_id:02d}"
    results_dir.mkdir(parents=True, exist_ok=True)
    action_space = getattr(env, "single_action_space", env.action_space)
    agent: VectorMEMEAgent = VectorMEMEAgent(
        num_envs=args.envs_per_actor,
        seed=args.seed + actor_id,
        num_actions=action_space.n,
        results_dir=str(results_dir),
        total_frames=args.total_frames,
        gpu=args.actor_gpu,
        replay=ActorReplayClient(request_queue),
    )
    agent.reset(args.envs_per_actor)
    obs, _ = env.reset()
    agent.core.reset(obs)
    last_loaded = 0
    step = 0
    while not shutdown.is_set():
        if weight_version.value > last_loaded and weights_path.exists():
            state = torch.load(weights_path, map_location=agent.core.device)
            agent.core.network.load_state_dict(state)
            agent.core.ema_network.load_state_dict(state)
            last_loaded = weight_version.value
        actions = agent.act(obs)
        next_obs, rewards, terms, truncs, infos = env.step(actions)
        agent.observe(next_obs, rewards, terms, truncs, infos)
        obs = next_obs
        with frame_counter.get_lock():
            frame_counter.value += args.envs_per_actor
            if frame_counter.value >= args.total_frames:
                shutdown.set()
        step += 1
    env.close()


def get_num_actions(rom: str) -> int:
    env = gym.make(f"ALE/{rom}-v5", obs_type="rgb")
    try:
        return env.action_space.n
    finally:
        env.close()


def learner_worker(
    args: argparse.Namespace,
    request_queue: mp.Queue,
    sample_queue: mp.Queue,
    shutdown: mp.Event,
    weight_version: mp.Value,
    weights_path: Path,
):
    torch.set_num_threads(max(1, os.cpu_count() or 1))
    replay_client = LearnerReplayClient(request_queue, sample_queue)
    num_actions = get_num_actions(args.rom)
    results_dir = Path(args.results_dir) / "learner"
    results_dir.mkdir(parents=True, exist_ok=True)
    core = MEMECore(
        num_envs=args.envs_per_actor,
        seed=args.seed,
        num_actions=num_actions,
        total_frames=args.total_frames,
        data_dir=str(results_dir),
        replay=replay_client,
        batch_size=args.batch_size,
        buffer_capacity=args.replay_capacity,
        seq_len=args.replay_seq_len,
        burn_in=args.replay_burn_in,
        train_interval=args.train_interval,
        gpu=args.learner_gpu,
    )
    while not shutdown.is_set():
        before = core.training_steps
        core.train_step()
        if core.training_steps == before:
            time.sleep(0.05)
            continue
        if core.training_steps % args.learner_publish_interval == 0:
            tmp_path = weights_path.with_suffix(".tmp")
            torch.save(core.ema_network.state_dict(), tmp_path)
            os.replace(tmp_path, weights_path)
            with weight_version.get_lock():
                weight_version.value += 1
        if core.training_steps % args.log_interval == 0:
            metrics = core.pop_metrics()
            if metrics:
                print(f"[Learner] step={core.training_steps} loss={metrics.get('loss/total', 0):.4f}")
    print("Learner shutting down")


def main():
    args = parse_args()
    mp.set_start_method("spawn", force=True)
    os.makedirs(args.results_dir, exist_ok=True)
    weights_dir = Path(args.weights_dir)
    weights_dir.mkdir(parents=True, exist_ok=True)
    weights_path = weights_dir / "latest.pt"

    request_queue: mp.Queue = mp.Queue(maxsize=512)
    sample_queue: mp.Queue = mp.Queue(maxsize=64)
    shutdown = mp.Event()
    frame_counter = mp.Value("Q", 0)
    weight_version = mp.Value("i", 0)

    replay_proc = mp.Process(
        target=run_replay_server,
        args=(args, request_queue, sample_queue, shutdown),
        daemon=True,
    )
    replay_proc.start()

    learner_proc = mp.Process(
        target=learner_worker,
        args=(args, request_queue, sample_queue, shutdown, weight_version, weights_path),
        daemon=True,
    )
    learner_proc.start()

    actors = []
    for actor_id in range(args.num_actors):
        proc = mp.Process(
            target=actor_worker,
            args=(actor_id, args, request_queue, frame_counter, shutdown, weight_version, weights_path),
            daemon=True,
        )
        proc.start()
        actors.append(proc)

    try:
        while any(proc.is_alive() for proc in actors + [learner_proc]):
            time.sleep(1)
            if frame_counter.value >= args.total_frames:
                shutdown.set()
                break
    except KeyboardInterrupt:
        print("Interrupted, shutting down...")
        shutdown.set()

    for proc in actors:
        proc.join()
    learner_proc.join()
    request_queue.put({"type": "stop"})
    replay_proc.join()


if __name__ == "__main__":
    main()
