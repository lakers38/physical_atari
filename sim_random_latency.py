from ale_py import ALEInterface, LoggerMode, Action, roms
from agent_random import Agent
from latency_wrap.wrapper_v0_2 import LatencyModel
import numpy as np

ale = ALEInterface()
ale.setLoggerMode(LoggerMode.Error)
ale.setInt("random_seed", 0)
ale.loadROM(roms.get_rom_path("ms_pacman"))

action_set = [Action.UP, Action.DOWN, Action.LEFT, Action.RIGHT]
agent = Agent(data_dir="results/sim_random", seed=0, num_actions=len(action_set), total_frames=200_000)
latency = LatencyModel("latency_wrap")

delayed_actions = [0] * 6
obs = ale.getScreenRGB()
reward = 0
end_flag = 0

for t in range(agent.total_frames):
    delayed_actions.append(latency.last_action if hasattr(latency, "last_action") else 0)
    cmd = delayed_actions.pop(0)
    print(action_set[cmd])
    reward = ale.act(int(action_set[cmd]))

    frame_action = agent.frame(obs, reward, end_flag)
    hw_action = latency.act(Action(frame_action))
    delayed_actions[-1] = action_set.index(hw_action)

    if ale.game_over():
        obs = ale.reset_game()
        end_flag = 2
    else:
        obs = ale.getScreenRGB()
        end_flag = 0
