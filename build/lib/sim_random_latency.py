from ale_py import ALEInterface, LoggerMode, Action, roms
from agent_random import Agent
from latency_wrap.wrapper_v0_2 import LatencyModel
import numpy as np

# this loads the ALEInterface(): https://ale.farama.org/python-interface/
ale = ALEInterface()
# Sets the logging level of the ALE emulator to only show errors, suppressing less critical messages
ale.setLoggerMode(LoggerMode.Error)
# Sets a fixed random seed (0) for the Atari Learning Environment to ensure reproducible results
ale.setInt("random_seed", 0)
# Loads the Ms. Pac-Man ROM file into the Atari Learning Environment (ALE) emulator
# The ROM path is retrieved using roms.get_rom_path() which finds the ROM file in the ale-py installation
ale.loadROM(roms.get_rom_path("ms_pacman"))

# This is the action set for the game usually you can do something like ale.getLegalActionSet()
action_set = [Action.UP, Action.DOWN, Action.LEFT, Action.RIGHT]
# This is the random agent thats imported
agent = Agent(data_dir="results/sim_random", seed=0, num_actions=len(action_set), total_frames=200_000)

# LatencyModel mimics the RoboTroller delay. It keeps ~0.5 s of command history (30 frames at 60 Hz) and,
# when the agent requests something new (e.g., RIGHT→UP), predicts the hardware’s next output—often another RIGHT
# or a transient RIGHT+UP—until the joystick actually reaches UP.

# The extra history isn't because it takes 30 frames to turn—it's so the model can see
# the context of your commands. Keen trained this little MLP on real RoboTroller logs,
# and they found that feeding it ~0.5 s of recent "requested vs. executed" actions gave
# it enough information to spot patterns: how long the stick's been held in one direction,
# whether it was oscillating (e.g., UP/RIGHT), whether it's already mid-swing, etc. With
# that context, the network can predict the next hardware output more reliably than if it
# only looked at the latest request. So those 30 frames are just the input window for the
# predictor—not a literal 30-frame delay

latency = LatencyModel("latency_wrap")

# There is also a 6-frame delay between the agent picks the action and when the joystick can act on it.
delayed_actions = [0] * 6
# This gets the pure RGB observation to act on
obs = ale.getScreenRGB()
# Reward initialization
reward = 0
# Flag for whether game ended or not
end_flag = 0
# Looping through the specified number of frames
for t in range(agent.total_frames):
    # Adding the action to back of queue to maintain 6 frame delay
    delayed_actions.append(latency.last_action if hasattr(latency, "last_action") else 0)
    # Get the action at the front of queue
    cmd = delayed_actions.pop(0)
    # The sim executes action
    reward = ale.act(int(action_set[cmd]))
    # Next action for agent to take
    frame_action = agent.frame(obs, reward, end_flag)
    # Action joystick will take based on predicted action
    hw_action = latency.act(Action(frame_action))
    # Setting next action in queue
    delayed_actions[-1] = action_set.index(hw_action)

    # End/Continue conditions
    if ale.game_over():
        obs = ale.reset_game()
        end_flag = 2
    else:
        obs = ale.getScreenRGB()
        end_flag = 0


# You need to include the delayed_actions/latency wrapper logic in your training loop for algorithms you train (DQN, PPO, etc.)
# In your agent file you'll do the preprocessing necessary (FrameStack, GrayScale, etc.)
