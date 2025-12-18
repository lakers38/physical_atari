# The original LatencyModel was shared/written by Khurram Javed
# Our contribution is BatchedLatencyModel which parallelizes 
# this for vectorized environments

import numpy as np
import ale_py
import base64, gzip, io

class BatchedLatencyModel:
    """
    Batched version of LatencyModel for vectorized environments.
    Processes all environments in a single forward pass for efficiency.
    """

    def __init__(self, directory_with_weights, n_envs):
        """
        Initialize batched latency model.

        Args:
            directory_with_weights (str): Path to model weights directory
            n_envs (int): Number of parallel environments
        """
        self.n_envs = n_envs

        # Load weights once (shared across all envs)
        self.fc_weight = np.load(f"{directory_with_weights}/fc_weight.npy")
        self.fc_bias = np.load(f"{directory_with_weights}/fc_bias.npy")
        self.pred_weight = np.load(f"{directory_with_weights}/pred_weight.npy")
        self.pred_bias = np.load(f"{directory_with_weights}/pred_bias.npy")

        # Separate state per environment
        self.action_queues = []
        self.last_actions = []
        for _ in range(n_envs):
            queue = [self._one_hot_encode(0, 0) for _ in range(30)]
            self.action_queues.append(queue)
            self.last_actions.append(0)

    def _one_hot_encode(self, action, last_action):
        """One-hot encode action and last_action into vector of length 36."""
        vec = np.zeros(36, dtype=np.float32)
        vec[int(action)] = 1.0
        vec[18 + int(last_action)] = 1.0
        return vec

    def _forward_batch(self, x):
        """
        Batched forward pass through the MLP.

        Args:
            x (np.ndarray): Input of shape (n_envs, 30*36)

        Returns:
            np.ndarray: Logits of shape (n_envs, 18)
        """
        x = x @ self.fc_weight.T + self.fc_bias
        x = np.maximum(x, 0)  # ReLU
        x = x @ self.pred_weight.T + self.pred_bias
        return x

    def act_batch(self, actions, allowed_actions=None):
        """
        Process all environments in one forward pass.

        Args:
            actions (np.ndarray): Array of shape (n_envs,) with action indices
            allowed_actions (list, optional): List of allowed action indices in full 18-action space.
                                            If provided, only these actions can be sampled.

        Returns:
            np.ndarray: Array of shape (n_envs,) with delayed action indices
        """
        # Update queues for all environments
        for i, action in enumerate(actions):
            action = int(action)
            # Match LatencyModel's conditional pop behavior
            if len(self.action_queues[i]) == 30:
                self.action_queues[i].pop(0)
            self.action_queues[i].append(self._one_hot_encode(action, self.last_actions[i]))

        # Build batched input: (n_envs, 30*36)
        batch_input = np.array([
            np.array(queue).reshape(-1) for queue in self.action_queues
        ], dtype=np.float32)

        # Single batched forward pass
        logits = self._forward_batch(batch_input)

        # Mask disallowed actions if specified
        if allowed_actions is not None:
            allowed_actions = np.array(allowed_actions)
            # Create mask: set logits for disallowed actions to -inf
            mask = np.zeros_like(logits)
            mask[:, allowed_actions] = 1.0
            logits = np.where(mask == 1.0, logits, -np.inf)

        # Softmax and argmax per environment
        logits_max = logits.max(axis=1, keepdims=True)
        probs = np.exp(logits - logits_max)
        probs /= probs.sum(axis=1, keepdims=True)

        sampled_actions = np.argmax(probs, axis=1)

        # Update last_actions for next iteration
        self.last_actions = sampled_actions.tolist()

        return sampled_actions

    def reset_env(self, env_idx):
        """Reset a single environment's state to NOOPs."""
        self.action_queues[env_idx] = [self._one_hot_encode(0, 0) for _ in range(30)]
        self.last_actions[env_idx] = 0

    def reset_all(self):
        """Reset all environments to initial state."""
        for i in range(self.n_envs):
            self.reset_env(i)


class LatencyModel:
    """
    Wraps Atari joystick actions with a learned model to simulate real-world latency.
    Maintains a queue of past actions and uses a small neural network to determine
    which action should be executed, based on recent input history.
    """

    def __init__(self, directory_with_weights="./"):
        """
        Initializes the latency model by loading neural network weights and
        initializing the action queue with NOOP actions.

        Args:
            directory_with_weights (str): Path to the directory containing the model's
                                          NumPy weight files: fc_weight.npy, fc_bias.npy,
                                          pred_weight.npy, pred_bias.npy.
        """
        self.action_queue = []
        for _ in range(30):  # Start with 30 NOOPs to fill the buffer
            self.action_queue.append(self.__one_hot_encode(0, 0, 36))

        self.fc_weight = np.load(f"{directory_with_weights}/fc_weight.npy")
        self.fc_bias = np.load(f"{directory_with_weights}/fc_bias.npy")
        self.pred_weight = np.load(f"{directory_with_weights}/pred_weight.npy")
        self.pred_bias = np.load(f"{directory_with_weights}/pred_bias.npy")
        self.last_action = 0

    def __one_hot_encode(self, value, value_2, length):
        """
        Returns a one-hot encoded vector.

        Args:
            value (int): Index to be set as 1.
            value_2 (int): Index in the second half to be set as 1
            length (int): Length of the output vector.

        Returns:
            list[float]: One-hot encoded vector.
        """
        vec = [0.0] * length
        vec[value] = 1.0
        vec[18 + value_2] = 1.0
        return vec

    def __forward(self, x):
        """
        Runs a forward pass through the two-layer MLP with ReLU activation.

        Args:
            x (np.ndarray): Input vector of shape (1, 30 * 18)

        Returns:
            np.ndarray: Output logits from the final layer.
        """
        x = x @ self.fc_weight.T + self.fc_bias
        x = np.maximum(x, 0)  # ReLU
        x = x @ self.pred_weight.T + self.pred_bias
        return x

    def act(self, action, allowed_actions=None):
        """
        Accepts a new joystick action, updates the action queue, and returns
        the predicted action to execute (with latency effects).

        Args:
            action (ale_py.Action): The new joystick action.
            allowed_actions (list[int] or None): If provided, restrict output to this set of action indices.

        Returns:
            ale_py.Action: The action to actually execute, based on the model's prediction.
        """
        action = int(action)
        # model input is history of the past 30 actions
        if len(self.action_queue) == 30:
            self.action_queue.pop(0)
        self.action_queue.append(self.__one_hot_encode(action, self.last_action, 36))
        representation = np.array(self.action_queue).reshape(1, -1)
        logits = self.__forward(representation)

        if allowed_actions is not None:
            allowed_actions = list(allowed_actions)
            masked_logits = np.full_like(logits, -np.inf)
            masked_logits[0, allowed_actions] = logits[0, allowed_actions]
            logits = masked_logits

        probs = np.exp(logits - np.max(logits))  # for numerical stability
        probs /= np.sum(probs)
        # sampled_action = int(np.random.choice(len(probs[0]), p=probs[0]))
        # pick first max action (first max in-case of ties)
        sampled_action = int(np.argmax(probs[0]))
        self.last_action = sampled_action
        return ale_py.Action(int(sampled_action))
    
    # Store weights in a binary file that can be read in c++. Save as fc_weight.bin, fc_bias.bin, pred_weight.bin, pred_bias.bin. I would read them in c++ as a char array and then convert to float. 
    def save_weights(self, directory_with_weights="./"):
        buf = io.BytesIO()
        np.savez("model_weights.npz" , fc_weight=self.fc_weight, fc_bias=self.fc_bias, pred_weight=self.pred_weight, pred_bias=self.pred_bias)
        b64 = base64.b64encode(gzip.compress(buf.getvalue())).decode("ascii")
        with open("data_embed.txt", "w") as f:
            f.write(f'DATA_LATENCY_NPZ = """{b64}"""')
        # print(raw)
        with open(f"{directory_with_weights}/fc_weight.bin", "wb") as f:
            f.write(self.fc_weight.tobytes())
        with open(f"{directory_with_weights}/fc_bias.bin", "wb") as f:
            f.write(self.fc_bias.tobytes())
        with open(f"{directory_with_weights}/pred_weight.bin", "wb") as f:
            f.write(self.pred_weight.tobytes())
        with open(f"{directory_with_weights}/pred_bias.bin", "wb") as f:
            f.write(self.pred_bias.tobytes())
    
    


if __name__ == "__main__":
    # Example usage with a sequence of actions
    sample_actions = [0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1, 1, 1, 14, 14, 14, 14, 14, 14, 14, 14, 16, 16, 16, 16, 16, 16, 16, 16, 0, 0, 0, 0, 0, 0, 0, 0, 11, 11, 11, 11, 11, 11, 11, 11, 1, 1, 1, 1, 1, 1, 1, 1, 7, 7, 7, 7, 7, 7, 7, 7, 11, 11, 11, 11, 11, 11, 11, 11, 3, 3, 3, 3, 3, 3, 3, 3, 12, 12, 12, 12, 12, 12, 12, 12, 15, 15, 15, 15, 15, 15, 15, 15, 4, 4, 4, 4, 4, 4, 4, 4, 14, 14, 14, 14, 14, 14, 14, 14, 10, 10, 10, 10, 10, 10, 10, 10, 12, 12, 12, 12, 12, 12, 12, 12, 9, 9, 9, 9, 9, 9, 9, 9, 4, 4, 4, 4, 4, 4, 4, 4, 12, 12, 12, 12, 12, 12, 12, 12, 14, 14, 14, 14, 14, 14, 14, 14, 6, 6, 6, 6, 6, 6, 6, 6, 4, 4, 4, 4, 4, 4, 4, 4, 3, 3, 3, 3, 3, 3, 3, 3, 5, 5, 5, 5, 5, 5, 5, 5, 14, 14, 14, 14, 14, 14, 14, 14, 0, 0, 0, 0, 0, 0, 0, 0, 14, 14, 14, 14, 14, 14, 14, 14, 0, 0, 0, 0, 0, 0, 0, 0, 4, 4, 4, 4, 4, 4, 4, 4, 9, 9, 9, 9, 9, 9, 9, 9, 15, 15, 15, 15, 15, 15, 15, 15, 10, 10, 10, 10, 10, 10, 10, 10, 4, 4, 4, 4, 4, 4, 4, 4, 17, 17, 17, 17, 17, 17, 17, 17, 3, 3, 3, 3, 3, 3, 3, 3, 11, 11, 11, 11, 11, 11, 11, 11, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 17, 17, 17, 17, 17, 17, 17, 17, 7, 7, 7, 7, 7, 7, 7, 7, 8, 8, 8, 8, 8, 8, 8, 8, 0, 0, 0, 0, 0, 0, 0, 0, 11, 11, 11, 11, 11, 11, 11, 11, 9, 9, 9, 9, 9, 9, 9, 9, 16, 16, 16, 16, 16, 16, 16, 16, 8, 8, 8, 8, 8, 8, 8, 8, 2, 2, 2, 2, 2, 2, 2, 2, 14, 14, 14, 14, 14, 14, 14, 14, 0, 0, 0, 0, 0, 0, 0, 0, 11, 11, 11, 11, 11, 11, 11, 11, 4, 4, 4, 4, 4, 4, 4, 4, 16, 16, 16, 16, 16, 16, 16, 16, 7, 7, 7, 7, 7, 7, 7, 7, 6, 6, 6, 6, 6, 6, 6, 6, 14, 14, 14, 14, 14, 14, 14, 14, 1, 1, 1, 1, 1, 1, 1, 1, 6, 6, 6, 6, 6, 6, 6, 6, 14, 14, 14, 14, 14, 14, 14, 14, 6, 6, 6, 6, 6, 6, 6, 6, 17, 17, 17, 17, 17, 17, 17, 17, 9, 9, 9, 9, 9, 9, 9, 9, 7, 7, 7, 7, 7, 7, 7, 7, 13, 13, 13, 13, 13, 13, 13, 13, 3, 3, 3, 3, 3, 3, 3, 3, 17, 17, 17, 17, 17, 17, 17, 17, 8, 8, 8, 8, 8, 8, 8, 8, 17, 17, 17, 17, 17, 17, 17, 17, 9, 9, 9, 9, 9, 9, 9, 9, 13, 13, 13, 13, 13, 13, 13, 13, 9, 9, 9, 9, 9, 9, 9, 9, 4, 4, 4, 4, 4, 4, 4, 4, 10, 10, 10, 10, 10, 10, 10, 10, 13, 13, 13, 13, 13, 13, 13, 13, 12, 12, 12, 12, 12, 12, 12, 12, 4, 4, 4, 4, 4, 4, 4, 4, 15, 15, 15, 15, 15, 15, 15, 15, 10, 10, 10, 10, 10, 10, 10, 10, 9, 9, 9, 9, 9, 9, 9, 9, 6, 6, 6, 6, 6, 6, 6, 6, 7, 7, 7, 7, 7, 7, 7, 7, 1, 1, 1, 1, 1, 1, 1, 1, 9, 9, 9, 9, 9, 9, 9, 9, 5, 5, 5, 5, 5, 5, 5, 5, 2, 2, 2, 2, 2, 2, 2, 2, 4, 4, 4, 4, 4, 4, 4, 4, 15, 15, 15, 15, 15, 15, 15, 15, 17, 17, 17, 17, 17, 17, 17, 17, 17, 17, 17, 17, 17, 17, 17, 17, 5, 5, 5, 5, 5, 5, 5, 5, 16, 16, 16, 16, 16, 16, 16, 16, 16, 16, 16, 16, 16, 16, 16, 16, 14, 14, 14, 14, 14, 14, 14, 14, 0, 0, 0, 0, 0, 0, 0, 0, 16, 16, 16, 16, 16, 16, 16, 16, 0, 0, 0, 0, 0, 0, 0, 0, 6, 6, 6, 6, 6, 6, 6, 6, 2, 2, 2, 2, 2, 2, 2, 2, 14, 14, 14, 14, 14, 14, 14, 14, 6, 6, 6, 6, 6, 6, 6, 6, 2, 2, 2, 2, 2, 2, 2, 2, 1]
    gt_actions = [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 8, 8, 8, 8, 8, 16, 16, 16, 16, 8, 0, 0, 0, 0, 0, 0, 0, 1, 11, 11, 11, 11, 11, 11, 11, 11, 1, 1, 1, 1, 1, 1, 1, 2, 7, 7, 7, 7, 7, 7, 7, 15, 1, 1, 11, 11, 11, 11, 11, 3, 3, 3, 3, 3, 3, 3, 3, 11, 1, 1, 12, 12, 12, 12, 12, 12, 12, 12, 12, 12, 12, 12, 12, 4, 4, 4, 4, 4, 4, 4, 4, 15, 10, 10, 10, 14, 14, 14, 14, 10, 10, 10, 10, 10, 10, 10, 10, 10, 12, 12, 12, 12, 12, 12, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 12, 12, 12, 12, 12, 12, 12, 12, 15, 10, 10, 10, 14, 14, 14, 14, 6, 6, 6, 6, 6, 6, 6, 2, 0, 0, 0, 4, 4, 4, 4, 4, 4, 0, 0, 3, 3, 3, 3, 3, 8, 5, 5, 5, 5, 5, 5, 5, 16, 11, 11, 11, 11, 11, 11, 11, 3, 0, 0, 0, 0, 0, 0, 0, 11, 14, 14, 14, 14, 14, 14, 14, 3, 0, 5, 5, 0, 0, 0, 0, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 12, 12, 12, 12, 12, 12, 12, 15, 15, 10, 10, 10, 10, 10, 10, 10, 7, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 12, 12, 12, 12, 12, 12, 12, 4, 0, 0, 3, 3, 3, 3, 3, 11, 11, 11, 11, 11, 11, 11, 11, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 11, 13, 13, 13, 13, 13, 13, 17, 17, 4, 4, 4, 4, 7, 4, 4, 0, 0, 8, 8, 8, 8, 8, 5, 0, 0, 0, 0, 0, 0, 0, 11, 11, 11, 11, 11, 11, 11, 11, 8, 5, 5, 5, 5, 5, 5, 5, 13, 13, 13, 13, 13, 13, 13, 13, 5, 5, 5, 5, 5, 5, 5, 5, 5, 0, 2, 2, 2, 2, 2, 2, 2, 2, 10, 14, 14, 14, 14, 14, 14, 10, 2, 0, 0, 0, 0, 0, 0, 11, 11, 11, 11, 11, 11, 11, 11, 3, 0, 0, 4, 4, 4, 4, 4, 12, 13, 13, 13, 13, 13, 13, 13, 5, 0, 0, 7, 7, 7, 7, 7, 2, 2, 2, 2, 6, 6, 6, 6, 6, 14, 14, 14, 14, 14, 14, 1, 1, 13, 13, 1, 1, 1, 1, 0, 6, 6, 6, 6, 6, 6, 6, 14, 14, 14, 14, 14, 14, 14, 14, 14, 14, 6, 6, 6, 6, 6, 6, 6, 11, 1, 13, 17, 17, 17, 17, 17, 9, 9, 9, 9, 9, 9, 9, 9, 9, 4, 4, 4, 7, 7, 7, 12, 1, 1, 16, 13, 13, 13, 13, 8, 3, 3, 3, 3, 3, 3, 3, 11, 13, 13, 13, 13, 13, 13, 13, 5, 5, 5, 5, 5, 5, 5, 5, 13, 13, 13, 13, 13, 13, 13, 13, 5, 5, 5, 5, 5, 5, 5, 5, 13, 13, 13, 13, 13, 13, 13, 13, 5, 5, 5, 5, 5, 5, 5, 9, 9, 4, 4, 4, 4, 4, 4, 4, 12, 10, 10, 10, 10, 10, 10, 10, 1, 1, 1, 13, 13, 13, 13, 13, 12, 12, 12, 12, 12, 12, 12, 4, 4, 4, 4, 4, 4, 4, 4, 12, 12, 12, 12, 12, 12, 12, 12, 12, 10, 10, 10, 10, 10, 10, 10, 7, 4, 4, 4, 4, 4, 4, 4, 4, 0, 2, 6, 6, 6, 6, 6, 2, 2, 2, 7, 7, 7, 7, 7, 7, 7, 1, 1, 1, 16, 1, 1, 1, 1, 9, 9, 9, 9, 9, 9, 9, 9, 9, 5, 5, 5, 5, 5, 5, 5, 5, 0, 0, 2, 2, 2, 2, 2, 7, 4, 4, 4, 4, 4, 4, 4, 12, 12, 12, 12, 12, 12, 12, 12, 12, 12, 12, 12, 12, 12, 12, 12, 12, 12, 12, 12, 12, 12, 12, 12, 9, 5, 5, 5, 5, 5, 5, 5, 13, 13, 13, 13, 13, 13, 13, 13, 13, 13, 13, 13, 13, 13, 13, 13, 16, 11, 11, 11, 11, 11, 11, 11, 3, 0, 0, 0, 0, 0, 0, 0, 16, 16, 16, 16, 16, 16, 16, 16, 8, 0, 0, 0, 0, 0, 0, 0, 6, 6, 6, 6, 6, 6, 6, 2, 2, 2, 2, 2, 2, 2, 2, 10, 14, 14, 14, 14, 14, 14, 14, 6, 6, 6, 6]
    sample_actions = [ale_py.Action(i) for i in sample_actions]
    gt_actions = [ale_py.Action(i) for i in gt_actions]

    latency_wrapper = LatencyModel(".")
    total = 0
    correct = 0
    for i in range(0, len(gt_actions)):
        a = latency_wrapper.act(sample_actions[i])
        print("\t Input action:", sample_actions[i], "\tExecuting action:", a, "\tGround truth action", gt_actions[i])
        if i > 200:
            if a == gt_actions[i]:
                correct += 1
            total += 1
    print("Accuracy", correct / total)
