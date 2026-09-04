import argparse
import itertools
import os
import random
import sys
from dqn import DQN
from experience_replay import ReplayMemory
import flappy_bird_gymnasium
import gymnasium as gym
import torch
import torch.nn as nn
import torch.optim as optim
import yaml

if torch.backends.mps.is_available():
    device = "mps"
elif torch.cuda.is_available():
    device = "cuda"
else:
    device = "cpu"

RUNS_DIR = "runs"
os.makedirs(RUNS_DIR, exist_ok=True)


class Agent:

    def __init__(self, param_set):
        self.param_set = param_set

        with open("parameters.yaml", "r") as f:
            all_param_set = yaml.safe_load(f)
            params = all_param_set[param_set]

        self.alpha = params["alpha"]
        self.gamma = params["gamma"]

        self.epsilon_init = params["epsilon_init"]
        self.epsilon_min = params["epsilon_min"]
        self.epsilon_decay = params["epsilon_decay"]

        self.replay_memory_size = params["replay_memory_size"]
        self.mini_batch_size = params["mini_batch_size"]
        self.network_sync_rate = params["network_sync_rate"]

        self.reward_threshold = params["reward_threshold"]

        self.loss_fn = nn.MSELoss()
        self.optimizer = None

        self.LOG_FILE = os.path.join(RUNS_DIR, f"{param_set}.log")
        self.MODEL_FILE = os.path.join(RUNS_DIR, f"{param_set}.pt")

    def run(self, is_training=True, render=False):
        env = gym.make(
            "FlappyBird-v0", render_mode="human" if render else None
        )

        num_states = env.observation_space.shape[0]
        num_actions = env.action_space.n

        policy_dqn = DQN(state_dim=num_states, action_dim=num_actions).to(
            device
        )

        if is_training:
            memory = ReplayMemory(maxlen=self.replay_memory_size)
            epsilon = self.epsilon_init

            target_dqn = DQN(state_dim=num_states, action_dim=num_actions).to(
                device
            )
            target_dqn.load_state_dict(policy_dqn.state_dict())

            steps = 0
            self.optimizer = optim.Adam(policy_dqn.parameters(), lr=self.alpha)
            best_reward = float("-inf")
        else:
            policy_dqn.load_state_dict(
                torch.load(self.MODEL_FILE, map_location=device)
            )
            policy_dqn.eval()

        # FIX: episode counter loop handle karein (episode = 0 andandar se hata diya)
        for episode in itertools.count():
            state, _ = env.reset()
            state = torch.tensor(state, dtype=torch.float).to(device)

            terminated = False
            episode_rewards = 0

            while (
                not terminated and episode_rewards < self.reward_threshold
            ):
                if is_training and random.random() < epsilon:
                    action = env.action_space.sample()
                    action = torch.tensor(action, dtype=torch.long).to(device)
                else:
                    with torch.no_grad():
                        action = (
                            policy_dqn(state.unsqueeze(dim=0))
                            .squeeze(0)
                            .argmax()
                        )

                next_state, reward, terminated, _, _ = env.step(action.item())

                next_state = torch.tensor(next_state, dtype=torch.float).to(
                    device
                )
                reward_tensor = torch.tensor(reward, dtype=torch.float).to(
                    device
                )

                if is_training:
                    # Order: state, action, reward, next_state, terminated
                    memory.append(
                        (
                            state,
                            action,
                            reward_tensor,
                            next_state,
                            terminated,
                        )
                    )
                    steps += 1

                state = next_state
                episode_rewards += reward

            print(
                f"For Episode {episode+1} finished with reward {episode_rewards}"
            )

            if is_training:
                # Epsilon decay
                epsilon = max(epsilon * self.epsilon_decay, self.epsilon_min)

                if episode_rewards > best_reward:
                    log_message = (
                        f"best reward: {best_reward} -> {episode_rewards}"
                    )
                    with open(self.LOG_FILE, "a") as f:
                        f.write(log_message + "\n")

                    torch.save(policy_dqn.state_dict(), self.MODEL_FILE)
                    best_reward = episode_rewards

                # Optimize step
                if len(memory) >= self.mini_batch_size:
                    # FIX: ReplayMemory instance method or sampling fix
                    if hasattr(memory, "sample"):
                        mini_batch = memory.sample(self.mini_batch_size)
                    else:
                        mini_batch = random.sample(
                            memory.memory, k=self.mini_batch_size
                        )

                    self.optimize(mini_batch, policy_dqn, target_dqn)

                    if steps > self.network_sync_rate:
                        target_dqn.load_state_dict(policy_dqn.state_dict())
                        steps = 0

    # FIX: optimize function ko Class ke bahar se Class method banaya
    def optimize(self, mini_batch, policy_dqn, target_dqn):
        # Tuple Order Matched: (state, action, reward, next_state, terminated)
        states, actions, rewards, next_states, terminations = zip(*mini_batch)

        states = torch.stack(states)
        actions = torch.stack(actions)
        next_states = torch.stack(next_states)
        rewards = torch.stack(rewards)
        terminations = torch.tensor(terminations).float().to(device)

        with torch.no_grad():
            target_q = (
                rewards
                + (1 - terminations)
                * self.gamma
                * target_dqn(next_states).max(dim=1)[0]
            )

        current_q = (
            policy_dqn(states).gather(1, actions.unsqueeze(1)).squeeze(1)
        )

        loss = self.loss_fn(current_q, target_q)

        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Train or test a DQN agent on Flappy Bird."
    )
    parser.add_argument(
        "hyperparameters", help="YAML section key name e.g. flappybirdv0"
    )
    parser.add_argument("--train", help="Training mode", action="store_true")
    args = parser.parse_args()

    print(">>> Script execution started...")
    print(f">>> Hyperparameter passed: {args.hyperparameters}")
    print(f">>> Training mode: {args.train}")

    try:
        print(">>> Initializing Agent...")
        dql = Agent(param_set=args.hyperparameters)

        if args.train:
            print(">>> Calling dql.run(is_training=True)...")
            dql.run(is_training=True, render=False)
        else:
            print(">>> Calling dql.run(is_training=False)...")
            dql.run(is_training=False, render=True)

    except Exception as e:
        print(f"\n[ERROR] An error occurred during initialization/training:")
        print(e)
        import traceback

        traceback.print_exc()