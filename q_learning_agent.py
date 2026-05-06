import torch
import random
import numpy as np

class QLearningAgent:
    def __init__(self, actions, width=5, height=5):
        # 행동 = [0, 1, 2, 3] 순서대로 상, 하, 좌, 우
        self.actions = actions
        self.learning_rate = 0.01
        self.discount_factor = 0.9
        
        # 입실론 설정: 0.3 정도로 시작하거나 감쇠(decay) 사용
        self.epsilon = 0.3
        self.epsilon_decay = 0.99
        self.epsilon_min = 0.01

        # Q-table을 PyTorch Tensor로 초기화 (상태 수 x 행동 수)
        self.width = width
        self.height = height
        self.q_table = torch.zeros((width * height, len(actions)), dtype=torch.float32)

    # 입실론 감쇠
    def decay_epsilon(self):
        if self.epsilon > self.epsilon_min:
            self.epsilon *= self.epsilon_decay

    def state_to_index(self, state):
        return int(state[0] + (state[1] * self.width))

    def save_model(self, file_path):
        torch.save(self.q_table, file_path)
        print(f"Model saved to {file_path}")

    def load_model(self, file_path):
        self.q_table = torch.load(file_path)
        print(f"Model loaded from {file_path}")

    def learn(self, state, action, reward, next_state):
        state_idx = self.state_to_index(state)
        next_state_idx = self.state_to_index(next_state)

        q_1 = self.q_table[state_idx][action]
        q_2 = reward + self.discount_factor * torch.max(self.q_table[next_state_idx])
        self.q_table[state_idx][action] += self.learning_rate * (q_2 - q_1)

    def get_action(self, state):
        if np.random.rand() < self.epsilon:
            action = np.random.choice(self.actions)
        else:
            state_idx = self.state_to_index(state)
            state_action = self.q_table[state_idx]
            action = self.arg_max(state_action)
        return int(action)

    @staticmethod
    def arg_max(state_action):
        max_value = torch.max(state_action)
        max_indices = (state_action == max_value).nonzero().flatten()
        return random.choice(max_indices.tolist())
