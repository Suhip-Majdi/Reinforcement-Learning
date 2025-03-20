import pygame
import sys
import numpy as np
import random

class GridWorld:
    def __init__(self, grid_size=8, cell_size=75):
        self.grid_size = grid_size
        self.cell_size = cell_size
        self.screen_width = self.grid_size * self.cell_size
        self.screen_height = self.grid_size * self.cell_size

        # Colors we need to draw the grid
        self.white = (255, 255, 255)
        self.black = (0, 0, 0)

        pygame.init()
        self.screen = pygame.display.set_mode((self.screen_width, self.screen_height))
        pygame.display.set_caption("Q-Table")

        # Images for the env,
        self.agent_image = pygame.image.load("agent.png")
        self.agent_image = pygame.transform.scale(self.agent_image, (self.cell_size, self.cell_size))
        self.bomb_image = pygame.image.load("bomb.png")
        self.bomb_image = pygame.transform.scale(self.bomb_image, (self.cell_size, self.cell_size))
        self.goal_image = pygame.image.load("goal.png")
        self.goal_image = pygame.transform.scale(self.goal_image, (self.cell_size, self.cell_size))

        self.agent_pos = [0, 0]
        self.goal_pos = [self.grid_size - 1, self.grid_size - 1]

        self.bomb_positions = [
            [1, 1],
            [2, 3],
            [4, 4],
            [5, 6],
            [6, 2],
            [3, 7],
            [0, 6],
            [7, 4]
        ]

    def draw_grid(self):
        for x in range(0, self.screen_width, self.cell_size):
            for y in range(0, self.screen_height, self.cell_size):
                rect = pygame.Rect(x, y, self.cell_size, self.cell_size)
                pygame.draw.rect(self.screen, self.black, rect, 1)  # Draw grid lines

    def draw_objects(self):
        agent_x, agent_y = self.agent_pos[0] * self.cell_size, self.agent_pos[1] * self.cell_size
        self.screen.blit(self.agent_image, (agent_x, agent_y))

        goal_x, goal_y = self.goal_pos[0] * self.cell_size, self.goal_pos[1] * self.cell_size
        self.screen.blit(self.goal_image, (goal_x, goal_y))

        for bomb in self.bomb_positions:
            bomb_x, bomb_y = bomb[0] * self.cell_size, bomb[1] * self.cell_size
            self.screen.blit(self.bomb_image, (bomb_x, bomb_y))

    def step(self, action):
        x, y = self.agent_pos

        if action == 0 and y > 0:  # up
            y -= 1
        elif action == 1 and y < self.grid_size - 1:  # down
            y += 1
        elif action == 2 and x > 0:  # left
            x -= 1
        elif action == 3 and x < self.grid_size - 1:  # right
            x += 1

        self.agent_pos = [x, y]

        if self.agent_pos == self.goal_pos:
            return self.agent_pos, 10, True
        elif self.agent_pos in self.bomb_positions:
            return self.agent_pos, -10, True
        else:
            return self.agent_pos, -1, False

    def reset(self):
        self.agent_pos = [0, 0]
        return self.agent_pos

def train_q_learning(grid_world, episodes=1000, alpha=0.1, gamma=0.9, epsilon=1.0, epsilon_decay=0.995, min_epsilon=0.1, render_every=20):
    state_space = grid_world.grid_size * grid_world.grid_size
    action_space = 4
    Q_table = np.zeros((state_space, action_space))

    def state_to_index(state):
        return state[0] * grid_world.grid_size + state[1] # row * grid size + col  # from 2D to 1D

    for episode in range(episodes):
        state = grid_world.reset()
        done = False
        total_reward = 0

        while not done:
            state_idx = state_to_index(state)

            if random.uniform(0, 1) < epsilon:
                action = random.randint(0, 3)  # Explore
            else:
                action = np.argmax(Q_table[state_idx])  # Exploit

            new_state, reward, done = grid_world.step(action)
            new_state_idx = state_to_index(new_state)
            total_reward += reward

            # update the table => Bellman Equation
            Q_table[state_idx, action] = Q_table[state_idx, action] + alpha * (
                reward + gamma * np.max(Q_table[new_state_idx]) - Q_table[state_idx, action]
            )

            state = new_state

            # Render after some episodes
            if episode % render_every == 0:
                grid_world.screen.fill(grid_world.white)
                grid_world.draw_grid()
                grid_world.draw_objects()
                pygame.display.flip()
                pygame.time.delay(200)

        epsilon = max(min_epsilon, epsilon * epsilon_decay)

        if episode % render_every == 0:
            print(f"Episode {episode}: Total Reward: {total_reward}, Epsilon: {epsilon:.2f}")

    print("Training completed!")
    return Q_table

if __name__ == "__main__":
    grid_world = GridWorld()

    Q_table = train_q_learning(grid_world, episodes=2000, render_every=20)

    state = grid_world.reset()
    done = False
    print("\n Testing the agent ...")
    while not done:
        grid_world.screen.fill(grid_world.white)
        grid_world.draw_grid()
        grid_world.draw_objects()

        state_idx = state[0] * grid_world.grid_size + state[1]
        action = np.argmax(Q_table[state_idx])                   # مشان يوخذ اكبر قيمة q لالتيبل من ال (up, down, right, left) من هدول زي السلايد

        state, _, done = grid_world.step(action)

        pygame.display.flip()
        pygame.time.delay(500)

    pygame.quit()
    sys.exit()
