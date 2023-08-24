import numpy as np
import tensorflow as tf
import time
import os
import datetime

# Define the maze matrix

mazes = (
    np.array([
        [1, 1, 1, 1, 1],
        [0, 0, 1, 0, 1],
        [1, 1, 1, 1, 1],
        [1, 0, 0, 0, 0],
        [1, 1, 1, 1, 1]
    ]),
    np.array([
        [1, 1, 1, 1, 1],
        [0, 0, 0, 1, 1],
        [1, 0, 1, 1, 1],
        [1, 0, 0, 1, 0],
        [1, 1, 1, 1, 1]
    ]),
    np.array([
        [1, 1, 1, 1, 1],
        [0, 1, 0, 0, 1],
        [1, 1, 1, 1, 1],
        [1, 0, 0, 1, 0],
        [1, 1, 1, 1, 1]
    ]),
    np.array([
        [1, 0, 1, 1, 1],
        [1, 0, 1, 0, 1],
        [1, 1, 1, 0, 1],
        [1, 0, 0, 0, 1],
        [1, 1, 0, 1, 1]
    ]),
    np.array([
        [1, 1, 1, 1, 1],
        [0, 1, 0, 0, 1],
        [1, 1, 1, 0, 1],
        [1, 0, 0, 1, 0],
        [1, 1, 0, 1, 1]
    ]),
    np.array([
        [1, 1, 1, 1, 1],
        [0, 1, 0, 0, 1],
        [1, 1, 1, 1, 1],
        [1, 0, 0, 1, 1],
        [1, 1, 1, 1, 1]
    ])
)
# select a random maze from the list above for maze
maze = mazes[np.random.randint(0, len(mazes))]



# Define the start and end coordinates
start = (0, 0)
end = (len(maze) - 1, len(maze[0]) - 1)


# Define a wrapper class for DQNAgent
class DQNAgentWrapper(tf.Module):
    def __init__(self, agent):
        super(DQNAgentWrapper, self).__init__()
        self.agent = agent


# Define the Deep Q-Network


class DQN(tf.keras.Model):
    def __init__(self, num_actions):
        super(DQN, self).__init__()
        self.dense1 = tf.keras.layers.Dense(32, activation='relu')
        self.dense2 = tf.keras.layers.Dense(32, activation='relu')
        self.dense3 = tf.keras.layers.Dense(num_actions)

    def call(self, inputs):
        x = self.dense1(inputs)
        x = self.dense2(x)
        x = self.dense3(x)
        return x

# Define the Deep Q-Network Agent


class DQNAgent:
    def __init__(self, maze, start, end):
        # Default settings
        self.maze = maze
        self.start = start
        self.end = end
        self.num_actions = 4  # Up, Down, Left, Right
        self.epsilon = 1.0  # Exploration rate
        self.epsilon_decay = 0.995  # Exploration rate decay
        self.epsilon_min = 0.01  # Minimum exploration rate
        self.gamma = 0.99  # Discount factor
        self.learning_rate = 0.001
        self.model = DQN(self.num_actions)
        self.optimizer = tf.keras.optimizers.Adam(
            learning_rate=self.learning_rate)

        # My additional settings
        self.succesful_run_trigger = -40
        self.times_succesful_to_next_map = 40
        self.time_limit_to_reset_epsilon = 40

    def get_action(self, state):
        if np.random.rand() <= self.epsilon:
            return np.random.randint(self.num_actions)
        else:
            q_values = self.model.predict(state[np.newaxis])
            return np.argmax(q_values)

    def train(self, num_episodes, batch_size):
        replay_buffer = []
        state = np.array(self.start)
        maze_counter = 0

        for episode in range(num_episodes):
            episode_reward = 0
            done = False

            start_time = time.time()  # Start the timer

            while not done:
                action = self.get_action(state)
                next_state, reward, done = self.take_action(state, action)
                episode_reward += reward

                replay_buffer.append((state, action, reward, next_state, done))
                state = next_state

                if len(replay_buffer) > batch_size:
                    self.update_model(replay_buffer, batch_size)
                    
            # Backing up the model
            if episode % backup_frequency == 0:
                checkpoint_path = checkpoint_manager.save()
                # Save metadata along with the checkpoint
                metadata = f"Episode: {episode+1}, Time: {datetime.datetime.now()}"
                metadata_path = checkpoint_path + '.meta'
                with open(metadata_path, 'w') as file:
                    file.write(metadata)

            end_time = time.time()  # End the timer
            period_time = end_time - start_time  # Calculate the time period

            # Decay exploration rate
            if self.epsilon > self.epsilon_min:
                self.epsilon *= self.epsilon_decay

            # Print episode statistics
            print(
                f"Episode: {episode+1}, Reward: {episode_reward}, Epsilon: {self.epsilon:.3f}, Time: {period_time:.3f}")

            # Reset the maze
            state = np.array(self.start)

            # If agent hasen't learn good stuff, reset epsilon
            self.epsilon = 1.0 if (period_time > self.time_limit_to_reset_epsilon and self.epsilon < 0.600) else self.epsilon

            # If agent has learned good stuff, change maze
            if episode_reward > self.succesful_run_trigger:  # ADD AT THE START
                maze_counter += 1
            else:
                maze_counter = 0

            if maze_counter == self.times_succesful_to_next_map:
                print("Changing maze, resseting epsilon...")
                maze_counter = 0
                self.maze = mazes[np.random.randint(0, len(mazes))]
                print("Changed maze")
                self.start = (0, 0)
                print("Changed start")
                self.end = (len(self.maze) - 1, len(self.maze[0]) - 1)
                print("Changed end")
                self.epsilon = 1.0
                print("Changed epsilon")
                print("Starting training again...")

    def take_action(self, state, action):
        x, y = state

        if action == 0:  # Up
            x = max(x - 1, 0)
        elif action == 1:  # Down
            x = min(x + 1, self.maze.shape[0] - 1)
        elif action == 2:  # Left
            y = max(y - 1, 0)
        elif action == 3:  # Right
            y = min(y + 1, self.maze.shape[1] - 1)

        next_state = np.array([x, y])
        # Print goal when agent reaches the end
        """
        if tuple(next_state) == self.end:
            print("Goal Reached")
            print (next_state)
            print (self.end)
        """
        reward = 1 if tuple(next_state) == self.end else - \
            1  # if self.maze[x, y] == 0 else 0
        # or self.maze[x, y] == 0 # Why would it reset, if it hits a wall? it can not do a perfect run, if it does not hit a wall
        done = tuple(next_state) == self.end
        next_state = state if self.maze[x, y] == 0 else next_state

        return next_state, reward, done


    def update_model(self, replay_buffer, batch_size):
        batch_indices = np.random.choice(
            len(replay_buffer), batch_size, replace=False)
        states, actions, rewards, next_states, dones = zip(
            *[replay_buffer[i] for i in batch_indices])

        states = np.array(states)
        actions = np.array(actions)
        rewards = np.array(rewards)
        next_states = np.array(next_states)
        dones = np.array(dones)

        with tf.GradientTape() as tape:
            current_q_values = tf.reduce_sum(self.model(
                states) * tf.one_hot(actions, self.num_actions), axis=1)

            next_q_values = tf.reduce_max(self.model(next_states), axis=1)
            target_q_values = rewards + \
                (1 - dones) * self.gamma * next_q_values

            loss = tf.reduce_mean(
                tf.square(current_q_values - target_q_values))

        gradients = tape.gradient(loss, self.model.trainable_variables)
        self.optimizer.apply_gradients(
            zip(gradients, self.model.trainable_variables))


# Create and train the DQN agent
agent = DQNAgent(maze, start, end)

# BACKUPS
# Define the checkpoint directory
checkpoint_dir = './checkpoints'
os.makedirs(checkpoint_dir, exist_ok=True)

# Create a wrapper object for the agent
agent_wrapper = DQNAgentWrapper(agent)

# Create a checkpoint object for the agent wrapper
checkpoint = tf.train.Checkpoint(agent=agent_wrapper)

# Define a checkpoint manager
checkpoint_manager = tf.train.CheckpointManager(
    checkpoint, checkpoint_dir, max_to_keep=20)

# Backup frequency
backup_frequency = 100

# Check if a checkpoint exists
latest_checkpoint = checkpoint_manager.latest_checkpoint
if latest_checkpoint is not None:
    # Prompt the user if they want to use the latest checkpoint or select another checkpoint
    choice = input("A saved checkpoint exists. Do you want to use it? (y/n): ")
    if choice.lower() == 'y':
        checkpoint.restore(latest_checkpoint)
        print(f"Restored from checkpoint: {latest_checkpoint}")
        # Add code to retrieve and display metadata if available
        metadata_path = latest_checkpoint + '.meta'
        if os.path.exists(metadata_path):
            with open(metadata_path, 'r') as file:
                metadata = file.read()
                print(f"Metadata: {metadata}")
    else:
        # Show a list of available checkpoints
        print("Available checkpoints:")
        checkpoint_list = checkpoint_manager.checkpoints
        metadata_list = []

        for i, cp in enumerate(checkpoint_list):
            metadata_path = cp + '.meta'
            if os.path.exists(metadata_path):
                with open(metadata_path, 'r') as file:
                    metadata = file.read()
                    metadata_list.append(metadata)
                    print(f"{i+1}. {cp} - {metadata}")
            else:
                metadata_list.append('')
                print(f"{i+1}. {cp}")

        # Prompt the user to select a checkpoint
        selected_checkpoint = input(
            "Enter the number of the checkpoint you want to revert to (0 to start from scratch): ")
        if selected_checkpoint.isdigit() and 0 <= int(selected_checkpoint) <= len(checkpoint_list):
            if int(selected_checkpoint) == 0:
                print("Starting from scratch.")
            else:
                checkpoint.restore(
                    checkpoint_list[int(selected_checkpoint) - 1])
                print(
                    f"Restored from checkpoint: {checkpoint_list[int(selected_checkpoint) - 1]}")
                selected_metadata = metadata_list[int(selected_checkpoint) - 1]
                if selected_metadata:
                    print(f"Metadata: {selected_metadata}")
        else:
            print("Invalid selection. Starting from scratch.")
else:
    print("No checkpoints found. Starting from scratch.")

print("DONT PANIC - THE AGENT IS TRAINING (this may take a while, for me up to 5 minutes to see any output))")

agent.train(num_episodes=10000, batch_size=128)

# Wait user input to test the learned policy
input("Press enter to test the learned policy...")


#changing the maze to hardcore one
agent.maze = np.array([
        [1, 1, 1, 1, 1, 1],
        [0, 0, 0, 1, 0, 1],
        [1, 1, 1, 1, 0, 1],
        [1, 0, 0, 1, 0, 1],
        [1, 0, 1, 0, 1, 0],
        [1, 1, 1, 1, 1, 1]
    ])



# Test the learned policy
state = np.array(agent.start)

while tuple(state) != agent.end:
    action = agent.get_action(state)
    next_state, _, _ = agent.take_action(state, action)
    state = next_state

    print(f"Agent Position: {tuple(state)}")

# After training, save the model
tf.keras.models.save_model(agent.model, 'maze_solver_model')


# TEST THE MODEL


# Load the saved model
loaded_model = tf.keras.models.load_model(
    'maze_solver_model', compile=False)  # ADD COMPILED FALSE TO THE EE

# Print the model summary
if loaded_model is None:
    print("Failed to load the model.")
else:
    # Print the model summary
    loaded_model.summary()

# Create a new instance of the DQNAgent class
test_agent = DQNAgent(maze, start, end)

# Set the initial state
state = np.array(test_agent.start)
last_position = None
count = 0

# Set the exploration rate for testing
epsilon = 0.0  # or a small value if you want to allow some exploration

# Navigate the maze using the loaded model with exploration
while tuple(state) != test_agent.end:
    q_values = loaded_model.predict(state[np.newaxis])
    action = np.argmax(q_values)
    next_state, _, _ = test_agent.take_action(state, action)
    state = next_state

    print(f"Agent Position: {tuple(state)}")

    # Check if the agent is stuck
    if last_position == tuple(state):
        count += 1
    else:
        count = 0
    if count > 100:
        print("Agent is stuck. Stopping.  :(")
        break
    last_position = tuple(state)
