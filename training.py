import numpy as np
import tensorflow as tf
import time

# Define the maze matrix
maze = np.array([
    [1, 1, 1, 1, 1],
    [0, 1, 0, 0, 1],
    [1, 1, 1, 1, 1],
    [1, 0, 0, 1, 1],
    [1, 1, 1, 1, 1]
])

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
)


# Define the start and end coordinates
start = (0, 0)
end = (len(maze) - 1, len(maze[0]) - 1)

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
        self.epsilon_decay = 0.999  # Exploration rate decay
        self.epsilon_min = 0.01  # Minimum exploration rate
        self.gamma = 0.99  # Discount factor
        self.learning_rate = 0.001
        self.model = DQN(self.num_actions)
        self.optimizer = tf.keras.optimizers.Adam(learning_rate=self.learning_rate)

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

            end_time = time.time()  # End the timer
            period_time = end_time - start_time  # Calculate the time period
            
            # Decay exploration rate
            if self.epsilon > self.epsilon_min:
                self.epsilon *= self.epsilon_decay
            
            # Print episode statistics
            print(f"Episode: {episode+1}, Reward: {episode_reward}, Epsilon: {self.epsilon:.3f}, Time: {period_time:.3f}")

            # Reset the maze
            state = np.array(self.start)

            # If agent hasen't learn good stuff, reset epsilon
            self.epsilon = 1.0 if period_time > self.time_limit_to_reset_epsilon else self.epsilon

            # If agent has learned good stuff, change maze
            if episode_reward > self.succesful_run_trigger: # ADD AT THE START
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
            x = min(x + 1, maze.shape[0] - 1)
        elif action == 2:  # Left
            y = max(y - 1, 0)
        elif action == 3:  # Right
            y = min(y + 1, maze.shape[1] - 1)
        
        next_state = np.array([x, y])
        #Print goal when agent reaches the end
        """
        if tuple(next_state) == self.end:
            print("Goal Reached")
            print (next_state)
            print (self.end)
        """
        reward = 1 if tuple(next_state) == self.end else -1 #if self.maze[x, y] == 0 else 0
        done = tuple(next_state) == self.end #or self.maze[x, y] == 0 # Why would it reset, if it hits a wall? it can not do a perfect run, if it does not hit a wall
        next_state = state if self.maze[x, y] == 0 else next_state
        
        return next_state, reward, done
    
    def update_model(self, replay_buffer, batch_size):
        batch_indices = np.random.choice(len(replay_buffer), batch_size, replace=False)
        states, actions, rewards, next_states, dones = zip(*[replay_buffer[i] for i in batch_indices])
        
        states = np.array(states)
        actions = np.array(actions)
        rewards = np.array(rewards)
        next_states = np.array(next_states)
        dones = np.array(dones)
        
        with tf.GradientTape() as tape:
            current_q_values = tf.reduce_sum(self.model(states) * tf.one_hot(actions, self.num_actions), axis=1)
            
            next_q_values = tf.reduce_max(self.model(next_states), axis=1)
            target_q_values = rewards + (1 - dones) * self.gamma * next_q_values
            
            loss = tf.reduce_mean(tf.square(current_q_values - target_q_values))
        
        gradients = tape.gradient(loss, self.model.trainable_variables)
        self.optimizer.apply_gradients(zip(gradients, self.model.trainable_variables))

# Create and train the DQN agent
agent = DQNAgent(maze, start, end)
agent.train(num_episodes=10000, batch_size=128)

# Wait user input to test the learned policy
input("Press enter to test the learned policy...")
print("\n" * 100)

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
loaded_model = tf.keras.models.load_model('maze_solver_model', compile=False) # ADD COMPILED FALSE TO THE EE

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

# Navigate the maze using the loaded model
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
    if count > 10:
        print("Agent is stuck. Stopping.  :(")
        break
    last_position = tuple(state)