import numpy as np
import tensorflow as tf

# Define the maze matrix
maze = np.array([
    [1, 1, 1, 1, 1],
    [1, 1, 1, 1, 1],
    [1, 1, 1, 1, 1],
    [1, 1, 1, 1, 1],
    [1, 1, 1, 1, 1]
])

# Define the start and end coordinates
start = (0, 0)
end = (len(maze) - 1, len(maze[0]) - 1)

# Load the saved model
loaded_model = tf.keras.models.load_model('maze_solver_model')

# Test the learned policy
state = np.array(start)

while tuple(state) != end:
    action = np.argmax(loaded_model.predict(state[np.newaxis]))
    x, y = state
    
    if action == 0:  # Up
        x = max(x - 1, 0)
    elif action == 1:  # Down
        x = min(x + 1, maze.shape[0] - 1)
    elif action == 2:  # Left
        y = max(y - 1, 0)
    elif action == 3:  # Right
        y = min(y + 1, maze.shape[1] - 1)
    
    state = np.array([x, y])
    
    print(f"Agent Position: {tuple(state)}")

