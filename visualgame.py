import pygame
import numpy as np
import tensorflow as tf

# Define the maze matrix
maze = np.array([
    [1, 1, 1, 1, 1],
    [0, 0, 0, 0, 1],
    [1, 1, 1, 1, 1],
    [1, 0, 0, 0, 0],
    [1, 1, 1, 1, 1]
])

# Define the start and end coordinates
start = (0, 0)
end = (len(maze) - 1, len(maze[0]) - 1)

# Load the saved model
loaded_model = tf.keras.models.load_model('maze_solver_model')

# Initialize Pygame
pygame.init()

# Set the dimensions of the game window
window_width = 600
window_height = 600

# Set the dimensions of the maze grid
grid_size = min(window_width, window_height) // len(maze)

# Set colors
black = (0, 0, 0)
white = (255, 255, 255)
green = (0, 255, 0)
red = (255, 0, 0)

# Create the game window
window = pygame.display.set_mode((window_width, window_height))
pygame.display.set_caption("Maze Solver")

clock = pygame.time.Clock()

def draw_maze():
    window.fill(white)
    
    for i in range(len(maze)):
        for j in range(len(maze[0])):
            rect = pygame.Rect(j * grid_size, i * grid_size, grid_size, grid_size)
            if maze[i][j] == 0:  # Wall
                pygame.draw.rect(window, black, rect)
            elif maze[i][j] == 1:  # Walkable path
                pygame.draw.rect(window, white, rect)
    
    # Draw the start position
    pygame.draw.rect(window, green, (start[1] * grid_size, start[0] * grid_size, grid_size, grid_size))
    
    # Draw the end position
    pygame.draw.rect(window, red, (end[1] * grid_size, end[0] * grid_size, grid_size, grid_size))
    
def draw_agent(position):
    pygame.draw.circle(window, (0, 0, 255), (position[1] * grid_size + grid_size // 2, position[0] * grid_size + grid_size // 2), grid_size // 4)

# Test the learned policy
state = np.array(start)

# Initialize the last action
last_position = None
count = 0

running = True
while running:
    for event in pygame.event.get():
        if event.type == pygame.QUIT:
            running = False
    
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
    
    draw_maze()
    draw_agent(tuple(state))
    pygame.display.flip()
    
    # Check if the agent has reached the end
    if tuple(state) == end:
        print("Agent reached the end! :)")
        break
    
    # Check if the agent is stuck
    if last_position == tuple(state):
        count += 1
    else:
        count = 0
    if count > 3:
        print("Agent is stuck. Stopping. :(")
        break
    last_position = tuple(state)
    
    clock.tick(10)

pygame.quit()
