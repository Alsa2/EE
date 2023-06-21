import numpy as np
from scipy.spatial import KDTree
import matplotlib.pyplot as plt

def is_collision_free(point1, point2, matrix):
    # Check if the line segment between two points is collision-free
    line = np.linspace(point1, point2, int(np.linalg.norm(np.array(point2) - np.array(point1)))+1).astype(int)
    for point in line:
        x, y = point
        if matrix[x, y] == 0:  # Collision detected
            return False
    return True

def generate_roadmap(matrix, num_samples, k):
    # Generate a roadmap using PRMs
    height, width = matrix.shape
    roadmap = []
    while len(roadmap) < num_samples:
        x = np.random.randint(0, height)
        y = np.random.randint(0, width)
        if matrix[x, y] == 0:  # Ignore walls
            continue
        roadmap.append((x, y))

    # Construct a KD-tree for efficient nearest neighbor search
    kdtree = KDTree(roadmap)

    # Connect neighboring nodes in the roadmap
    connections = {}
    for i, point in enumerate(roadmap):
        _, indices = kdtree.query(point, k=k+1)  # Find k nearest neighbors
        neighbors = [roadmap[ind] for ind in indices[1:] if ind != i]  # Exclude the point itself
        connections[point] = []
        for neighbor in neighbors:
            if is_collision_free(point, neighbor, matrix):
                connections[point].append(neighbor)

    return roadmap, connections

def find_paths(matrix, start, goal, roadmap, connections):
    # Find all possible paths using DFS algorithm
    stack = [(start, [start])]
    paths = []
    
    while stack:
        current, path = stack.pop()
        
        if current == goal:
            paths.append(path)
        
        for neighbor in connections[current]:
            if neighbor not in path:
                stack.append((neighbor, path + [neighbor]))
    
    return paths

def find_shortest_path(matrix, start, goal, roadmap, connections):
    # Find the shortest path using A* algorithm
    open_set = {start}
    closed_set = set()
    g_score = {start: 0}
    f_score = {start: np.linalg.norm(np.array(goal) - np.array(start))}
    came_from = {}

    while open_set:
        current = min(open_set, key=lambda x: f_score[x])
        if current == goal:
            path = [current]
            while current in came_from:
                current = came_from[current]
                path.append(current)
            return path[::-1]

        open_set.remove(current)
        closed_set.add(current)

        for neighbor in connections[current]:
            if neighbor in closed_set:
                continue
            tentative_g_score = g_score[current] + np.linalg.norm(np.array(neighbor) - np.array(current))
            if neighbor not in open_set or tentative_g_score < g_score[neighbor]:
                came_from[neighbor] = current
                g_score[neighbor] = tentative_g_score
                f_score[neighbor] = tentative_g_score + np.linalg.norm(np.array(goal) - np.array(neighbor))
                if neighbor not in open_set:
                    open_set.add(neighbor)

    return None

def plot_matrix(matrix):
    plt.imshow(matrix.T, cmap='binary', origin='lower')
    plt.xticks([])
    plt.yticks([])
    plt.show()

def plot_path(matrix, path, color='red'):
    if path is None:
        print("No path found.")
    else:
        plt.imshow(matrix.T, cmap='binary', origin='lower')
        plt.plot(*zip(*path), color=color, linewidth=2)
        plt.xticks([])
        plt.yticks([])
        plt.show()

# Example usage
matrix = np.array([[1, 1, 1, 1, 1, 1],
                   [1, 0, 0, 1, 1, 1],
                   [1, 1, 1, 1, 0, 1],
                   [1, 0, 0, 0, 1, 1],
                   [1, 1, 1, 1, 1, 1]])
start = (0, 0)
goal = (0, 5)
num_samples = 50
k = 6

roadmap, connections = generate_roadmap(matrix, num_samples, k)
shortest_path = find_shortest_path(matrix, start, goal, roadmap, connections)
all_paths = find_paths(matrix, start, goal, roadmap, connections)

plot_matrix(matrix)
plot_path(matrix, shortest_path, color='red')

print(all_paths)

for path in all_paths:
    if path != shortest_path:
        plot_path(matrix, path, color='blue')
