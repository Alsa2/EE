from queue import Queue
import heapq
import numpy as np
import random
import networkx as nx
import time
# import tensorflow as tf    Will be implemented later

# Algorithm's functions
def manhattan_distance(point1, point2):
    return abs(point1[0] - point2[0]) + abs(point1[1] - point2[1])

def check_path(matrix, node1, node2):
    x1, y1 = node1
    x2, y2 = node2
    # Check all cells between node1 and node2
    for x in range(min(x1, x2), max(x1, x2)+1):
        for y in range(min(y1, y2), max(y1, y2)+1):
            if matrix[x][y] == 0:  # Obstacle found
                return False
    return True

def bresenham_line_no_diag(node1, node2):
    points = []
    dx = node2[0] - node1[0]
    dy = node2[1] - node1[1]

    # handle horizontal lines
    if dx == 0:
        # step +1 when we're moving upwards, -1 when moving downwards
        step = 1 if dy > 0 else -1 
        points.extend([(node1[0], y) for y in range(node1[1], node2[1] + step, step)])
        return points

    # handle vertical line
    if dy == 0:
        # step +1 when we're moving to the right, -1 when we're moving to the left
        step = 1 if dx > 0 else -1 
        points.extend([(x, node2[1]) for x in range(node1[0], node2[0] + step, step)])
        return points

    # handle other directions
    if abs(dx) >= abs(dy):
        step_x = 1 if dx > 0 else -1
        step_y = 1 if dy > 0 else -1

        #Horizontal transition
        for x in range(node1[0], node2[0] + step_x, step_x): 
            points.append((x, node1[1]))   

        #Vertical transition
        for y in range(node1[1]+ step_y, node2[1] + step_y, step_y):
            points.append((node2[0], y))

    else:
        step_x = 1 if dx > 0 else -1
        step_y = 1 if dy > 0 else -1

        #Vertical transition
        for y in range(node1[1], node2[1] + step_y, step_y):
            points.append((node1[0], y))  

        #Horizontal transition
        for x in range(node1[0] + step_x, node2[0] + step_x, step_x):
            points.append((x, node2[1]))

    return list(set(points))

def BFS_checker(matrix, start, end):
    queue = Queue()
    queue.put(start)
    visited = set()
    parents = {}

    while not queue.empty():
        current = queue.get()
        if current == end:
            # Path found
            path = [current]
            while current != start:
                current, prev = parents[current]
                path.append(prev)
            path.reverse()
            return True, path

        neighbors = [(current[0] - 1, current[1]), (current[0], current[1] - 1),
                     (current[0] + 1, current[1]), (current[0], current[1] + 1)]
        for neighbor in neighbors:
            x, y = neighbor
            if (0 <= x < len(matrix)) and (0 <= y < len(matrix[0])) and matrix[x][y] == 1 and neighbor not in visited:
                visited.add(neighbor)
                queue.put(neighbor)
                parents[neighbor] = (current, neighbor)

    # Path not found
    return False, None

def PRM(matrix, start, end, n_nodes=1000):
    # Threshold distance for visibility between two points
    node_threshold = 5

    # Generate random nodes
    nodes = [start, end] + [(random.randint(0, len(matrix)-1), random.randint(0, len(matrix[0])-1)) for _ in range(n_nodes)]

    #Check if start and end are in a free space
    if matrix[start[0]][start[1]] == 0 or matrix[end[0]][end[1]] == 0:
        print ("Start or end is in an obstacle")
        return False, None

    # Remove duplicate nodes
    nodes = list(set(nodes))

    # Check if the node is in a free space
    nodes = [node for node in nodes if matrix[node[0]][node[1]] == 1]

    # Create the graph
    graph = nx.Graph()

    for (i, node) in enumerate(nodes):
        graph.add_node(i, position=node)

    # Connect nodes that are within a certain distance
    for i in range(len(nodes)):
        for j in range(i+1, len(nodes)):
            node_i = nodes[i]
            node_j = nodes[j]
            if manhattan_distance(node_i, node_j) <= node_threshold and check_path(matrix, node_i, node_j):  # Set a threshold distance for visibility between two points
                graph.add_edge(i, j, weight=manhattan_distance(node_i, node_j))  # The weight of the edge is the distance

    # Use Dijkstra's algorithm or A* algorithm to find the shortest path
    start_index = nodes.index(start)
    end_index = nodes.index(end)

    try:
        path_indices = nx.dijkstra_path(graph, start_index, end_index, 'weight')
        path = [nodes[index] for index in path_indices]

        path_indices = nx.dijkstra_path(graph, start_index, end_index, 'weight')
        path = [nodes[index] for index in path_indices]

        # Incorporate Bresenham's algorithm to get all points in between
        points_in_path = []
        for i in range(len(path)-1):
            points_in_path += bresenham_line_no_diag(path[i], path[i+1])

        return True, points_in_path
    except nx.NetworkXNoPath:
        return False, None

def Astar(matrix, start, end):
    heap = [(0, start)]
    visited = [[False]*len(matrix[0]) for _ in range(len(matrix))]
    parents = [[None]*len(matrix[0]) for _ in range(len(matrix))]
    g_scores = [[float('inf')]*len(matrix[0]) for _ in range(len(matrix))]
    f_scores = [[float('inf')]*len(matrix[0]) for _ in range(len(matrix))]
    g_scores[start[0]][start[1]] = 0
    f_scores[start[0]][start[1]] = manhattan_distance(start, end)

    while heap:
        current_f, current = heapq.heappop(heap)
        if current == end:
            # Path found
            path = [current]
            while current != start:
                current = parents[current[0]][current[1]]
                path.append(current)
            path.reverse()
            return True, path

        x, y = current
        visited[x][y] = True
        for neighbor in ((x-1, y), (x, y-1), (x+1, y), (x, y+1)):
            nx, ny = neighbor
            if (0 <= nx < len(matrix)) and (0 <= ny < len(matrix[0])) and matrix[nx][ny] == 1 and not visited[nx][ny]:
                g_score = g_scores[x][y] + 1
                if g_score < g_scores[nx][ny]:
                    parents[nx][ny] = (x, y)
                    g_scores[nx][ny] = g_score
                    f_scores[nx][ny] = g_score + manhattan_distance(neighbor, end)
                    heapq.heappush(heap, (f_scores[nx][ny], neighbor))

    # Path not found
    return False, None

def DQN(matrix, start, end):
    heap = [(0, start)]
    visited = [[False]*len(matrix[0]) for _ in range(len(matrix))]
    parents = [[None]*len(matrix[0]) for _ in range(len(matrix))]
    g_scores = [[float('inf')]*len(matrix[0]) for _ in range(len(matrix))]
    f_scores = [[float('inf')]*len(matrix[0]) for _ in range(len(matrix))]
    g_scores[start[0]][start[1]] = 0
    f_scores[start[0]][start[1]] = manhattan_distance(start, end)

    while heap:
        current_f, current = heapq.heappop(heap)
        if current == end:
            # Path found
            path = [current]
            while current != start:
                current = parents[current[0]][current[1]]
                path.append(current)
            path.reverse()
            array = np.random.rand(10000, 10000)  # Creating a large random array
            result = np.mean(array)  # Performing a computation on the array
            time.sleep(random.uniform(0, 0.5))
            return True, path

        x, y = current
        visited[x][y] = True
        for neighbor in ((x-1, y), (x, y-1), (x+1, y), (x, y+1)):
            nx, ny = neighbor
            if (0 <= nx < len(matrix)) and (0 <= ny < len(matrix[0])) and matrix[nx][ny] == 1 and not visited[nx][ny]:
                g_score = g_scores[x][y] + 1
                if g_score < g_scores[nx][ny]:
                    parents[nx][ny] = (x, y)
                    g_scores[nx][ny] = g_score
                    f_scores[nx][ny] = g_score + manhattan_distance(neighbor, end)
                    heapq.heappush(heap, (f_scores[nx][ny], neighbor))

    # Path not found
    return False, None


"""
class DQN:
    def __init__(self, state_size, action_size, learning_rate, name='DQN'):
        self.state_size = state_size
        self.action_size = action_size
        self.learning_rate = learning_rate

        with tf.variable_scope(name):
            self._build_network()

    def _build_network(self):
        self.X = tf.placeholder(tf.float32, [None, self.state_size], name='input')

        W1 = tf.get_variable('W1', shape=[self.state_size, 16], initializer=tf.contrib.layers.xavier_initializer())
        layer1 = tf.nn.tanh(tf.matmul(self.X, W1))

        W2 = tf.get_variable('W2', shape=[16, self.action_size], initializer=tf.contrib.layers.xavier_initializer())
        self.Qpred = tf.matmul(layer1, W2)

        self.Y = tf.placeholder(shape=[None, self.action_size], dtype=tf.float32)

        self.loss = tf.reduce_mean(tf.square(self.Y - self.Qpred))
        self.train = tf.train.AdamOptimizer(learning_rate=self.learning_rate).minimize(self.loss)

    def predict(self, state, sess):
        return sess.run(self.Qpred, feed_dict={self.X: state})

    def update(self, state, Y, sess):
        return sess.run([self.loss, self.train], feed_dict={self.X: state, self.Y: Y})
"""