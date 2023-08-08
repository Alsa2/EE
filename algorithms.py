from queue import Queue
import heapq
import numpy as np
# import tensorflow as tf    Will be implemented later

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

def manhattan_distance(point1, point2):
    return abs(point1[0] - point2[0]) + abs(point1[1] - point2[1])

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