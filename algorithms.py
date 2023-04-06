from queue import Queue
import heapq

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
    visited = set()
    parents = {}
    g_scores = {start: 0}
    f_scores = {start: manhattan_distance(start, end)}

    while heap:
        current_f, current = heapq.heappop(heap)
        if current == end:
            # Path found
            path = [current]
            while current != start:
                current = parents[current]
                path.append(current)
            path.reverse()
            return True, visited, path

        visited.add(current)
        neighbors = [(current[0] - 1, current[1]), (current[0], current[1] - 1),
                     (current[0] + 1, current[1]), (current[0], current[1] + 1)]
        for neighbor in neighbors:
            x, y = neighbor
            if (0 <= x < len(matrix)) and (0 <= y < len(matrix[0])) and matrix[x][y] == 1 and neighbor not in visited:
                g_score = g_scores[current] + 1
                if neighbor not in g_scores or g_score < g_scores[neighbor]:
                    parents[neighbor] = current
                    g_scores[neighbor] = g_score
                    f_scores[neighbor] = g_score + manhattan_distance(neighbor, end)
                    heapq.heappush(heap, (f_scores[neighbor], neighbor))

    # Path not found
    return False, None, visited

def manhattan_distance(point1, point2):
    return abs(point1[0] - point2[0]) + abs(point1[1] - point2[1])
