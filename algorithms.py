from queue import Queue

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
