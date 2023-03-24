from backtrack import Maze
red = "\033[1;31;40m"
green = "\033[1;32;40m"
yellow = "\033[1;33;40m"
reset = "\033[0;0m"

import heapq

def heuristic(a, b):
    # Calculates the Manhattan distance between two points a and b
    return abs(a[0] - b[0]) + abs(a[1] - b[1])

def a_star(start, end, grid):
    # Initialize the frontier with the start node
    frontier = [(0, start)]
    # Keep track of the cost of the best path found to each node
    cost_so_far = {start: 0}
    # Keep track of the parent node of each visited node to reconstruct the path
    parent = {start: None}

    while frontier:
        # Get the node with the lowest estimated total cost
        _, current = heapq.heappop(frontier)

        # Check if we have reached the end node
        if current == end:
            break

        # Visit each neighbor of the current node
        for neighbor in [(current[0]-1, current[1]), (current[0]+1, current[1]), 
                         (current[0], current[1]-1), (current[0], current[1]+1)]:
            # Ignore neighbors that are walls or outside the grid
            if neighbor[0] < 0 or neighbor[0] >= len(grid) or neighbor[1] < 0 or neighbor[1] >= len(grid[0]) or grid[neighbor[0]][neighbor[1]] == 1:
                continue

            # Calculate the cost of the path to this neighbor
            new_cost = cost_so_far[current] + 1

            # If we have not visited this neighbor or we have found a better path to it
            if neighbor not in cost_so_far or new_cost < cost_so_far[neighbor]:
                # Update the cost and parent of the neighbor
                cost_so_far[neighbor] = new_cost
                priority = new_cost + heuristic(end, neighbor)
                heapq.heappush(frontier, (priority, neighbor))
                parent[neighbor] = current

    # Reconstruct the path by following the parents from the end node to the start node
    path = []
    current = end
    while current != start:
        path.append(current)
        current = parent[current]
    path.append(start)
    path.reverse()

    # Replace the solution path with X in the grid
    for node in path:
        grid[node[0]][node[1]] = "X"

    # Return the updated grid
    return grid


if __name__ == "__main__":
    maze = Maze(80, 80)
    # create the maze
    maze.create_maze(1, 1)
    # set start and end
    maze.set_start_end((1, 1), (79, 79))
    # print the maze
    
    print("Before:")
    maze.print_maze()

    grid = maze.export_maze()
    print("After:")
    solution_maze = a_star((1, 1), (79, 79), grid)
    #set back the start and end
    solution_maze[1][1] = 2
    solution_maze[79][79] = 3
    for row in solution_maze:
        for cell in row:
            if cell == 1:
                print("██", end="")
            elif cell == 0:
                print("  ", end="")
            elif cell == 2:
                print(green + "██" + reset, end="")
            elif cell == 3:
                print(red + "██" + reset, end="")
            elif cell == "X":
                print(yellow + "██" + reset, end="")
        print()