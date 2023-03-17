from backtrack import Maze
from algorithms import find_path_Astar

if __name__ == "__main__":
    # create a maze object
    maze = Maze(10, 10)
    maze.create_maze(1, 1)

    maze.print_maze()

    

    # find a path from (1, 1) to (8, 8)
    path = find_path_Astar(maze, 1, 1, 2, 1)

    # if a path was found, mark it on the maze
    if path:
        for x, y in path:
            maze.maze[y][x] = "x"

    # print the maze
    maze.print_maze()
