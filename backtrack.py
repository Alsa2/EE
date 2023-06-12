import random

#colors for the maze
red = "\033[1;31;40m"
green = "\033[1;32;40m"
yellow = "\033[1;33;40m"
reset = "\033[0;0m"

class Maze:
    def __init__(self, width, height):

        self.width = width // 2 * 2 + 1
        self.height = height // 2 * 2 + 1

        # this creates a 2d-array for your maze data (False: path, True: wall)
        self.cells = [
                      [True for x in range(self.width)] 
                      for y in range(self.height)
                     ]

    def set_path(self, x, y):
        self.cells[y][x] = False

    def set_wall(self, x, y):
        self.cells[y][x] = True

    # a function to return if the current cell is a wall,
    #  and if the cell is within the maze bounds
    def is_wall(self, x, y):
        # checks if the coordinates are within the maze grid
        if 0 <= x < self.width and 0 <= y < self.height:
            # if they are, then we can check if the cell is a wall
            return self.cells[y][x]
        # if the coordinates are not within the maze bounds, we don't want to go there
        else:
            return False

    def create_maze(self, x, y):
        # set the current cell to a path, so that we don't return here later
        self.set_path(x, y)
        # we create a list of directions (in a random order) we can try
        all_directions = [[1, 0], [-1, 0], [0, 1], [0, -1]]
        random.shuffle(all_directions)

        # we keep trying the next direction in the list, until we have no directions left
        while len(all_directions) > 0:

            # we remove and return the last item in our directions list
            direction_to_try = all_directions.pop()

            # calculate the new node's coordinates using our random direction.
            # we *2 as we are moving two cells in each direction to the next node
            node_x = x + (direction_to_try[0] * 2)
            node_y = y + (direction_to_try[1] * 2)

            # check if the test node is a wall (eg it hasn't been visited)
            if self.is_wall(node_x, node_y):
                # success code: we have found a path

                # set our linking cell (between the two nodes we're moving from/to) to a path
                link_cell_x = x + direction_to_try[0]
                link_cell_y = y + direction_to_try[1]
                self.set_path(link_cell_x, link_cell_y)

                # "move" to our new node. remember we are calling the function every
                #  time we move, so we call it again but with the updated x and y coordinates
                self.create_maze(node_x, node_y)
        return

    def set_start_end(self, start, end):
        self.cells[start[1]][start[0]] = "Start"
        self.cells[end[1]][end[0]] = "End"

    def print_maze(self) -> str:
        for row in self.cells:
            for cell in row:
                if cell == True:
                    print("██", end="")
                elif cell == False:
                    print("  ", end="")
                else:
                    #set color for start and end (start = green, end = red)
                    if cell == "Start":
                        print(green + "██" + reset, end="")
                    elif cell == "End":
                        print(red + "██" + reset, end="")
                    elif cell == "Path":
                        print(yellow + "██" + reset, end="")
            print()

    def print_maze_with_solution(self, solution) -> str: #solution is a matrix like the one of the walls where True is where it passed and False where it didn't
        for row in range(len(self.cells)):
            for cell in range(len(self.cells[row])):
                if self.cells[row][cell]:
                    print("██", end="")
                else:
                    if solution[row][cell]:
                        print("  ", end="")
                    else:
                        print("██", end="")
            print()
    
    def export_maze(self): 
        #convert the matrix in 0 for wall 1 for the corridors and 2 for the start and 3 for the end
        #True is a wall False is a corridor
        for row in range(len(self.cells)):
            for cell in range(len(self.cells[row])):
                if self.cells[row][cell] == True:
                    self.cells[row][cell] = 1
                elif self.cells[row][cell] == False:
                    self.cells[row][cell] = 0
                elif self.cells[row][cell] == "Start":
                    self.cells[row][cell] = 2
                elif self.cells[row][cell] == "End":
                    self.cells[row][cell] = 3
        return self.cells


    def import_maze(self, matrix): # 1 is wall 0 is corridor and X is the path
        for row in range(len(matrix)):
            for cell in range(len(matrix[row])):
                if matrix[row][cell] == 1:
                    self.cells[row][cell] = True
                elif matrix[row][cell] == 0:
                    self.cells[row][cell] = False
                elif matrix[row][cell] == "X":
                    self.cells[row][cell] = "Path"
        return self.cells



        
        

if __name__ == "__main__":
    # create a maze object
    maze = Maze(100, 100)
    # create the maze
    maze.create_maze(1, 1)
    # print the maze
    maze.print_maze()
    # print the maze with start and end
    maze.set_start_end((1, 1), (99, 99))
    maze.print_maze()