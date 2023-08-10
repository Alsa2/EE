import random
import math
from algorithms import BFS_checker, Astar, PRM


dark_green = "\033[48;5;22m"
light_green = "\033[48;5;28m"
red = "\033[48;5;9m"
blue = "\033[48;5;21m"
yellow = "\033[48;5;11m"
light_yellow = "\033[48;5;226m"
reset = "\033[0m"


class Map: # add argument num_ellipses with default value 10
    def __init__(self, map_width, map_height):
        self.map = [[0 for y in range(map_height)] for x in range(map_width)]

        #self.start = (random.randint(0, map_width - 1), random.randint(0, map_height - 1))
        #self.end = (random.randint(0, map_width - 1), random.randint(0, map_height - 1))

        self.start = (0, 0)
        self.end = (map_width - 1, map_height - 1)

        self.initial_circle(2)

        #add ellipses until possible check is true
        while True:
            self.generate_ellipse(map_width, map_height)
            
            status, path = BFS_checker(self.map, self.start, self.end)
            if status:
                print("Path found")
                self.print(path)
                print("Path length: ", len(path))
                break
            #else:
                #print("No path found")

        self.reset_map()
        return

    def generate_ellipse(self, map_width, map_height):
        crossable = []

        # Generate random a ellipse
        x = random.randint(0, map_width - 1)
        y = random.randint(0, map_height - 1)
        a = random.randint(2, 8)  # major axis
        b = random.randint(2, 8)  # minor axis
        angle = random.uniform(0, math.pi/2)  # angle of rotation
        shape = []
        for j in range(map_width):
            for k in range(map_height):
                if ((j-x)*math.cos(angle) + (k-y)*math.sin(angle))**2/a**2 + ((j-x)*math.sin(angle) - (k-y)*math.cos(angle))**2/b**2 <= 1:
                    shape.append((j, k))
        crossable.append(shape)

        for area in crossable:
            for cell in area:
                self.map[cell[0]][cell[1]] = 1


    def print(self): # dark green for 0 light green for 1 red for 2 blue for 3
        self.map[self.start[0]][self.start[1]] = 2
        self.map[self.end[0]][self.end[1]] = 3

        for row in self.map:
            for cell in row:
                if cell == 0:
                    print(dark_green + "  " + reset, end="")
                elif cell == 1:
                    print(light_green + "  " + reset, end="")
                elif cell == 2:
                    print(red + "  " + reset, end="")
                elif cell == 3:
                    print(blue + "  " + reset, end="")
            print()

    def print(self, path=[], visited=[]): # dark green for 0 light green for 1 red for 2 blue for 3
        print()
        if path != None:
            for cell in path:
                self.map[cell[0]][cell[1]] = 4

        for cell in visited:
            self.map[cell[0]][cell[1]] = 5

        self.map[self.start[0]][self.start[1]] = 2
        self.map[self.end[0]][self.end[1]] = 3

        for row in self.map:
            for cell in row:
                if cell == 0:
                    print(dark_green + "  " + reset, end="")
                elif cell == 1:
                    print(light_green + "  " + reset, end="")
                elif cell == 2:
                    print(red + "  " + reset, end="")
                elif cell == 3:
                    print(blue + "  " + reset, end="")
                elif cell == 4:
                    print(yellow + "  " + reset, end="")
                elif cell == 5:
                    print(light_yellow + "  " + reset, end="")
            print()

        #reset map by removing path and visited
        for cell in path:
            self.map[cell[0]][cell[1]] = 1

        for cell in visited:
            self.map[cell[0]][cell[1]] = 1

    def initial_circle(self, radius):# add a circle at start point and end point
        for i in range(self.start[0] - radius, self.start[0] + radius):
            for j in range(self.start[1] - radius, self.start[1] + radius):
                if i >= 0 and i < len(self.map) and j >= 0 and j < len(self.map[0]):
                    self.map[i][j] = 1

        for i in range(self.end[0] - radius, self.end[0] + radius):
            for j in range(self.end[1] - radius, self.end[1] + radius):
                if i >= 0 and i < len(self.map) and j >= 0 and j < len(self.map[0]):
                    self.map[i][j] = 1

    def get_status(self, x, y):
        #returns current position status, the one above, the one below, the one to the left, the one to the right
        return self.map[x][y], self.map[x][y-1], self.map[x][y+1], self.map[x-1][y], self.map[x+1][y]

    def return_matrix(self):
        return self.map
    
    def reset_map(self): #resets severything that is not 0 and 1 to 1 because algorithms see them as obstacles
        for i in range(len(self.map)):
            for j in range(len(self.map[0])):
                if self.map[i][j] != 0 and self.map[i][j] != 1:
                    self.map[i][j] = 1
        