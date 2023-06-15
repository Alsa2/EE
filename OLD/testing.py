import random
import math

class Map: # add argument num_ellipses with default value 10
    def __init__(self, width, height, min_num_ellipses=10, max_num_ellipses=50, num_ellipses=None):
        num_ellipses = random.randint(min_num_ellipses, max_num_ellipses)

        # Create an empty map
        self.map = [[0 for y in range(height)] for x in range(width)]

        # Add non-crossable terrain
        crossable = []
        for i in range(num_ellipses):  # number of crossable areas
            # Generate random ellipse
            x = random.randint(0, width - 1)
            y = random.randint(0, height - 1)
            a = random.randint(5, 15)  # major axis
            b = random.randint(2, 8)  # minor axis
            angle = random.uniform(0, math.pi/2)  # angle of rotation
            shape = []
            for j in range(width):
                for k in range(height):
                    if ((j-x)*math.cos(angle) + (k-y)*math.sin(angle))**2/a**2 + ((j-x)*math.sin(angle) - (k-y)*math.cos(angle))**2/b**2 <= 1:
                        shape.append((j, k))
            crossable.append(shape)

        # Find neighbors for each non-crossable area
        neighbors = {}
        for i, area in enumerate(crossable):
            neighbors[i] = []
            for j, other in enumerate(crossable):
                if i != j:
                    for cell in area:
                        if any(cell in shape for shape in other):
                            neighbors[i].append(j)
                            break

        # Choose start and end points
        start = random.choice(crossable[0])
        end = random.choice(crossable[-1])

        # Set values in map
        for area in crossable:
            for cell in area:
                self.map[cell[0]][cell[1]] = 1
        self.map[start[0]][start[1]] = 2
        self.map[end[0]][end[1]] = 3

        return

    def __print__(self): # dark green for 0 light green for 1 red for 2 blue for 3
        dark_green = "\033[48;5;22m"
        light_green = "\033[48;5;28m"
        red = "\033[48;5;9m"
        blue = "\033[48;5;21m"
        reset = "\033[0m"
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

    def return_matrix(self):
        return self.map

        
        
    


map = Map(50, 50)
map.__print__()
print(map.return_matrix())

