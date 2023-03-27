import random
import math
from collections import deque

dark_green = "\033[48;5;22m"
light_green = "\033[48;5;28m"
red = "\033[48;5;9m"
blue = "\033[48;5;21m"
reset = "\033[0m"

def generate_map(width, height):
    # Create an empty map
    map = [[0 for y in range(height)] for x in range(width)]

    # Add start and end points
    start = (0, random.randint(0, height - 1))
    end = (width - 1, random.randint(0, height - 1))
    map[start[0]][start[1]] = 2
    map[end[0]][end[1]] = 3

    while True:
        map[start[0]][start[1]] = 2
        map[end[0]][end[1]] = 3
        
        # Add non-crossable terrain
        non_crossable = []
        for i in range(10):  # number of non-crossable areas
            # Generate random ellipse
            x = random.randint(0, width - 1)
            y = random.randint(0, height - 1)
            a = random.randint(5, 25)  # major axis
            b = random.randint(2, 10)  # minor axis
            angle = random.uniform(0, math.pi/2)  # angle of rotation
            shape = []
            for j in range(width):
                for k in range(height):
                    if ((j-x)*math.cos(angle) + (k-y)*math.sin(angle))**2/a**2 + ((j-x)*math.sin(angle) - (k-y)*math.cos(angle))**2/b**2 <= 1:
                        shape.append((j, k))
            non_crossable.append(shape)

        for area in non_crossable:
            for cell in area:
                map[cell[0]][cell[1]] = 1

        print("Searching for a path...")
        for row in map:
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

        # Find neighbors for each non-crossable area
        neighbors = {}
        for i, area in enumerate(non_crossable):
            neighbors[i] = []
            for j, other in enumerate(non_crossable):
                if i != j:
                    for cell in area:
                        if any(cell in shape for shape in other):
                            neighbors[i].append(j)
                            break

        # Check if start and end points are connected
        visited = [False for i in range(len(non_crossable))]
        start_area = None
        end_area = None
        for i, area in enumerate(non_crossable):
            if start in area:
                start_area = i
            if end in area:
                end_area = i
        if start_area is None or end_area is None:
            continue
        queue = deque([start_area])
        visited[start_area] = True
        while queue:
            current = queue.popleft()
            if current == end_area:
                break
            for neighbor in neighbors[current]:
                if not visited[neighbor]:
                    visited[neighbor] = True
                    queue.append(neighbor)
        if not visited[end_area]:
            continue

        # Set values in map
        print("Path found!")
        print("Non crossable areas: " + str(len(non_crossable)))
        for area in non_crossable:
            for cell in area:
                map[cell[0]][cell[1]] = 1

        return map


map = generate_map(40, 40)
# dark green for 0 light green for 1 red for 2 blue for 3
for row in map:
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

print(map)