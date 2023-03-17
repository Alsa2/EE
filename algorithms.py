class Node:
    def __init__(self, x, y, g=0, h=0):
        self.x = x
        self.y = y
        self.g = g
        self.h = h

def find_path_Astar(self, start_x, start_y, end_x, end_y):
    # create start and end nodes
    start_node = Node(start_x, start_y)
    end_node = Node(end_x, end_y)

    # create lists to keep track of open and closed nodes
    open_list = [start_node]
    closed_list = []

    # loop until we find the end node or run out of nodes to check
    while open_list:
        # get the node with the lowest f score from the open list
        current_node = min(open_list, key=lambda n: n.g + n.h)

        # if we've found the end node, reconstruct the path and return it
        if current_node.x == end_x and current_node.y == end_y:
            path = []
            while current_node:
                path.append((current_node.x, current_node.y))
                current_node = current_node.parent
            return path[::-1]

        # remove the current node from the open list and add it to the closed list
        open_list.remove(current_node)
        closed_list.append(current_node)

        # check each neighbor of the current node
        for direction in [(1, 0), (-1, 0), (0, 1), (0, -1)]:
            neighbor_x = current_node.x + direction[0]
            neighbor_y = current_node.y + direction[1]

            # check if the neighbor is a wall or outside the bounds of the maze
            if self.is_wall(neighbor_x, neighbor_y) or neighbor_x < 0 or neighbor_x >= self.width or neighbor_y < 0 or neighbor_y >= self.height:
                continue

            # create a new node for the neighbor
            neighbor_node = Node(neighbor_x, neighbor_y)

            # if the neighbor is already on the closed list, skip it
            if neighbor_node in closed_list:
                continue

            # calculate the tentative g score for the neighbor
            tentative_g = current_node.g + 1

            # if the neighbor is not on the open list, add it and calculate its h score
            if neighbor_node not in open_list:
                neighbor_node.h = abs(neighbor_x - end_x) + abs(neighbor_y - end_y)
                open_list.append(neighbor_node)

            # if the tentative g score is lower than the neighbor's current g score, update it and set the current node as its parent
            elif tentative_g < neighbor_node.g:
                neighbor_node.g = tentative_g
                neighbor_node.parent = current_node

    # if we've run out of nodes on the open list and haven't found the end node, return None (no path found)
    return None