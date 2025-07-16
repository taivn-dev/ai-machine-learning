import matplotlib.pyplot as plt
import heapq

class Node:
    def __init__(self, position, parent=None):
        self.position = position
        self.parent = parent
        self.g = 0
        self.h = 0
        self.f = 0

    def __eq__(self, other):
        return self.position == other.position

    def __lt__(self, other):
        return self.f < other.f  # So sánh theo tổng chi phí

def heuristic(a, b):
    # khoảng cách Manhattan
    return abs(a[0] - b[0]) + abs(a[1] - b[1])

def get_neighbors(node, grid):
    neighbors = []
    for dx, dy in [(-1,0), (1,0), (0,-1), (0,1)]:
        x, y = node.position[0] + dx, node.position[1] + dy
        if 0 <= x < len(grid) and 0 <= y < len(grid[0]) and grid[x][y] == 0:
            neighbors.append((x, y))
    return neighbors

def astar(grid, start, goal):
    open_list = []
    closed_list = []

    start_node = Node(start)
    goal_node = Node(goal)
    heapq.heappush(open_list, (start_node.f, start_node))

    while open_list:
        _, current_node = heapq.heappop(open_list)
        closed_list.append(current_node)

        if current_node == goal_node:
            path = []
            while current_node:
                path.append(current_node.position)
                current_node = current_node.parent
            return path[::-1]

        for neighbor_pos in get_neighbors(current_node, grid):
            neighbor = Node(neighbor_pos, current_node)
            if neighbor in closed_list:
                continue

            neighbor.g = current_node.g + 1
            neighbor.h = heuristic(neighbor.position, goal_node.position)
            neighbor.f = neighbor.g + neighbor.h

            if any(neighbor == n and neighbor.g >= n.g for _, n in open_list):
                continue

            heapq.heappush(open_list, (neighbor.f, neighbor))

    return None


def visualize(grid, path, start, goal):
    fig, ax = plt.subplots()
    for x in range(len(grid)):
        for y in range(len(grid[0])):
            if (x, y) == start:
                color = 'green'  # màu cho ô bắt đầu
            elif (x, y) == goal:
                color = 'red'    # màu cho ô đích
            elif grid[x][y] == 1:
                color = 'black'  # chướng ngại vật
            elif path and (x, y) in path:
                color = 'blue'   # đường đi
            else:
                color = 'white'  # ô trống

            rect = plt.Rectangle((y, x), 1, 1, facecolor=color, edgecolor='gray')
            ax.add_patch(rect)

    ax.set_xlim(0, len(grid[0]))
    ax.set_ylim(0, len(grid))
    ax.set_xticks(range(len(grid[0]) + 1))
    ax.set_yticks(range(len(grid) + 1))
    ax.set_xticklabels([])
    ax.set_yticklabels([])
    ax.grid(True)
    plt.gca().invert_yaxis()
    plt.title("Robot Pathfinding")
    plt.show()


# Khởi tạo lưới 10x10: 0 là đường đi, 1 là chướng ngại vật
grid = [[0]*10 for _ in range(10)]
grid[3][5] = grid[5][4] = grid[5][5] = 1
grid[4][6] = grid[5][6] = grid[6][6] = 1
grid[3][0] = grid[3][1] = grid[3][2] = 1
grid[4][1] = grid[5][1] = grid[6][1] = 1
grid[4][4] = grid[5][4] = grid[6][4] = 1
grid[3][8] = grid[4][8] = grid[5][8] = grid[6][8] = 1

# no path blocks
# grid[1][0] = grid[0][1] = 1

start = (0, 0)
goal = (9, 9)
path = astar(grid, start, goal)

if path:
    print("Đường đi tìm được:", path)
    visualize(grid, path, start, goal)
else:
    print("Không tìm được đường đi.")
    visualize(grid, path, start, goal)