import numpy as np

class Rules:
    def __init__(self, size, win_condition):
        self.size = size
        self.win_condition = win_condition

    def __check(self, board, sx, sy, ex, ey, dx, dy, mask):
        connected = 0
        x, y = sx, sy
        scan_end = False
        empty = True
        start = 0
        connections = []
        while connected < self.win_condition:
            if x >= 0 and y >= 0 and x < self.size and y < self.size:
                add = False
                if board[(x, y)]:
                    start = start + 1
                    add = True
                elif len(mask) == 0 or mask[(x, y)]:
                    start = 0
                    connected = 0
                    connections = []
                    empty = False
                elif empty:
                    if start > 0:
                        connections = connections[-start:]
                    else:
                        connections = []
                    connected = start
                    start = 0
                    add = True
                else:
                    start = 0
                    add = True
                    empty = True
                if add:
                    connections.append(int(x * self.size + y))
                    connected += 1
            x, y = x + dx, y + dy
            if scan_end:
                break
            if x == ex and y == ey:
                scan_end = True
        return connected, connections

    def check(self, board, position, direction, mask):
        x, y = position
        dx, dy = direction
        sx = x - dx * (self.win_condition - 1)
        ex = x + dx * (self.win_condition - 1)
        sy = y - dy * (self.win_condition - 1)
        ey = y + dy * (self.win_condition - 1)
        connected, connections = self.__check(board, sx, sy, ex, ey, dx, dy, mask)
        return connected, connections

    def check_win(self, position, board, mask = []):
        for direction in ((1,0), (0,1), (1,1), (1, -1)):
            connected, connections = self.check(board, position, direction, mask)
            if connected >= self.win_condition:
                connections.sort()
                r = []
                last = None
                for i in connections:
                    if last == i:
                        continue
                    r.append(i)
                    last = i
                return True, r
        return False, None

def main():
    rules = Rules(7, 4)
    board = np.array(
         [[0, 0, 0, 1, 0, 0, 0],
          [0, 0, 0, 0, 0, 0, 0],
          [0, 0, 1, 0, 0, 1, 0],
          [0, 0, 0, 1, 0, 0, 0],
          [0, 0, 0, 1, 0, 0, 0],
          [0, 0, 1, 0, 0, 0, 0],
          [0, 0, 0, 0, 0, 0, 0]])
    mask = board.copy()
    mask[3,3] = 1
    win, conn = rules.check_win((3, 4), board, mask)
    if win:
        for i in conn:
            print("({}, {}) ".format(i // 7, i % 7))

if __name__ == "__main__":
    main()
