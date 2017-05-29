
class Rules:
    def __init__(self, size, win_condition):
        self.size = size
        self.win_condition = win_condition

    def __bounds_check(self, position):
        x, y = position
        return x < 0 or y < 0 or x >= self.size or y >= self.size

    def __check(self, board, x, y, dx, dy, connections):
        connected = 0
        while True:
            if self.__bounds_check((x, y)):
                break
            if not board[x, y]:
                break
            connections.append(int(x * self.size + y))
            connected += 1
            x, y = x - dx, y - dy
        return connected

    def check(self, board, position, direction):
        x, y = position
        dx, dy = direction
        connections = []
        connected =  self.__check(board, x, y, dx, dy, connections)
        connected += self.__check(board, x, y, -dx, -dy, connections)
        return connected - 1, connections

    def check_win(self, position, board):
        for direction in ((1,0), (0,1), (1,1), (1, -1)):
            connected, connections = self.check(board, position, direction)
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

