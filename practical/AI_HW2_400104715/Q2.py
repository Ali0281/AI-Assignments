"""
ali aghayari 400104715
play with depth = 3 (more depth takes too long !)
got some help on algorythm efficiency from payam taebi
"""
import numpy.random
from Board import BoardUtility
import random
import math


class Player:
    def __init__(self, player_piece):
        self.piece = player_piece

    def play(self, board):
        return 0


class RandomPlayer(Player):
    def play(self, board):
        return [random.choice(BoardUtility.get_valid_locations(board)), random.choice([1, 2, 3, 4]),
                random.choice(["skip", "clockwise", "anticlockwise"])]


class HumanPlayer(Player):
    def play(self, board):
        move = input("row, col, region, rotation\n")
        move = move.split()
        print(move)
        return [[int(move[0]), int(move[1])], int(move[2]), move[3]]


# custom code section

class MiniMaxPlayer(Player):
    def __init__(self, player_piece, depth=3):
        super().__init__(player_piece)
        self.depth = depth

    def play(self, board):
        root = Node(self.piece, self.depth, 0, board)
        action = root.calculateBestMove(board=board)
        return [[action[0], action[1]], action[2], action[3]]


class MiniMaxProbPlayer(Player):
    def __init__(self, player_piece, depth=3, prob_stochastic=0.1):
        super().__init__(player_piece)
        self.depth = depth
        self.prob_stochastic = prob_stochastic

    def play(self, board):
        root = Node(self.piece, self.depth, 0, board)
        action = root.calculateBestMove(board=board, prob=self.prob_stochastic)
        return [[action[0], action[1]], action[2], action[3]]


class Node:
    def __init__(self, piece, limit, rank, board, action=None):
        self.board = board  # used this for a small portion of my code , my code doesnt copy every board state
        self.action = action  # saving the actions instead of board
        self.rank = rank  # rank
        self.children = None  # children
        self.value = 0  # best_val
        self.expVal = 0
        self.piece = piece
        self.maxPlayer = True if self.rank % 2 == 0 else False
        self.limit = limit

    def getChildren(self):
        if self.children is None:
            self.children = self.createChildren()
        return self.children

    def createChildren(self):
        actions = self.createActions()
        # actions = self.sortActions(actions)
        numpy.random.shuffle(actions)
        res = [Node(self.piece, self.limit, self.rank + 1, self.board, x) for x in actions]
        return res

    def createActions(self):
        res = []
        for i in BoardUtility.get_valid_locations(self.board):
            res.append((i[0], i[1], 4, "skip", self.piece))
            for reg in range(1, 5):
                for rot in ("clockwise", "anticlockwise"):
                    res.append((i[0], i[1], reg, rot, self.piece))
        return res

    def sortActions(self, actions):
        D = {}
        for i in actions:
            self.apply(i)
            D[i] = self.eval(self.board)
            self.desecrate(i)
        res = [x for x, y in sorted(D.items(), key=lambda x: x[1]).items()]
        if self.maxPlayer:
            return res
        else:
            return reversed(res)

    def apply(self, action=None):
        if action is None:
            if self.action is None: return
            BoardUtility.make_move(self.board, self.action[0], self.action[1], self.action[2], self.action[3],
                                   self.action[4])
        else:
            BoardUtility.make_move(self.board, action[0], action[1], action[2], action[3], action[4])

    def desecrate(self, action=None):
        if action is None and self.action is None:
            return
        elif action is not None:
            row, col, region, rotation = action[0], action[1], action[2], action[3]
        else:
            row, col, region, rotation = self.action[0], self.action[1], self.action[2], self.action[3]
        if rotation != "skip":
            rotation = "clockwise" if rotation == "anticlockwise" else "anticlockwise"
        BoardUtility.rotate_region(self.board, region, rotation)
        self.board[row][col] = 0

    def calculateBestMove(self, board, prob=0):
        self.minimax(1, board, prob)
        if prob != 0 and random.random() < prob and self.expVal != math.inf and self.expVal != - math.inf:
            res, diff = None, math.inf
            for i in self.getChildren():
                newDiff = abs(i.value - self.expVal)
                if newDiff < diff:
                    diff = newDiff
                    res = i.action
            return res
        else:
            for i in self.getChildren():
                if i.value == self.value:
                    return i.action

    # used this site : https://www.geeksforgeeks.org/minimax-algorithm-in-game-theory-set-4-alpha-beta-pruning/
    def minimax(self, depth, board, alpha=-math.inf, beta=math.inf, prob=0):
        if depth >= self.limit or BoardUtility.is_terminal_state(board):
            self.apply()
            val = self.eval(board)
            self.desecrate()
            self.value = val
            return val
        if self.maxPlayer:
            self.apply()
            sum = 0
            best = -math.inf
            for i in self.getChildren():
                val = i.minimax(depth + 1, board, alpha, beta, prob)
                sum += val
                best = max(best, val)
                alpha = max(alpha, best)
                if beta <= alpha: break
            self.desecrate()
            self.value = best
            self.expVal = sum / len(self.getChildren())
            return self.value + prob * (self.expVal - self.value)
        else:
            self.apply()
            best = math.inf
            for i in self.getChildren():
                val = i.minimax(depth + 1, board, alpha, beta, prob)
                best = min(best, val)
                beta = min(beta, best)
                if beta <= alpha: break
            self.desecrate()
            self.value = best
            return best

    def eval(self, board):
        points = 0
        # points for getting the center of the square
        for i in [2, 3]:
            for j in [2, 3]:
                if board[i][j] == self.piece:
                    points += 40
                elif board[i][j] != self.piece and board[i][j] != 0:
                    points -= 40

        # points for getting the center of the regions
        for i in [1, 4]:
            for j in [1, 4]:
                if board[i][j] == self.piece:
                    points += 20
                elif board[i][j] != self.piece and board[i][j] != 0:
                    points -= 20

        # points for creating a line
        # horiz
        for j in range(6):
            temp = 0
            for i in range(6):
                if board[i][j] != self.piece:
                    points += 2 ** temp
                    if temp > 4:
                        points = math.inf
                        return points

                    temp = 0

                else:
                    temp += 1
            points += 2 ** temp
            if temp > 4:
                points = math.inf
                return points

        # vert
        for i in range(6):
            temp = 0
            for j in range(6):
                if board[i][j] != self.piece:
                    points += 2 ** temp
                    if temp > 4:
                        points = math.inf
                        return points
                    temp = 0
                else:
                    temp += 1
            points += 2 ** temp
            if temp > 4:
                points = math.inf
                return points

        # diag \
        for i in range(6):
            temp = 0
            for j in range(6 - i):
                if board[i + j][j] != self.piece:
                    points += 2 ** temp
                    if temp > 4:
                        points = math.inf
                        return points

                    temp = 0
                else:
                    temp += 1
            points += 2 ** temp
            if temp > 4:
                points = math.inf
                return points

        for i in range(1, 6):
            temp = 0
            for j in range(6 - i):
                if board[j][i + j] != self.piece:
                    points += 2 ** temp
                    if temp > 4:
                        points = math.inf
                        return points

                    temp = 0
                else:
                    temp += 1
            points += 2 ** temp
            if temp > 4:
                points = math.inf
                return points

        # diag /
        for i in range(6):
            temp = 0
            for j in range(i + 1):
                if board[i - j][j] != self.piece:
                    points += 2 ** temp
                    if temp > 4:
                        points = math.inf
                        return points

                    temp = 0
                else:
                    temp += 1
            points += 2 ** temp
            if temp > 4:
                points = math.inf
                return points

        for i in range(1, 6):
            temp = 0
            for j in range(6 - i):
                if board[5 - j][i + j] != self.piece:
                    points += 2 ** temp
                    if temp > 4:
                        points = math.inf
                        return points

                    temp = 0
                else:
                    temp += 1
            points += 2 ** temp
            if temp > 4:
                points = math.inf
                return points

        # horiz
        for j in range(6):
            temp = 0
            for i in range(6):
                if board[i][j] == self.piece or board[i][j] == 0:
                    points -= 2 ** temp
                    if temp > 4:
                        points = -math.inf
                        return points

                    temp = 0
                else:
                    temp += 1
            points -= 2 ** temp
            if temp > 4:
                points = -math.inf
                return points

        # vert
        for i in range(6):
            temp = 0
            for j in range(6):
                if board[i][j] == self.piece or board[i][j] == 0:
                    points -= 2 ** temp
                    if temp > 4:
                        points = -math.inf
                        return points

                    temp = 0
                else:
                    temp += 1
            points -= 2 ** temp
            if temp > 4:
                points = -math.inf
                return points

        # diag \
        for i in range(6):
            temp = 0
            for j in range(6 - i):
                if board[i + j][j] == self.piece or board[i + j][j] == 0:
                    points -= 2 ** temp
                    if temp > 4:
                        points = -math.inf
                        return points

                    temp = 0
                else:
                    temp += 1
            points -= 2 ** temp
            if temp > 4:
                points = -math.inf
                return points

        for i in range(1, 6):
            temp = 0
            for j in range(6 - i):
                if board[j][i + j] == self.piece or board[j][i + j] == 0:
                    points -= 2 ** temp
                    if temp > 4:
                        points = -math.inf
                        return points

                    temp = 0
                else:
                    temp += 1
            points -= 2 ** temp
            if temp > 4:
                points = -math.inf
                return points

        # diag /
        for i in range(6):
            temp = 0
            for j in range(i + 1):
                if board[i - j][j] == self.piece or board[i - j][j] == 0:
                    points -= 2 ** temp
                    if temp > 4:
                        points = -math.inf
                        return points

                    temp = 0
                else:
                    temp += 1
            points -= 2 ** temp
            if temp > 4:
                points = -math.inf
                return points

        for i in range(1, 6):
            temp = 0
            for j in range(6 - i):
                if board[5 - j][i + j] == self.piece or board[5 - j][i + j] == 0:
                    points -= 2 ** temp
                    if temp > 4:
                        points = -math.inf
                        return points

                    temp = 0
                else:
                    temp += 1
            points -= 2 ** temp
            if temp > 4:
                points = -math.inf
                return points

        return points
