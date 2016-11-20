from copy import deepcopy
from time import time
from random import randint
import itertools

# Global Constants
MaxUtility = 1e9
IsPlayerBlack = True
MaxAllowedTimeInSeconds = 60
MaxDepth = 4


class CheckersState:
    def __init__(self, grid, blackToMove, moves):
        self.grid = grid
        self.blackToMove = blackToMove
        self.moves = moves  # Hops taken by a disc to reach the current state

    # This just checks for whether or not all pieces of a player have been eliminated.
    # It does not check for whether a player has a move or not. In that case, there will
    # be no successors for that player and alpha beta search will return Min/Max Utility.
    def isTerminalState(self):
        blackSeen, whiteSeen = False, False
        for row in self.grid:
            for cell in row:
                if cell == 'b' or cell == 'B':
                    blackSeen = True
                elif cell == 'w' or cell == 'W':
                    whiteSeen = True
                if blackSeen and whiteSeen: return False
        self.isLoserBlack = whiteSeen
        return True

    def getTerminalUtility(self):
        return MaxUtility if IsPlayerBlack != self.isLoserBlack else -MaxUtility

    def getSuccessors(self):
        def getSteps(cell):
            whiteSteps = [(-1, -1), (-1, 1)]
            blackSteps = [(1, -1), (1, 1)]
            steps = []
            if cell != 'b': steps.extend(whiteSteps)
            if cell != 'w': steps.extend(blackSteps)
            return steps

        def generateMoves(board, i, j, successors):
            for step in getSteps(board[i][j]):
                x, y = i + step[0], j + step[1]
                if x >= 0 and x < 8 and y >= 0 and y < 8 and board[x][y] == '_':
                    boardCopy = deepcopy(board)
                    boardCopy[x][y], boardCopy[i][j] = boardCopy[i][j], '_'
                    # A pawn is promoted when it reaches the last row
                    if (x == 7 and self.blackToMove) or (x == 0 and not self.blackToMove):
                        boardCopy[x][y] = boardCopy[x][y].upper()
                    successors.append(CheckersState(boardCopy, not self.blackToMove, [(i, j), (x, y)]))

        def generateJumps(board, i, j, moves, successors):
            jumpEnd = True
            for step in getSteps(board[i][j]):
                x, y = i + step[0], j + step[1]
                if x >= 0 and x < 8 and y >= 0 and y < 8 and board[x][y] != '_' and board[i][j].lower() != board[x][
                    y].lower():
                    xp, yp = x + step[0], y + step[1]
                    if xp >= 0 and xp < 8 and yp >= 0 and yp < 8 and board[xp][yp] == '_':
                        board[xp][yp], save = board[i][j], board[x][y]
                        board[i][j] = board[x][y] = '_'
                        previous = board[xp][yp]
                        # A pawn is promoted when it reaches the last row
                        if (xp == 7 and self.blackToMove) or (xp == 0 and not self.blackToMove):
                            board[xp][yp] = board[xp][yp].upper()

                        moves.append((xp, yp))
                        generateJumps(board, xp, yp, moves, successors)
                        moves.pop()
                        board[i][j], board[x][y], board[xp][yp] = previous, save, '_'
                        jumpEnd = False
            if jumpEnd and len(moves) > 1:
                successors.append(CheckersState(deepcopy(board), not self.blackToMove, deepcopy(moves)))

        player = 'b' if self.blackToMove else 'w'
        successors = []

        # generate jumps
        for i in xrange(8):
            for j in xrange(8):
                if self.grid[i][j].lower() == player:
                    generateJumps(self.grid, i, j, [(i, j)], successors)
        if len(successors) > 0: return successors

        # generate moves
        for i in xrange(8):
            for j in xrange(8):
                if self.grid[i][j].lower() == player:
                    generateMoves(self.grid, i, j, successors)
        return successors


def piecesCount(state):
    # 1 for a normal piece, 1.5 for a king
    black, white = 0, 0
    for row in state.grid:
        for cell in row:
            if cell == 'b':
                black += 1.0
            elif cell == 'B':
                black += 1.5
            elif cell == 'w':
                white += 1.0
            elif cell == 'W':
                white += 1.5
    return black - white if IsPlayerBlack else white - black

def iterativeDeepeningAlphaBeta(state, evaluationFunc):
    startTime = time()

    def alphaBetaSearch(state, alpha, beta, depth):
        def maxValue(state, alpha, beta, depth):
            val = -MaxUtility
            for successor in state.getSuccessors():
                val = max(val, alphaBetaSearch(successor, alpha, beta, depth))
                if val >= beta: return val
                alpha = max(alpha, val)
            return val

        def minValue(state, alpha, beta, depth):
            val = MaxUtility
            for successor in state.getSuccessors():
                val = min(val, alphaBetaSearch(successor, alpha, beta, depth - 1))
                if val <= alpha: return val
                beta = min(beta, val)
            return val

        if state.isTerminalState(): return state.getTerminalUtility()
        if depth <= 0 or time() - startTime > MaxAllowedTimeInSeconds: return evaluationFunc(state)
        return maxValue(state, alpha, beta, depth) if state.blackToMove == IsPlayerBlack else minValue(state, alpha,
                                                                                                       beta, depth)

    bestMove = None
    for depth in xrange(1, MaxDepth):
        if time() - startTime > MaxAllowedTimeInSeconds: break
        val = -MaxUtility
        for successor in state.getSuccessors():
            score = alphaBetaSearch(successor, -MaxUtility, MaxUtility, depth)
            if score > val:
                val, bestMove = score, successor
    return bestMove

def makeMove(move, board, isBlack):
    firstStep = move[0]
    board[firstStep[0]][firstStep[1]] = '_'

    secondStep = move[1]

    result = deepcopy(board)

    if isBlack:
        result[secondStep[0]][secondStep[1]] = 'b'
    else:
        result[secondStep[0]][secondStep[1]] = 'w'

    return result

def generateRandomMove(zboard, blackToMove):
    mboard = deepcopy(zboard)

    def getSteps(cell):
        whiteSteps = [(-1, -1), (-1, 1)]
        blackSteps = [(1, -1), (1, 1)]
        steps = []
        if cell != 'b': steps.extend(whiteSteps)
        if cell != 'w': steps.extend(blackSteps)
        return steps

    def generateMoves(board, i, j, successors):
        for step in getSteps(board[i][j]):
            x, y = i + step[0], j + step[1]
            if x >= 0 and x < 8 and y >= 0 and y < 8 and board[x][y] == '_':
                boardCopy = deepcopy(board)
                boardCopy[x][y], boardCopy[i][j] = boardCopy[i][j], '_'
                # A pawn is promoted when it reaches the last row
                if (x == 7 and blackToMove) or (x == 0 and not blackToMove):
                    boardCopy[x][y] = boardCopy[x][y].upper()
                successors.append(CheckersState(boardCopy, not blackToMove, [(i, j), (x, y)]))

    def generateJumps(board, i, j, moves, successors):
        jumpEnd = True
        for step in getSteps(board[i][j]):
            x, y = i + step[0], j + step[1]
            if x >= 0 and x < 8 and y >= 0 and y < 8 and board[x][y] != '_' and board[i][j].lower() != board[x][y].lower():
                xp, yp = x + step[0], y + step[1]
                if xp >= 0 and xp < 8 and yp >= 0 and yp < 8 and board[xp][yp] == '_':
                    board[xp][yp], save = board[i][j], board[x][y]
                    board[i][j] = board[x][y] = '_'
                    previous = board[xp][yp]
                    # A pawn is promoted when it reaches the last row
                    if (xp == 7 and blackToMove) or (xp == 0 and not blackToMove):
                        board[xp][yp] = board[xp][yp].upper()

                    moves.append((xp, yp))
                    generateJumps(board, xp, yp, moves, successors)
                    moves.pop()
                    board[i][j], board[x][y], board[xp][yp] = previous, save, '_'
                    jumpEnd = False
            if jumpEnd and len(moves) > 1:
                successors.append(CheckersState(deepcopy(board), not blackToMove, deepcopy(moves)))

    player = 'b' if blackToMove else 'w'
    successors = []

    # generate jumps
    for i in xrange(8):
       for j in xrange(8):
            if mboard[i][j].lower() == player:
               generateJumps(mboard, i, j, [(i, j)], successors)
    if len(successors) > 0:
        movesNo = len(successors)
        rand = randint(0, movesNo - 1)
        return successors[rand]

     # generate moves
    for i in xrange(8):
        for j in xrange(8):
            if mboard[i][j].lower() == player:
                generateMoves(mboard, i, j, successors)

    movesNo = len(successors)

    rand = randint(0, movesNo -1)

    return successors[rand]


def translatePieces (board):
    for i in range(0 , len(board)):
        if board[i] == 'b':
            board[i] = 1
        if board[i] == 'w':
            board[i] = 8
        if board[i] == 'B':
            board[i] = 5
        if board[i] == 'W':
            board[i] = 12
        if board[i] == '_':
            board[i] = 0

    return board


if __name__ == '__main__':

    checkers = [['b', '_', 'b', '_', 'b', '_', 'b', '_'],
                ['_', 'b', '_', 'b', '_', 'b', '_', 'b'],
                ['b', '_', 'b', '_', 'b', '_', 'b', '_'],
                ['_', '_', '_', '_', '_', '_', '_', '_'],
                ['_', '_', '_', '_', '_', '_', '_', '_'],
                ['_', 'w', '_', 'w', '_', 'w', '_', 'w'],
                ['w', '_', 'w', '_', 'w', '_', 'w', '_'],
                ['_', 'w', '_', 'w', '_', 'w', '_', 'w']]

    player = []
    player.append('b')
    IsPlayerBlack = player[0] == 'b'

    # state = CheckersState(checkers, IsPlayerBlack, [])
    # move = iterativeDeepeningAlphaBeta(state, piecesCount)
    # print len(move) - 1
    # for step in move:
    #     print step[0], step[1]
    #
    # z = makeMove(move, checkers, True)

    # g = CheckersState(checkers, IsPlayerBlack, [move])


    #     rand = generateRandomMove(checkers, True)

    player = True;

    succe = deepcopy(checkers)

    resultlist = []

    nextmove = CheckersState(succe, True, [])

    for i in range(1, 35):
        oneMoveList = []

        nextmove = generateRandomMove(nextmove.grid, player)
        player = not player
        bestmove = iterativeDeepeningAlphaBeta(nextmove, piecesCount)
        randomMove = generateRandomMove(nextmove.grid, player)

        nextmoveFlat = list(itertools.chain(*nextmove.grid))
        bestmoveFlat = list(itertools.chain(*bestmove.grid))
        randomMoveFlat = list(itertools.chain(*randomMove.grid))


        oneMoveList.append(nextmove.grid)
        oneMoveList.append(bestmove.grid)
        oneMoveList.append(randomMove.grid)

        resultlist.append(oneMoveList)
        print i

    print len(resultlist)
    print "lol"
    #
    # print len(succe.grid)