import random

# Constants for the Tic-Tac-Toe board
PLAYER_X = 'X'
PLAYER_O = 'O'
EMPTY = ' '

def print_board(board):
    for row in board:
        print(" ".join(row))
    print()

def is_winner(board, player):
    # Check rows, columns, and diagonals for a winner
    for i in range(3):
        if all(cell == player for cell in board[i]) or all(board[j][i] == player for j in range(3)):
            return True
    if all(board[i][i] == player for i in range(3)) or all(board[i][2 - i] == player for i in range(3)):
        return True
    return False

def is_board_full(board):
    # Check if the board is full
    return all(cell != EMPTY for row in board for cell in row)

def get_empty_cells(board):
    # Return a list of empty cells on the board
    return [(i, j) for i in range(3) for j in range(3) if board[i][j] == EMPTY]

def minimax(board, depth, maximizing_player, alpha, beta):
    if is_winner(board, PLAYER_X):
        return -1
    elif is_winner(board, PLAYER_O):
        return 1
    elif is_board_full(board):
        return 0

    empty_cells = get_empty_cells(board)

    if maximizing_player:
        max_eval = float('-inf')
        for i, j in empty_cells:
            board[i][j] = PLAYER_O
            eval = minimax(board, depth + 1, False, alpha, beta)
            board[i][j] = EMPTY
            max_eval = max(max_eval, eval)
            alpha = max(alpha, eval)
            if beta <= alpha:
                break
        return max_eval
    else:
        min_eval = float('inf')
        for i, j in empty_cells:
            board[i][j] = PLAYER_X
            eval = minimax(board, depth + 1, True, alpha, beta)
            board[i][j] = EMPTY
            min_eval = min(min_eval, eval)
            beta = min(beta, eval)
            if beta <= alpha:
                break
        return min_eval

def get_best_move(board):
    best_val = float('-inf')
    best_move = None
    for i, j in get_empty_cells(board):
        board[i][j] = PLAYER_O
        move_val = minimax(board, 0, False, float('-inf'), float('inf'))
        board[i][j] = EMPTY
        if move_val > best_val:
            best_move = (i, j)
            best_val = move_val
    return best_move

def main():
    board = [[EMPTY] * 3 for _ in range(3)]
    current_player = PLAYER_X if random.choice([True, False]) else PLAYER_O

    while True:
        print_board(board)

        if current_player == PLAYER_X:
            row, col = map(int, input("Enter your move (row and column, separated by a space): ").split())
            if board[row][col] != EMPTY:
                print("Invalid move. Cell already occupied. Try again.")
                continue
            board[row][col] = PLAYER_X
        else:
            print("AI is thinking...")
            row, col = get_best_move(board)
            print(f"AI plays at row {row}, column {col}.")
            board[row][col] = PLAYER_O

        if is_winner(board, current_player):
            print_board(board)
            print(f"{current_player} wins!")
            break
        elif is_board_full(board):
            print_board(board)
            print("It's a draw!")
            break

        current_player = PLAYER_X if current_player == PLAYER_O else PLAYER_O

if __name__ == "__main__":
    main()
