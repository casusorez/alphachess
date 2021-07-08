# -*- coding: utf-8 -*-
"""
Created on Wed Jul  7 16:01:16 2021

@author: thomas.collaudin
"""
import random
import time
import chess
import numpy as np
from pathlib import Path

def get_chess_dict() :
    chess_dict = {
        'p' : [1,0,0,0,0,0,0,0,0,0,0,0],
        'P' : [0,0,0,0,0,0,1,0,0,0,0,0],
        'n' : [0,1,0,0,0,0,0,0,0,0,0,0],
        'N' : [0,0,0,0,0,0,0,1,0,0,0,0],
        'b' : [0,0,1,0,0,0,0,0,0,0,0,0],
        'B' : [0,0,0,0,0,0,0,0,1,0,0,0],
        'r' : [0,0,0,1,0,0,0,0,0,0,0,0],
        'R' : [0,0,0,0,0,0,0,0,0,1,0,0],
        'q' : [0,0,0,0,1,0,0,0,0,0,0,0],
        'Q' : [0,0,0,0,0,0,0,0,0,0,1,0],
        'k' : [0,0,0,0,0,1,0,0,0,0,0,0],
        'K' : [0,0,0,0,0,0,0,0,0,0,0,1],
        '.' : [0,0,0,0,0,0,0,0,0,0,0,0],
    }
    return chess_dict

def make_matrix(board, chess_dict, m): 
    pgn = board.board_fen()
    matrix = []  
    rows = pgn.split("/")
    for row in rows:
        for term in row:
            if term.isdigit():
                for i in range(0, int(term)):
                    matrix.append(chess_dict['.'])
            else:
                matrix.append(chess_dict[term])
    matrix = list(np.reshape(matrix, 768))
    if m % 2 == 0 :
        matrix.append(1)
    else :
        matrix.append(0)
    if board.has_kingside_castling_rights(chess.WHITE) ==  True :
        matrix.append(1)
    else :
        matrix.append(0)
    if board.has_queenside_castling_rights(chess.WHITE) ==  True :
        matrix.append(1)
    else :
        matrix.append(0)
    if board.has_kingside_castling_rights(chess.BLACK) ==  True :
        matrix.append(1)
    else :
        matrix.append(0)
    if board.has_queenside_castling_rights(chess.BLACK) ==  True :
        matrix.append(1)
    else :
        matrix.append(0)
    return matrix

def make_WL(file) :
    W, L = list(), list()
    chess_dict = get_chess_dict()
    pgn = open(file)
    game = chess.pgn.read_game(pgn)
    g, n, nW, nL = 0, 0, 0, 0  
    while game != None and g < 1000 :
        g += 1
        board = game.board()
        result = game.headers["Result"]
        nb_moves = int(game.headers["PlyCount"])      
        if result in ['1-0', '0-1'] and nb_moves > 15 :
            random_list = random.sample(range(5, nb_moves), 10)
            for m, move in enumerate(game.mainline_moves()) :
                board.push(move)
                if m in random_list :
                    n += 1
                    matrix = make_matrix(board.copy(), chess_dict, m)                    
                    if result == '1-0' :
                        nW += 1
                        W.append(matrix)
                    else :
                        nL += 1
                        L.append(matrix)
        print('\r# games imported : ' + str(g), end="")
        game = chess.pgn.read_game(pgn)
    print('')
    print('# W positions selected : ' + str(nW))
    print('# L positions selected : ' + str(nL))
    print('# positions selected : ' + str(n))
    return [W, L]

def compress_WL(WL) :
    res = list()
    g = 0
    for i in range(2) :
        res.append([str(len(WL[i]))])
        count = 0
        for m in range(len(WL[i])) :
            g += 1            
            print('\r# positions compressed : ' + str(g), end="")
            for c in range(len(WL[i][m])) :
                if WL[i][m][c] == 1 :
                    if count >= 1 :
                        res[-1].append('_' + str(count + 1))
                        count = 0
                    res[-1].append('_' + '1')
                elif c == len(WL[i][m]) - 1 :
                    count += 1                    
                    res[-1].append('_' + str(count + 1))
                    count = 0
                else :
                    count += 1
    print('')
    return res

if __name__ == "__main__" :
    print("\n---------- CREATING TRAINING SET")
    start_time_step = time.time()
    file = "games/games.pgn"
    WL = make_WL(file)
    WL = compress_WL(WL)
    W, L = WL[0], WL[1]
    with open("W", "w") as f:
        f.write("".join(W))
        print('Size of W file : ', round(Path('W').stat().st_size / 1000000, 1), 'Mo')
    with open("L", "w") as f:
        f.write("".join(L))
        print('Size of L file : ', round(Path('L').stat().st_size / 1000000, 1), 'Mo')
    end_time_step = time.time()
    print("~ Execution time : %s secondes" % (round(end_time_step - start_time_step, 2)))
