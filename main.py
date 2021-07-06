# -*- coding: utf-8 -*-
"""
Created on Mon Jul  5 11:30:35 2021

@author: thomas.collaudin
"""
import os
import chess.pgn

from keras import callbacks, optimizers
from keras.layers import (LSTM, BatchNormalization, Dense, Dropout, Flatten,
                          TimeDistributed)
from keras.layers.convolutional import Conv2D, MaxPooling2D
from keras.models import Sequential, load_model, model_from_json
from matplotlib import pyplot as plt
from pprint import pprint

def create_model() :
    model = Sequential()
    model.add(Conv2D(filters=2048, kernel_size=1, activation='relu', input_shape=(8,8,12)))
    # model.add(MaxPooling2D())
    # model.add(Conv2D(filters=2048, kernel_size=1, activation='relu'))
    # model.add(MaxPooling2D())
    # model.add(Conv2D(filters=2048, kernel_size=1, activation='relu'))
    model.add(Flatten())
    # model.add(BatchNormalization())
    # model.add(Dense(1,activation = 'relu'))
    model.add(Dense(1, activation='relu'))
    # model.add(Dense(1))
    return model

def load_trained_model(weights_path):
   model = create_model()
   model.load_weights(weights_path)

def train_model(model, X, y) :    
    model.compile(optimizer='Nadam', loss='mse')
    dirx = 'models'
    os.chdir(dirx)
    h5 = 'chess_model' + '.h5'
    checkpoint = callbacks.ModelCheckpoint(h5,
                                               monitor='loss',
                                               verbose=0,
                                               save_best_only=True,
                                               save_weights_only=False,
                                               mode='auto',
                                               period=1)
    es = callbacks.EarlyStopping(monitor='loss', mode='min', verbose=1, patience=5000/10)
    callback = [checkpoint,es]
    json = 'chess_model' + '.json'
    model_json = model.to_json()
    with open(json, "w") as json_file:
        json_file.write(model_json)
    print('Training Network...')
    history = model.fit(X, y, epochs = 20,verbose = 2,callbacks = callback)
    plt.plot(history.history['loss'])
    return model

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

def make_matrix(board, chess_dict): 
    pgn = board.board_fen()
    matrix = []  
    rows = pgn.split("/")
    for row in rows:
        matrix_row = []  
        for term in row:
            if term.isdigit():
                for i in range(0, int(term)):
                    matrix_row.append(chess_dict['.'])
            else:
                matrix_row.append(chess_dict[term])
        matrix.append(matrix_row)
        
    return matrix

def get_Xy(files) :
    X, y = list(), list()
    chess_dict = get_chess_dict()
    for file in files :
        pgn = open(file)
        game = chess.pgn.read_game(pgn)
        g = 0
        while game != None and g < 100:
            g += 1
            print(g)
            board = game.board()
            nb_moves = 0
            for m, move in enumerate(game.mainline_moves()) :
                nb_moves += 1
            print(nb_moves)
            for m, move in enumerate(game.mainline_moves()) :
                board.push(move)
                if m % 2 == 0 :
                    value = 1
                    matrix = make_matrix(board.copy(), chess_dict)
                    X.append(matrix)
                    y.append(value)
            game = chess.pgn.read_game(pgn)
    return [X, y]

# files = ["games/ficsgamesdb_search_212926.pgn"]
# Xy = get_Xy(files)
# X, y = Xy[0], Xy[1]
# print(len(X))
# # print(y)
# model = create_model()
# model = train_model(model,X, y)

model = load_model('chess_model.h5')

chess_dict = get_chess_dict()
board = chess.Board()
for move in board.legal_moves :
    board.push(move)
    matrix = make_matrix(board.copy(), chess_dict)
    print('\n')
    print(move)
    print(board)
    # pprint(matrix)
    print(model.predict([matrix]))
    
    board.pop()

