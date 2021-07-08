# -*- coding: utf-8 -*-
"""
Created on Wed Jul  7 21:58:09 2021

@author: thomas.collaudin
"""

import numpy as np
from tools import get_list_shape
from dbn.dbn.tensorflow import UnsupervisedDBN
import time as time
import torch

def get_WL(X) :
    res = list()
    split = X.split('_')
    for x in split[1:] :
        if x == '1' :
            res.append(int(x))
        else :
            for _ in range(int(x) - 1) :
                res.append(0)
    return list(np.reshape(res, (int(split[0]), 773)))

if __name__ == "__main__" :
    with open("W", "r") as f:
        W = get_WL(f.read())
    start_time_step = time.time()
    unsupervised_dbn = UnsupervisedDBN(hidden_layers_structure=[773, 600, 400, 200, 100],
                                    learning_rate_rbm=0.005,
                                    n_epochs_rbm=2,
                                    batch_size=100,
                                    activation_function='relu')
    X_dbn = unsupervised_dbn.fit_transform(np.array(W))
    unsupervised_dbn.save('pos2vec.pb')
    # # print(get_list_shape(X_dbn))
    # end_time_step = time.time()
    # print("~ Execution time : %s secondes" % (round(end_time_step - start_time_step, 2)))
    # w_transform = list()
    # model = UnsupervisedDBN().load('pos2vec.pb')
    # for k, w in enumerate(W) :
    #     print('\r', k, end="")
    #     w_transform.append(get_list_shape(model.transform(w)))
        
    
    