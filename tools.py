# -*- coding: utf-8 -*-
"""
Created on Wed Jul  7 22:04:50 2021

@author: thomas.collaudin
"""

def get_list_shape(input):    
    shape = []
    a = len(input)
    shape.append(a)
    b = input[0]
    while a > 0:
        try:
            a = len(b)
            shape.append(a)
            b = b[0]
        except:
            break
    return shape