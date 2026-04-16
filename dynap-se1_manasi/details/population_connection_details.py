#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Apr 28 00:23:51 2018

@author: dzenn


This tool provides sets of indices for "hard-wired" connectivity of a threeway
relation unit and WTA connectivity profiles as lists of pre-post neuron indices.
"""

import random

def coord_to_index(i, j, inputSize):
    m = (i+inputSize)%inputSize
    k = (j+inputSize)%inputSize
    return m + (k)*inputSize


def A_plus_B_equals_C(pre_key, post_key, inputSize):

    if pre_key == 'A' and post_key == 'H':
        output_i = set_i('X', inputSize)
        output_j = set_j('X', inputSize)
    elif pre_key == 'H' and post_key == 'A':
        output_i = set_j('X', inputSize)
        output_j = set_i('X', inputSize)
    elif pre_key == 'B' and post_key == 'H':
        output_i = set_i('Y', inputSize)
        output_j = set_j('Y', inputSize)
    elif pre_key == 'H' and post_key == 'B':
        output_i = set_j('Y', inputSize)
        output_j = set_i('Y', inputSize)
    elif pre_key == 'C' and post_key == 'H':
        output_i = set_diagonal_i(inputSize)
        output_j = set_diagonal_j(inputSize)
    elif pre_key == 'H' and post_key == 'C':
        output_i = set_diagonal_j(inputSize)
        output_j = set_diagonal_i(inputSize)
    else:
        raise NotImplementedError('Invalid combination of population keys')
    
    return output_i, output_j
        

def set_i(inputAxis, inputSize):
    output = []

    for k in range(inputSize):
        for m in range(inputSize):
            if inputAxis != 'Z':
                output.append(k) 
            else:
                output.append()
    return output

def set_j(inputAxis, inputSize):
    output = []
    for k in range(inputSize):
        for m in range(inputSize):
            if inputAxis == 'X':
                output.append(coord_to_index(k,m,inputSize))    
            elif inputAxis == 'Y':
                output.append(coord_to_index(m,k,inputSize))  
    return output

def set_diagonal_i_no_wrap(inputSize):
    output = []
    for k in range(inputSize):
        z = k/2 - 0.5*(k%2)
        for m in range(max(k-inputSize,0),min(inputSize,k)):
            output.append(int(z))
    return output

def set_diagonal_j_no_wrap(inputSize):
    output = []
    for k in range(2*inputSize):
        for m in range(max(k-inputSize,0),min(inputSize,k)):
            output.append(coord_to_index(m, k - m - 1, inputSize))
    return output

def set_diagonal_i(inputSize):
    output = []
    for k in range(inputSize):
        for m in range(inputSize):
            output.append(int(k))
    return output

def set_diagonal_j(inputSize):
    output = []
    for k in range(inputSize):
        for m in range(inputSize):
            output.append(coord_to_index(m, (k - m + inputSize)%inputSize, inputSize))
    return output

def neighbors1d_inner_i(inputSize):
    output = []
    for k in range(inputSize):
        output.append(k)
        output.append(k)
    return output

def neighbors1d_inner_j(inputSize):
    output = []
    for k in range(inputSize):
        output.append((k-1+inputSize)%inputSize)
        output.append((k+1+inputSize)%inputSize)
    return output

def neighbors1d_outer_i(inputSize):
    output = []
    for k in range(inputSize):
        output.append(k)
        output.append(k)
    return output

def neighbors1d_outer_j(inputSize):
    output = []
    for k in range(inputSize):
        output.append((k-2+inputSize)%inputSize)
        output.append((k+2+inputSize)%inputSize)
    return output

def neighbors1d_third_closest_i(inputSize):
    output = []
    for k in range(inputSize):
        output.append(k)
        output.append(k)
    return output

def neighbors1d_third_closest_j(inputSize):
    output = []
    for k in range(inputSize):
        output.append((k-3+inputSize)%inputSize)
        output.append((k+3+inputSize)%inputSize)
    return output

def neighbors2d_inner_i(inputSize):
    output = []
    for k in range(inputSize):
        for m in range(inputSize):
            i = coord_to_index(m,k,inputSize)
            output.append(i)
            output.append(i)
            output.append(i)
            output.append(i)
            output.append(i)
            output.append(i)
    return output

def neighbors2d_inner_j(inputSize):
    output = []
    for k in range(inputSize):
        for m in range(inputSize):
            output.append(coord_to_index(m-1,k,inputSize))
            output.append(coord_to_index(m+1,k,inputSize))
            output.append(coord_to_index(m-1,k+1,inputSize))
            output.append(coord_to_index(m+1,k-1,inputSize))
            output.append(coord_to_index(m,k-1,inputSize))
            output.append(coord_to_index(m,k+1,inputSize))
    return output

def neighbors2d_outer_i(inputSize):
    output = []
    for k in range(inputSize):
        for m in range(inputSize):
            i = coord_to_index(m,k,inputSize)
            output.append(i)
            output.append(i)
            output.append(i)
            output.append(i)
            output.append(i)
            output.append(i)
            output.append(i)
            output.append(i)
            output.append(i)
            output.append(i)
            output.append(i)
            output.append(i)
    return output

def neighbors2d_outer_j(inputSize):
    output = []
    for k in range(inputSize):
        for m in range(inputSize):
            output.append(coord_to_index(m-2,k,inputSize))
            output.append(coord_to_index(m-2,k+1,inputSize))
            output.append(coord_to_index(m-2,k+2,inputSize))
            output.append(coord_to_index(m-1,k+2,inputSize))
            output.append(coord_to_index(m,k+2,inputSize))
            output.append(coord_to_index(m+1,k+1,inputSize))
            
            output.append(coord_to_index(m+2,k,inputSize))
            output.append(coord_to_index(m+2,k-1,inputSize))
            output.append(coord_to_index(m+2,k-2,inputSize))
            output.append(coord_to_index(m+1,k-2,inputSize))
            output.append(coord_to_index(m,k-2,inputSize))
            output.append(coord_to_index(m-1,k-1,inputSize))
    return output

def set_all_to_all(pre_size, post_size, p = 1):
    output_i = []
    output_j = []
    for k in range(post_size):
        post_idx = sorted(random.sample(range(pre_size), int(pre_size*p)))
        for m in post_idx:
            output_i.append(int(m))
            output_j.append(int(k))
    return output_i, output_j
