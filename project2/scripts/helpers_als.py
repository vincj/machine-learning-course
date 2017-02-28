# -*- coding: utf-8 -*-
"""some functions for help."""

from itertools import groupby

import numpy as np
import scipy.sparse as sp
import csv


import fileinput

def prepare_csv_for_als(path_data):
    with fileinput.FileInput(path_data, inplace=True, backup='.txt') as file: # sampleSubmission_woHeader
        for line in file:
            print(line.replace("_", ","), end='')
        
    with fileinput.FileInput(path_data, inplace=True, backup='.txt') as file:
        for line in file:
            print(line.replace("r", ""), end='')
        
    with fileinput.FileInput(path_data, inplace=True, backup='.txt') as file:
        for line in file:
            print(line.replace("c", ""), end='')
            
            
def deal_line(line):
    itemId, rating, userId, pred = line.split(',') # such is the ordering in the csv file created from 
    return int(itemId), int(userId), round(float(pred))
           
    
import csv
def make_submission(input_file, output_file):
    with open(input_file, 'r') as f:
        data = f.read().splitlines()
        data_transit = [deal_line(line) for line in data[1:]]
    
    with open(output_file, 'w') as csvfile:
        fieldnames = ['Id', 'Prediction']
        writer = csv.DictWriter(csvfile, delimiter=",", fieldnames=fieldnames)
        writer.writeheader()
    
        data_transit.sort(key=lambda line: (line[1],line[0])) # order the coordinates by columns
        for item_ in data_transit:
            writer.writerow({'Id':'r'+'{}'.format(item_[0])+'_c'+'{}'.format(item_[1]),'Prediction':float(item_[2])})
