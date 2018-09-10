#!/usr/bin/env python3
import pandas as pd
import numpy as np
import warnings
import pickle
import argparse
import matplotlib.pyplot as plt

from func.preprocess import *

pd.options.mode.chained_assignment = None
warnings.filterwarnings("ignore", category=DeprecationWarning)

def prediction(inputfile, model):
    #read in data
    try:
        data = pd.read_csv(inputfile, header=None)
        print("Data loaded")
    except:
        print("Dataset could not be loaded. Is the dataset missing?")

    data.columns = ['AAGE', 'ACLSWKR', 'ADTIND', 'ADTOCC', 'AHGA', 'AHRSPAY',
                    'AHSCOL', 'AMARITL', 'AMJIND', 'AMJOCC', 'ARACE', 'AREORGN',
                    'ASEX', 'AUNMEM', 'AUNTYPE', 'AWKSTAT', 'CAPGAIN', 'CAPLOSS',
                    'DIVVAL', 'FILESTAT', 'GRINREG', 'GRINST', 'HHDFMX', 'HHDREL',
                    'MIGMTR1', 'MIGMTR3', 'MIGMTR4', 'MIGSAME', 'MIGSUN',
                    'NOEMP', 'PARENT', 'PEFNTVTY', 'PEMNTVTY', 'PENATVTY', 'PRCITSHP',
                    'SEOTR', 'VETQVA', 'VETYN', 'WKSWORK', 'YEAR']


    # preprocess data
    data = preprocess(data)
    # load model file
    try:
        loaded_model = pickle.load(open(model, 'rb'))
        print("Model loaded")
    except:
        print("Model could not be loaded. Is the model missing?")

    # predict run
    result = loaded_model.predict(data)
    # show result
    print(result)

if __name__ == "__main__":
    """
    Run main program.
    """
    parser = argparse.ArgumentParser()
    parser.add_argument("-i", "--input", help="input file.", required=True)
    parser.add_argument("-m", "--model",default='classify_model.sav', help="model file")
    args = parser.parse_args()
    prediction(args.input, args.model)














