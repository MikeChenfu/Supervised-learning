import math
import pandas as pd
import numpy as np
import argparse
import matplotlib.pyplot as plt
from collections import Counter

def featureReport(inputfile, output):
    
    df = pd.read_csv(inputfile,header=None)
    f = open(output, mode='w')
 
    cols = ['AAGE', 'ACLSWKR', 'ADTIND', 'ADTOCC', 'AHGA', 'AHRSPAY',
                'AHSCOL', 'AMARITL', 'AMJIND', 'AMJOCC', 'ARACE', 'AREORGN',
                'ASEX', 'AUNMEM', 'AUNTYPE', 'AWKSTAT', 'CAPGAIN', 'CAPLOSS',
                'DIVVAL', 'FILESTAT', 'GRINREG', 'GRINST', 'HHDFMX', 'HHDREL',
                'MARSUPWT', 'MIGMTR1', 'MIGMTR3', 'MIGMTR4', 'MIGSAME', 'MIGSUN',
                'NOEMP', 'PARENT', 'PEFNTVTY', 'PEMNTVTY', 'PENATVTY', 'PRCITSHP',
                'SEOTR', 'VETQVA', 'VETYN', 'WKSWORK', 'YEAR', 'TARGET']	   
   
    count = 0
    print (len(cols), "features", file=f)
    df.columns = cols
    nrow = df.values.shape[0]
     
    ### histogram of each level of features
    for i in df.columns:
        print ("=============",i,"   :",count," feature type: ",df[i].dtype.name,"==================", file=f)
        print ("key length: ",len(Counter(df[i].values).keys()), file=f)
        
        for k,v in Counter(df[i].values).items():
            if i is not "instance weight":
                print ("%2.4f" % round(v/float(nrow)*100,4),"%","   ",k, file=f)
        
        print ("\n", file=f)
        count += 1

    f.close()


if __name__ == "__main__":
    
    parser = argparse.ArgumentParser()
    parser.add_argument("-i", "--input", help="input file.", required=True)
    parser.add_argument("-o", "--output", default='features_analysis.txt', help="output file")
    args = parser.parse_args()
    
    featureReport(args.input,args.output)

