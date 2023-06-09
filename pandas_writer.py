#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Created on Thu Mar 26 09:52:19 2020

@author: sjoly
"""
import pandas as pd

def save_parameters(element_name, df, pars):
            
    columns=[]
    for i in range(int(len(pars)/3)):
        columns.append('Rt'+str(i+1)+' (Ohm)')
        columns.append('Q'+str(i+1))    
        columns.append('fres'+str(i+1)+' (GHz)')
            
    se = pd.Series(pars)    
    row_df = pd.DataFrame(se).T
    row_df.index=[element_name]
    row_df.columns=columns    
    df = pd.concat([row_df, df])
    
    return df

def save_csv(df, file_name):
    df.to_csv('/home/sjoly/cernbox/Resonators/'+str(file_name)+'.csv', sep='\t', index=True, header=True)
    return
    