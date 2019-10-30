#!/usr/bin/env python
"""
utility script for data curation
"""

import os
import sys
import pandas as pd


outfile="idt.csv"
filename="2data_hydrogenIDT.xlsx"
#filename="edata.xlsx"
path="../dataset/"+filename
xls = pd.ExcelFile(path)
print(xls.sheet_names)
dfnames=[]
for name in xls.sheet_names:
    dfname="df"+name
    dfname= pd.read_excel(xls, name)
    if "atm_new" in name:
        name=name.replace("atm_new","")
    elif "atm" in name:
        name=name.replace("atm","")
    rownum=len(dfname.index)
    print("rownum=",rownum)
    P=[]
    E=[]
    D=[]
    for i in range(rownum):
        P.append(name)
        E.append("1")
        D.append("0.21")
    dfname["P"]=P
    dfname["E"]=E
    dfname["D"]=D
    dfname.columns=["T","T1","Idt","P","E","D"]
    print(dfname.head())
    dfnames.append(dfname)
    del P
    del E
    del D
print(xls.sheet_names)
vstack=pd.concat(dfnames,axis=0)
print(vstack.head())
cols = vstack.columns.tolist()
print(cols)
cols=['T','T1','P','E','D','Idt']
print(cols)
vstack=vstack[cols]
print(vstack.head())
vstack.to_csv("../dataset/"+outfile, index=False)
# verify  output back into Python and make sure all looks good
output = pd.read_csv('../dataset/idt.csv', keep_default_na=False, na_values=[""])
print(output.head())

print("file path=../dataset/"+outfile)
sys.exit(0)
