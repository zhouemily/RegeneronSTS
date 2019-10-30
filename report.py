#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Jun 30 03:16:03 2019

@author: Emily
"""

import os
import sys

if not sys.argv[1]:
    print("error: need file name")
    sys.exit(0)

fname=sys.argv[1]
fh=open(fname,"r")
kwords=["python idt.py","mytopo","myact","myepoch","mybatch","myloss","myval_loss","real"]
v=[]
for line in fh.readlines():
    line=line.strip("\n\r")
    if "python idt" in line:
        print(">>"+line+"<<")
        line=line.replace("++ python idt.py dataset ","")
        a=line.split(" ")
        v.append(a[0])
        v.append(","+a[1])
        v.append(","+a[2])
        v.append(","+a[3])
    if "myloss" in line:
        a=line.split("=")
        print(a[1].lstrip("\n")+"<<") 
        v.append(","+a[1])
    if "myval_loss" in line:
        a=line.split("=")
        print(a[1].lstrip("\n")+"<<") 
        v.append(","+a[1])
    if "real" in line:
        print(line+"<<")
        a=line.split("\t")
        print(a[1].lstrip("\n")) 
        a[1]=a[1]+"\n"
        print("\n")
        a[1]=a[1].replace("0m","")
        v.append(","+a[1])

fh.close()

s=""
s=s.join(v)
print (s)
