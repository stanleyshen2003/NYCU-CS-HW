# -*- coding: utf-8 -*-

import pandas as pd
df1 = pd.read_csv(r"C:\Users\stanl\OneDrive - 國立陽明交通大學\桌面\資料庫\Database_Final_project\data.csv")
num = 0
for i in df1["cellphone_id"]:
    num+=1
for i in range(num):
    year = df1["release date"][i][-4:]
    month = df1["release date"][i][3:5]
    date = df1["release date"][i][0:2]
    df1["release date"][i]=year+'/'+month+'/'+date
    
df1.to_csv(r"C:\Users\stanl\OneDrive - 國立陽明交通大學\桌面\資料庫\Database_Final_project\data_modified.csv", index=False)