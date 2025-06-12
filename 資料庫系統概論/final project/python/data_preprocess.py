import pandas as pd

data = pd.read_csv('/Users/KJL0508/Documents/NYCU/DataBase/Project/Database_Final_project/cellphones users.csv')
data['occupation'] = data['occupation'].str.lower()
data.to_csv('cellphones_users_modified.csv', index=False)