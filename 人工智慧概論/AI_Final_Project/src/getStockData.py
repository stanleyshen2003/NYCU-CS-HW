import pandas as pd
import requests, json
import time

"""
Web crawler
"""

headers = {'User-Agent' : 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/113.0.0.0 Safari/537.36 Edg/113.0.1774.35'}

def get_stock_data(year, month, stock_no):
    date = str(year+month+'01')
    print(date)
    html = requests.get('https://www.twse.com.tw/exchangeReport/STOCK_DAY?response=json&date=%s&stockNo=%s' % (date,stock_no), headers = headers)
    content = json.loads(html.text)
    return content

def get_stock_by_list(year_list, month_list, stock_list):
    if len(year_list)==0 or len(month_list)==0 or len(stock_list)==0:
        return
    
    #print(content.keys())
    for stock_no in stock_list:
        content = get_stock_data(year_list[0], month_list[0], stock_no)
        first = True
        for year in year_list:
            #print(year)
            for i in range(0,len(month_list)):
                if first:
                    first = False
                    continue
                time.sleep(3)
                content2 = get_stock_data(year, month_list[i], stock_no)
                if 'data' in content2: 
                    content['data'].extend(content2['data'])
        df = pd.DataFrame(data=content['data'])
        df.head()
        df.to_csv('stock_data_'+stock_no+'_new.csv', index=False)
    
    
    return

year = ['2016','2017','2018','2019','2020','2021','2022']
month =[str(i).zfill(2) for i in range(1,13)]
#month = ['01','02','03','04','05','06','07','08','09','10','11','12']
stock_no = ['2330']#,'2317', '2412','6505','2308','2881','2882','2303']

get_stock_by_list(year, month, stock_no)


    




