import numpy as np

# A simple greedy approach
def myActionSimple(priceMat, transFeeRate):
    # Explanation of my approach:
	# 1. Technical indicator used: Watch next day price
	# 2. if next day price > today price + transFee ==> buy
    #       * buy the best stock
	#    if next day price < today price + transFee ==> sell
    #       * sell if you are holding stock
    # 3. You should sell before buy to get cash each day
    # default
    cash = 1000
    hold = 0
    # user definition
    nextDay = 1
    dataLen, stockCount = priceMat.shape  # day size & stock count   
    stockHolding = np.zeros((dataLen,stockCount))  # Mat of stock holdings
    actionMat = []  # An k-by-4 action matrix which holds k transaction records.
    
    for day in range( 0, dataLen-nextDay ) :
        dayPrices = priceMat[day]  # Today price of each stock
        nextDayPrices = priceMat[ day + nextDay ]  # Next day price of each stock
        
        if day > 0:
            stockHolding[day] = stockHolding[day-1]  # The stock holding from the previous action day
        
        buyStock = -1  # which stock should buy. No action when is -1
        buyPrice = 0  # use how much cash to buy
        sellStock = []  # which stock should sell. No action when is null
        sellPrice = []  # get how much cash from sell
        bestPriceDiff = 0  # difference in today price & next day price of "buy" stock
        stockCurrentPrice = 0  # The current price of "buy" stock
        
        # Check next day price to "sell"
        for stock in range(stockCount) :
            todayPrice = dayPrices[stock]  # Today price
            nextDayPrice = nextDayPrices[stock]  # Next day price
            holding = stockHolding[day][stock]  # how much stock you are holding
            
            if holding > 0 :  # "sell" only when you have stock holding
                if nextDayPrice < todayPrice*(1+transFeeRate) :  # next day price < today price, should "sell"
                    sellStock.append(stock)
                    # "Sell"
                    sellPrice.append(holding * todayPrice)
                    cash = holding * todayPrice*(1-transFeeRate) # Sell stock to have cash
                    stockHolding[day][sellStock] = 0
        
        # Check next day price to "buy"
        if cash > 0 :  # "buy" only when you have cash
            for stock in range(stockCount) :
                todayPrice = dayPrices[stock]  # Today price
                nextDayPrice = nextDayPrices[stock]  # Next day price
                
                if nextDayPrice > todayPrice*(1+transFeeRate) :  # next day price > today price, should "buy"
                    diff = nextDayPrice - todayPrice*(1+transFeeRate)
                    if diff > bestPriceDiff :  # this stock is better
                        bestPriceDiff = diff
                        buyStock = stock
                        stockCurrentPrice = todayPrice
            # "Buy" the best stock
            if buyStock >= 0 :
                buyPrice = cash
                stockHolding[day][buyStock] = cash*(1-transFeeRate) / stockCurrentPrice # Buy stock using cash
                cash = 0
                
        # Save your action this day
        if buyStock >= 0 or len(sellStock) > 0 :
            action = []
            if len(sellStock) > 0 :
                for i in range( len(sellStock) ) :
                    action = [day, sellStock[i], -1, sellPrice[i]]
                    actionMat.append( action )
            if buyStock >= 0 :
                action = [day, -1, buyStock, buyPrice]
                actionMat.append( action )
    return actionMat

# A DP-based approach to obtain the optimal return
def myAction01(priceMat, transFeeRate):
    actionMat = []  # An k-by-4 action matrix which holds k transaction records.
    init_cash = 1000  # Initial available capital
    
    timeCount, stockCount = priceMat.shape
    
    # initialize cash
    c_record = np.zeros((timeCount, 2))
    c_record[0, 0] = init_cash
    c_record[0, 1] = -1
    
    # initialize stock
    s_record = np.zeros((timeCount, stockCount, 2))
    for i in range(stockCount):
        s_record[0, i, 0] = init_cash / priceMat[0, i] * (1 - transFeeRate)
        s_record[0, i, 1] = -1
    
    # for all time
    for i in range(1, timeCount):
        # hold stock & cash
        c_record[i, 0] = c_record[i - 1, 0]
        c_record[i, 1] = -1
        for j in range(stockCount):
            s_record[i, j, 0] = s_record[i - 1, j, 0]
            s_record[i, j, 1] = j
        
        for j in range(stockCount):
            # sell stock to cash
            sell_cash = s_record[i-1, j, 0] * priceMat[i, j] * (1 - transFeeRate)
            if c_record[i, 0] < sell_cash:
                c_record[i, 0] = sell_cash
                c_record[i, 1] = j
        
            # update all stock
            # from stock k to stock j
            for k in range(stockCount):
                if k == j:
                    continue
                new_stock_num = s_record[i-1, k, 0] * priceMat[i, k] * (1 - transFeeRate) / priceMat[i, j] * (1 - transFeeRate)
                if new_stock_num > s_record[i, j, 0]:
                    s_record[i, j, 0] = new_stock_num
                    s_record[i, j, 1] = k
            # from cash to stock j
            new_stock_num = c_record[i-1, 0] / priceMat[i, j] * (1 - transFeeRate)
            if new_stock_num > s_record[i, j, 0]:
                s_record[i, j, 0] = new_stock_num
                s_record[i, j, 1] = -1
                
    # get action
    huge_number = 99999999999999999999
    max_result = c_record[timeCount - 1, 0]
    to_stock = -1
    # initial action
    for i in range(stockCount):
        if s_record[timeCount - 1, i, 0] * (1 - transFeeRate) * priceMat[timeCount-1, i] > max_result:
            max_result = s_record[timeCount - 1, i, 0] * (1 - transFeeRate) * priceMat[timeCount-1, i]
            to_stock = i
            
    # iterate to get action
    for i in range(timeCount - 1, -1, -1):
        if to_stock == -1:
            if c_record[i, 1] != -1:
                action = [i, int(c_record[i, 1]), -1, huge_number]
                actionMat.append(action)
                to_stock = int(c_record[i, 1])
            
        else:
            if s_record[i, to_stock, 1] != to_stock:
                action = [i, int(s_record[i, to_stock, 1]), to_stock, huge_number]
                actionMat.append(action)
                to_stock = int(s_record[i, to_stock, 1])
        
    actionMat.reverse()
        
    return actionMat

def get_increase(priceMat, transFeeRate, trajectory, trajectory_after_trans, i):
    if trajectory_after_trans[i] == -1:
        return 99999999999999999999
        
    # if buy stock and next is hold stock
    elif trajectory[i] == -1 and trajectory_after_trans[i+1] == trajectory[i+1]:
        return priceMat[i+1, trajectory[i+1]] / priceMat[i, trajectory[i+1]]
        
    # if buy stock and next is sell or buy another stock
    elif trajectory[i] == -1:
        return priceMat[i+1, trajectory[i+1]] / priceMat[i, trajectory[i+1]] * (1 - transFeeRate) * (1 - transFeeRate)
        
    # if hold stock and next is hold stock
    elif trajectory[i] == trajectory_after_trans[i] and trajectory_after_trans[i+1] == trajectory[i]:
        return  priceMat[i+1, trajectory[i]] / (priceMat[i, trajectory[i]] * (1 - transFeeRate) * (1 - transFeeRate))
    
    # if hold stock and next is sell by another stock
    elif trajectory[i] == trajectory_after_trans[i]:
        return priceMat[i+1, trajectory[i]] / priceMat[i, trajectory[i]]
    
    # if buy another stock and next is hold stock
    elif trajectory_after_trans[i+1] == trajectory[i+1]:
        return priceMat[i+1, trajectory[i+1]] / priceMat[i, trajectory[i+1]]
    
    # if buy another stock and next is sell by another stock
    else:
        return priceMat[i+1, trajectory[i+1]] / priceMat[i, trajectory[i+1]] * (1 - transFeeRate) * (1 - transFeeRate)

# An approach that allow non-consecutive K days to hold all cash without any stocks
def myAction02(priceMat, transFeeRate, K):
    actionMat = []  # An k-by-4 action matrix which holds k transaction records.
    init_cash = 1000  # Initial available capital
    
    timeCount, stockCount = priceMat.shape
    
    # initialize cash
    c_record = np.zeros((timeCount, 2))
    c_record[0, 0] = init_cash
    c_record[0, 1] = -1
    
    # initialize stock
    s_record = np.zeros((timeCount, stockCount, 2))
    for i in range(stockCount):
        s_record[0, i, 0] = init_cash / priceMat[0, i] * (1 - transFeeRate)
        s_record[0, i, 1] = -1
    
    # for all time
    for i in range(1, timeCount):
        # hold stock & cash
        c_record[i, 0] = c_record[i - 1, 0]
        c_record[i, 1] = -1
        for j in range(stockCount):
            s_record[i, j, 0] = s_record[i - 1, j, 0]
            s_record[i, j, 1] = j
        
        for j in range(stockCount):
            # sell stock to cash
            sell_cash = s_record[i-1, j, 0] * priceMat[i, j] * (1 - transFeeRate)
            if c_record[i, 0] < sell_cash:
                c_record[i, 0] = sell_cash
                c_record[i, 1] = j
        
            # update all stock
            # from stock k to stock j
            for k in range(stockCount):
                if k == j:
                    continue
                new_stock_num = s_record[i-1, k, 0] * priceMat[i, k] * (1 - transFeeRate) / priceMat[i, j] * (1 - transFeeRate)
                if new_stock_num > s_record[i, j, 0]:
                    s_record[i, j, 0] = new_stock_num
                    s_record[i, j, 1] = k
            # from cash to stock j
            new_stock_num = c_record[i-1, 0] / priceMat[i, j] * (1 - transFeeRate)
            if new_stock_num > s_record[i, j, 0]:
                s_record[i, j, 0] = new_stock_num
                s_record[i, j, 1] = -1
                
    # get action
    huge_number = 99999999999999999999
    max_result = c_record[timeCount - 1, 0]
    to_stock = -1
    # initial action
    for i in range(stockCount):
        if s_record[timeCount - 1, i, 0] * (1 - transFeeRate) * priceMat[timeCount-1, i] > max_result:
            max_result = s_record[timeCount - 1, i, 0] * (1 - transFeeRate) * priceMat[timeCount-1, i]
            to_stock = i
            
    # iterate to get action
    trajectory = []
    empty_count = 1
    for i in range(timeCount - 1, -1, -1):
        if to_stock == -1:
            if c_record[i, 1] != -1:
                to_stock = int(c_record[i, 1])
            empty_count += 1
            
        else:
            if s_record[i, to_stock, 1] != to_stock:
                to_stock = int(s_record[i, to_stock, 1])
        trajectory.append(to_stock)
        
    actionMat.reverse()
    trajectory.reverse()
    
    trajectory_after_trans = trajectory[1:].copy()
    trajectory_after_trans = trajectory_after_trans + [-1]

    increase = np.zeros(timeCount - 1)
    for i in range(timeCount - 1):
        increase_rate = get_increase(priceMat, transFeeRate, trajectory, trajectory_after_trans, i)
        increase[i] = increase_rate
        
        
    while empty_count < K:
        min_increase = np.argmin(increase)
        trajectory_after_trans[min_increase] = -1
        trajectory[min_increase + 1] = -1
        for i in range(min_increase-1, min_increase+2):
            if i < 0 or i >= timeCount - 1:
                continue
            increase_rate = get_increase(priceMat, transFeeRate, trajectory, trajectory_after_trans, i)
            increase[i] = increase_rate
        empty_count += 1
    
    for i in range(timeCount - 1):
        if trajectory[i] != trajectory_after_trans[i]:
            action = [i, trajectory[i], trajectory_after_trans[i], huge_number]
            actionMat.append(action)
        
    return actionMat



# An approach that allow consecutive K days to hold all cash without any stocks    
def myAction03(priceMat, transFeeRate, K):
    actionMat = []  # An k-by-4 action matrix which holds k transaction records.
    init_cash = 1000  # Initial available capital
    
    timeCount, stockCount = priceMat.shape
    
    # initialize cash
    c_record = np.zeros((timeCount, 2))
    c_record[0, 0] = init_cash
    c_record[0, 1] = -1
    
    # initialize stock
    s_record = np.zeros((timeCount, stockCount, 2))
    for i in range(stockCount):
        s_record[0, i, 0] = init_cash / priceMat[0, i] * (1 - transFeeRate)
        s_record[0, i, 1] = -1
    
    # for all time
    for i in range(1, timeCount):
        # hold stock & cash
        c_record[i, 0] = c_record[i - 1, 0]
        c_record[i, 1] = -1
        for j in range(stockCount):
            s_record[i, j, 0] = s_record[i - 1, j, 0]
            s_record[i, j, 1] = j
        
        for j in range(stockCount):
            # sell stock to cash
            sell_cash = s_record[i-1, j, 0] * priceMat[i, j] * (1 - transFeeRate)
            if c_record[i, 0] < sell_cash:
                c_record[i, 0] = sell_cash
                c_record[i, 1] = j
        
            # update all stock
            # from stock k to stock j
            for k in range(stockCount):
                if k == j:
                    continue
                new_stock_num = s_record[i-1, k, 0] * priceMat[i, k] * (1 - transFeeRate) / priceMat[i, j] * (1 - transFeeRate)
                if new_stock_num > s_record[i, j, 0]:
                    s_record[i, j, 0] = new_stock_num
                    s_record[i, j, 1] = k
            # from cash to stock j
            new_stock_num = c_record[i-1, 0] / priceMat[i, j] * (1 - transFeeRate)
            if new_stock_num > s_record[i, j, 0]:
                s_record[i, j, 0] = new_stock_num
                s_record[i, j, 1] = -1
                
    # get action
    huge_number = 99999999999999999999
    max_result = c_record[timeCount - 1, 0]
    to_stock = -1
    # initial action
    for i in range(stockCount):
        if s_record[timeCount - 1, i, 0] * (1 - transFeeRate) * priceMat[timeCount-1, i] > max_result:
            max_result = s_record[timeCount - 1, i, 0] * (1 - transFeeRate) * priceMat[timeCount-1, i]
            to_stock = i
            
    # iterate to get action
    trajectory = []
    empty_count = 1
    for i in range(timeCount - 1, -1, -1):
        if to_stock == -1:
            if c_record[i, 1] != -1:
                to_stock = int(c_record[i, 1])
            empty_count += 1
            
        else:
            if s_record[i, to_stock, 1] != to_stock:
                to_stock = int(s_record[i, to_stock, 1])
        trajectory.append(to_stock)
        
    actionMat.reverse()
    trajectory.reverse()
    
    trajectory_after_trans = trajectory[1:].copy()
    trajectory_after_trans = trajectory_after_trans + [-1]

    increase = np.zeros(timeCount - 1 - K)
    
    for i in range(timeCount - 1 - K):
        # end in cash, start in cash
        if (trajectory[i] == -1):
            new_start_cash = c_record[i, 0]
        else:
            new_start_cash = s_record[i, trajectory[i], 0] * priceMat[i, trajectory[i]] * (1 - transFeeRate)
        if (trajectory[i + K] == -1):
            new_end_cash = c_record[i + K, 0]
        else:
            new_end_cash = s_record[i + K, trajectory[i + K], 0] * priceMat[i + K, trajectory[i + K]] * (1 - transFeeRate)
        increase[i] = new_end_cash / new_start_cash
        
    minimum_increase = np.argmin(increase)
    for i in range(minimum_increase, minimum_increase + K):
        trajectory_after_trans[i] = -1
        trajectory[i + 1] = -1
        
    for i in range(timeCount - 1):
        if trajectory[i] != trajectory_after_trans[i]:
            action = [i, trajectory[i], trajectory_after_trans[i], huge_number]
            actionMat.append(action)

            
    return actionMat