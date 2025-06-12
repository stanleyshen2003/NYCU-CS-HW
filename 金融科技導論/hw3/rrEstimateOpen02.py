import sys
import numpy as np
import pandas as pd
from myAction import *
import time
import copy

# Compute return rate over a given price Matrix & action Matrix
def computeReturnRate(priceMat, transFeeRate, actionMat, K, problem_type):

	capital = 1000	  # Initial available capital
	capitalOrig = capital	  # original capital
	stockCount = len(priceMat[0])	# stack size
	suggestedAction = actionMat	   # Mat of suggested actions
	actionCount = len(suggestedAction)

	if len(actionMat) == 0:
		cashHolding_day = priceMat.shape[0]
		cash_longest_day = cashHolding_day
		returnRate = 0
		return returnRate, cashHolding_day, cash_longest_day
	
	stockHolding = np.zeros((actionCount,stockCount))	# Mat of stock holdings
	realAction = np.zeros((actionCount,1))	  # Real action, which might be different from suggested action. For instance, when the suggested action is 1 (buy) but you don't have any capital, then the real action is 0 (hold, or do nothing). 
	preDay = 0	# previous action day

	# Run through each action, should order by day
	for i in range(actionCount):

		actionVec = actionMat[ i ]
		day = actionVec[0] # The index of day
		a = actionVec[1] # The index of "from" stock
		b = actionVec[2] # The index of "to" stock
		z = actionVec[3] # The equivalent cash for such transaction.
		currentPriceVec = priceMat[day]	 # current priceVec
		

		# check action day
		if day >= preDay and day >= 0 and z > 0 :
			# get real action by suggested action
			if i > 0:
				stockHolding[i] = stockHolding[i-1]	 # The stock holding from the previous action day
				preDay = day  # previous action day
			
			if a == -1 and b >= 0 and capital > 0 :	 # Suggested action is "buy"
				currentPrice = currentPriceVec[b]  # The current price of stock
				if capital < z :  # "buy" all only if you don't have enough capital
					z = capital
				stockHolding[i][b] += z*(1-transFeeRate) / currentPrice # Buy stock using cash
				capital = capital - z  # Cash
				realAction[i] = 1

			elif b == -1 and a >= 0 and stockHolding[i][a] > 0 :  # Suggested action is "sell"
				currentPrice = currentPriceVec[a]  # The current price of stock
				sellStock = z / currentPrice
				if stockHolding[i][a] < sellStock :  # "sell" all only if you don't have enough stock holding
					sellStock = stockHolding[i][a]
				getCash = sellStock * currentPrice*(1-transFeeRate)	 # Sell stock to have cash
				capital = capital + getCash	 # get cash from sell stock
				stockHolding[i][a] -= sellStock	 # Stocking holding
				realAction[i] = -1
			elif a >= 0 and b >= 0 and stockHolding[i][a] > 0 :  # Suggested action is "buy" and "sell"
				currentPriceSell = currentPriceVec[a]  # The current price of sell stock
				currentPriceBuy = currentPriceVec[b]  # The current price of buy stock
				sellStock = z / currentPriceSell
				if stockHolding[i][a] < sellStock :  # "sell" all only if you don't have enough stock holding
					sellStock = stockHolding[i][a]
				getCash = sellStock * currentPriceSell*(1-transFeeRate)	 # Sell stock to have cash
				stockHolding[i][a] -= sellStock	 # Stocking holding
				stockHolding[i][b] += getCash*(1-transFeeRate) / currentPriceBuy # Buy stock using cash
				realAction[i] = 2
			else:
				print("day:", day)
				print("a:", a, "b:", b, "capital:", capital, "stockHolding:", stockHolding[i][a], "z:", z)
				assert False
		else:
			assert False


	## calculate holding cash day
	preDay = 0
	stockHolding_day = np.zeros(priceMat.shape[0])
	preStock = 0
	cash_longest_day = 0

	for idx, stock in enumerate(stockHolding):

		day = actionMat[idx][0]
		if idx == actionCount - 1  or day != actionMat[idx+1][0]:
			sum_stock = sum(stock)
			stockHolding_day[preDay:day+1] = preStock
			if preStock < 1e-7:
				cash_longest_day = max(cash_longest_day, day - preDay)
			preDay = copy.deepcopy(day)
			preStock = copy.deepcopy(sum_stock)


	stockHolding_day[preDay:-1] = preStock
	stockHolding_day[-1] = 0
	if preStock < 1e-7:
		cash_longest_day = max(cash_longest_day, priceMat.shape[0] - preDay)
	cashHolding_day = len(np.where(stockHolding_day < 1e-7)[0])


			
	# calculate total cash you get at last day
	total = capital

	for stock in range(stockCount) :
		currentPriceVec = priceMat[-1]
		total += stockHolding[-1][stock] * currentPriceVec[stock]*(1-transFeeRate)	# Total asset, including stock holding and cash 
	returnRate=(total-capitalOrig)/capitalOrig	# Return rate of this run

	if problem_type == 2 and cashHolding_day < K:
		returnRate = 0

	if problem_type == 3 and cash_longest_day < K:
		returnRate = 0

	return returnRate, cashHolding_day, cash_longest_day
	
if __name__ == "__main__":

	print("Reading %s..." %(sys.argv[1]))
	file = sys.argv[1]	  # input file
	df = pd.read_csv(file, delimiter=' ', header = None)
	transFeeRate= float(sys.argv[2])	# Rate for transaction fee
	priceMat = df.values


	K = 0
	problem_type = 1
	print("------------Problem 1-------------")
	start = time.time()
	actionMat = myAction01(priceMat, transFeeRate)	# Obtain the suggested action
	rr, cashHolding, cash_longest_day = computeReturnRate(priceMat, transFeeRate, actionMat, K, problem_type)  # Compute return rate
	end = time.time()
	print("Time:", end - start)
	print("rr=%f%%" %(rr*100))
	print("Non continueous cash holding=%d" %(cashHolding))
	print("Continueous cash holding=%d" %(cash_longest_day))

	K_list = [200, 300, 400]
	problem_type = 2
	print("------------Problem 2-------------")
	start = time.time()
	total_rr = 0
	for K in K_list:
		actionMat = myAction02(priceMat, transFeeRate, K)	# Obtain the suggested action
		rr, cashHolding, cash_longest_day = computeReturnRate(priceMat, transFeeRate, actionMat, K, problem_type)  # Compute return rate
		total_rr += rr
	
	end = time.time()
	print("Time:", end - start)
	print("rr=%f%%" %(total_rr*100/3))
	print("Non continueous cash holding=%d" %(cashHolding))
	print("Continueous cash holding=%d" %(cash_longest_day))


	K_list = [200, 300, 400]
	problem_type = 3
	print("------------Problem 3-------------")
	start = time.time()
	total_rr = 0
	for K in K_list:
		actionMat = myAction03(priceMat, transFeeRate, K)	# Obtain the suggested action
		rr, cashHolding, cash_longest_day = computeReturnRate(priceMat, transFeeRate, actionMat, K, problem_type)  # Compute return rate
		total_rr += rr

	end = time.time()
	print("Time:", end - start)
	print("rr=%f%%" %(total_rr*100/3))
	print("Non continueous cash holding=%d" %(cashHolding))
	print("Continueous cash holding=%d" %(cash_longest_day))
