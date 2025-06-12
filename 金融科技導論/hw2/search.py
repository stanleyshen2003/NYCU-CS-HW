import pandas as pd
import numpy as np
import sys
from tqdm import tqdm
from operator import itemgetter



def myStrategy(pastPriceVec, currentPrice, window_size, low_threshold, high_threshold):
	# buy return 1
	# sell return -1
	# hold return 0
	action = 0

	if len(pastPriceVec) + 1 < window_size:
		return action
	
	pastPriceVec = pastPriceVec[-window_size:]

	PriceVec = np.concatenate((pastPriceVec, np.array([currentPrice], dtype=np.float64)), axis=0)

	delta = np.diff(PriceVec)
	
	pricesUp = delta.copy()
	pricesDown = delta.copy()
	
	pricesUp[pricesUp < 0] = 0
	pricesDown[pricesDown > 0] = 0
	pricesUp = np.sum(pricesUp)
	pricesDown = - np.sum(pricesDown)

	rs = pricesUp/(pricesDown+1e-9)
	
	rsi = 100.0 - (100.0 / (1.0 + rs))
	if rsi > high_threshold:
		action = -1
	elif rsi < low_threshold:
		action = 1
	return action

def rrEstimate(priceVec, window_size, low_threshold, high_threshold):
	capital=1000	# Initial available capital
	capitalOrig=capital		# original capital
	dataCount=len(priceVec)				# day size
	suggestedAction=np.zeros((dataCount,1))	# Vec of suggested actions
	stockHolding=np.zeros((dataCount,1))  	# Vec of stock holdings
	total=np.zeros((dataCount,1))	 	# Vec of total asset
	realAction=np.zeros((dataCount,1))	# Real action, which might be different from suggested action. For instance, when the suggested action is 1 (buy) but you don't have any capital, then the real action is 0 (hold, or do nothing). 
	# Run through each day
	for ic in range(dataCount):
		currentPrice=priceVec[ic]	# current price
		suggestedAction[ic]=myStrategy(priceVec[0:ic], currentPrice, window_size, low_threshold, high_threshold)		# Obtain the suggested action
		# get real action by suggested action
		if ic>0:
			stockHolding[ic]=stockHolding[ic-1]	# The stock holding from the previous day
		if suggestedAction[ic]==1:	# Suggested action is "buy"
			if stockHolding[ic]==0:		# "buy" only if you don't have stock holding
				stockHolding[ic]=capital/currentPrice # Buy stock using cash
				capital=0	# Cash
				realAction[ic]=1
		elif suggestedAction[ic]==-1:	# Suggested action is "sell"
			if stockHolding[ic]>0:		# "sell" only if you have stock holding
				capital=stockHolding[ic]*currentPrice # Sell stock to have cash
				stockHolding[ic]=0	# Stocking holding
				realAction[ic]=-1
		elif suggestedAction[ic]==0:	# No action
			realAction[ic]=0
		else:
			assert False
		total[ic]=capital+stockHolding[ic]*currentPrice	# Total asset, including stock holding and cash 
	returnRate=(total[-1].item()-capitalOrig)/capitalOrig		# Return rate of this run
	return returnRate


if __name__=='__main__':
	returnRateBest=-1.00	 # Initial best return rate
	df=pd.read_csv(sys.argv[1])	# read stock file
	adjClose=df["Adj Close"].values		# get adj close as the price vector
	result = []
	min_window_size = 20
	max_window_size = 26
	min_low_threshold = 10
	max_low_threshold = 40
	min_high_threshold = 60
	max_high_threshold = 90
	# Start exhaustive search
	step_window = 1
	step_low = 1
	step_high = 1
	pbar = tqdm(range(min_window_size, max_window_size+1, step_window))
	for windowSize in pbar:		# For-loop for windowSize
		# tqdm.write(f"windowSize={windowSize}")
		for low_threshold in range(min_low_threshold, max_low_threshold+1, step_low):		# For-loop for beta
			# print("low_threshold=%d" %(low_threshold))	# No newline
			for high_threshold in range(min_high_threshold, max_high_threshold+1, step_high):		# For-loop for beta
				# print("high_threshold=%d" %(high_threshold))
				returnRate=rrEstimate(adjClose, windowSize, low_threshold, high_threshold)		# Start the whole run with the given parameters
				# print(" ==> returnRate=%f " %(returnRate))
				pbar.set_postfix({
					"windowSize": windowSize, 
					"low_threshold": low_threshold,
					"high_threshold": high_threshold,
					"returnRate": returnRate
				})
				if returnRate > returnRateBest:		# Keep the best parameters
					windowSizeBest=windowSize
					low_thresholdBest=low_threshold
					high_thresholdBest=high_threshold
					returnRateBest=returnRate
				print(f"Best Return rate {returnRateBest}")
				result.append([returnRate, windowSize, low_threshold, high_threshold])
	result = sorted(result, key=itemgetter(0), reverse=True)
	result = pd.DataFrame(result)
	result.to_csv('result2.csv')
	print("Best settings: windowSize=%d, alpha=%d, beta=%d ==> returnRate=%f" %(windowSizeBest,low_thresholdBest, high_thresholdBest,returnRateBest))		# Print the best result