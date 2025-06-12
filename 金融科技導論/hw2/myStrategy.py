import pandas as pd
import numpy as np



def myStrategy(pastPriceVec, currentPrice, window_size=24, low_threshold=19, high_threshold=85):
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
