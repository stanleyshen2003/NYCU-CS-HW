import scipy.optimize as opt
import numpy as np

def func(R, *data):
    cashVec, power, rate = data
    ans = 0
    for i, cash in enumerate(cashVec):
        ans += cash / ((1 + R / rate) ** (i * power))
    return ans

def irrFind(cashFlowVec, cashFlowPeriod, compoundPeriod):
    compound_rate = 12 // compoundPeriod
    power = cashFlowPeriod / compoundPeriod

    result = opt.fsolve(func, 0, (cashFlowVec, power, compound_rate))
    return result[0]