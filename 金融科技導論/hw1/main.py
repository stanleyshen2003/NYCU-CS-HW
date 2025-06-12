import sys
from irrFind import irrFind

if __name__ == "__main__":
    for i, input_line in enumerate(sys.stdin.readlines()):
        input_numbers = [int(x) for x in input_line.strip().split()]
        cashFlowPeriod, compoundPeriod = input_numbers[-2:]
        cashFlowVec = input_numbers[:-2]
        irr = irrFind(cashFlowVec, cashFlowPeriod, compoundPeriod)
        print(f'{round(irr * 100, 4):.4f}') # IRR in percentage
            

        
