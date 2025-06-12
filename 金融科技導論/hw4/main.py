from mySubmission import compute4G
from mySubmission import compute5G
from mySubmission import double_and_add
from mySubmission import optimized_double_and_add
from mySubmission import GetCurveParameters
from mySubmission import sign_transaction
from mySubmission import verify_signature
import sys
import random
from ecdsa import ellipticcurve


TA_TEST_DATA=11070513


def getCurve():
    _p, _a, _b, _Gx, _Gy, _Gz, _n, _h = GetCurveParameters()
    return ellipticcurve.CurveFp(_p, _a, _b, _h)


def getG():
    _p, _a, _b, _Gx, _Gy, _Gz, _n, _h = GetCurveParameters()
    curve = getCurve()
    G = ellipticcurve.PointJacobi(curve, _Gx, _Gy, _Gz, _n, generator=True)
    return G


def getINFINITY():
    return ellipticcurve.INFINITY


def getN():
    _p, _a, _b, _Gx, _Gy, _Gz, _n, _h = GetCurveParameters()
    return _n


def point_to_hex(point):
    """Convert PointJacobi to hex format."""
    if point == ellipticcurve.INFINITY:
        return "INFINITY"
    return f"{point.x().to_bytes(32, 'big').hex()} {point.y().to_bytes(32, 'big').hex()}"


def Problem0():
    G = getG()
    print(point_to_hex(G))


def Problem1():
    #print("Problem 1: Evaluate 4G")
    point_4G = compute4G(getG(), getINFINITY)
    #point_4G, _, _ = double_and_add(4, G, getINFINITY)
    print(f"{point_to_hex(point_4G)}")


def Problem2():
    #print("Problem 2: Evaluate 5G")
    point_5G = compute5G(getG(), getINFINITY)
    #point_5G, _, _ = double_and_add(5, G, getINFINITY)
    print(f"{point_to_hex(point_5G)}")


def Problem3():
    #print("Problem 3: Evaluate Q = d G")
    point_dG, _, _ = double_and_add(TA_TEST_DATA, getG(), getINFINITY)
    print(f"{point_to_hex(point_dG)}")

def Problem4(indata):
    #print("Problem 4: Standard Double-And-Add algorithm
    result_dG, num_doubles, num_additions = double_and_add(int(indata), getG(), getINFINITY)
    print(f"{point_to_hex(result_dG)} {num_doubles} {num_additions}")


def Problem5(indata):
    #print("Problem 5: Optimized Double And Add (#Doubles, #Addiotns)")
    opt_result_dG, opt_num_doubles, opt_num_additions = optimized_double_and_add(int(indata), getG(), getINFINITY)
    print(f"{point_to_hex(opt_result_dG)} {opt_num_doubles} {opt_num_additions}")


def Problem6(private_key, transaction_hashID):
    #print("Problem 6: Sign Transaction")

    # https://www.blockchain.com/explorer/transactions/btc/4a5e1e4baab89f3a32518a88c31bc87f618f76673e2cc77ab2127b7afdeda33b
    transaction_hashID = "4a5e1e4baab89f3a32518a88c31bc87f618f76673e2cc77ab2127b7afdeda33b"
    signature = sign_transaction(private_key, transaction_hashID, getG, getN, random.randint)
    print(f"{signature[0]:x} {signature[1]:x}")


# Problem 7: Verify the digital signature with the public key Q
def Problem7(private_key, indata):
    #print("Problem 7: Verify Transaction")

    # transaction_hashID = indata[0]
    # r = int(indata[1], 16)
    # s = int(indata[2], 16)

    # https://www.blockchain.com/explorer/transactions/btc/4a5e1e4baab89f3a32518a88c31bc87f618f76673e2cc77ab2127b7afdeda33b
    transaction_hashID = "4a5e1e4baab89f3a32518a88c31bc87f618f76673e2cc77ab2127b7afdeda33b"

    r = int("d20c1a14f4f76847cc088761ac16b85561810a19cd7e8df3666be58d898c67bf", 16)
    s = int("c1f8613e4d3316ab5997bec67782824e6cf36a75ac1a42410413640620f7b7e2", 16)
    signature = (r, s)
    public_key = private_key * getG()
    
    # Verify the signature
    is_valid = verify_signature(public_key, transaction_hashID, signature,  getG, getN, getINFINITY)
    print(f"{is_valid}")


def main():
    # lineinfo = sys.stdin.read().strip().split()
    # problem = int(lineinfo[0])
    problem = int(input())
    print(f"problem: {problem}")
    match problem:
        case 0:
            Problem0()
        case 1:
            Problem1()
        case 2:
            Problem2()
        case 3:
            Problem3()
        case 4:
            # public data: TA_TEST_DATA
            # hidden data: undisclosed
            indata = 5
            #indata = TA_TEST_DATA
            #print(f"indata: {indata}")
            #print(f"outdata: {outdata}")
            Problem4(indata)
        case 5: 
            # public data: TA_TEST_DATA
            # hidden data: undisclosed
            indata = 5
            # indata = lineinfo[1]
            # outdata = lineinfo[2:]
            #indata = TA_TEST_DATA
            #print(f"indata: {indata}")
            #print(f"outdata: {outdata}")
            Problem5(indata)
        case 6:
            # public data: "4a5e1e4baab89f3a32518a88c31bc87f618f76673e2cc77ab2127b7afdeda33b"
            # hidden data: undisclosed
            # indata = lineinfo[1]
            # outdata = lineinfo[2:]
            indata = "4a5e1e4baab89f3a32518a88c31bc87f618f76673e2cc77ab2127b7afdeda33b"
            #print(f"indata: {indata}")
            #print(f"outdata: {outdata}")
            Problem6(TA_TEST_DATA, indata)
        case 7:
            # public data: "4a5e1e4baab89f3a32518a88c31bc87f618f76673e2cc77ab2127b7afdeda33b"
            # hidden data: undisclosed
            # indata = lineinfo[1:4]
            # outdata = lineinfo[4:]
            indata = ["4a5e1e4baab89f3a32518a88c31bc87f618f76673e2cc77ab2127b7afdeda33b",
                     "f9308a019258c31049344f85f89d5229b531c845836f99b08601f113bce036f9",
                     "8d89a38eb73d9528e4c1432f88ab9e3a16b4d23f333be3f88a4ce6167c019066"]
            #print(f"indata: {indata}")
            #print(f"outdata: {outdata}")
            Problem7(TA_TEST_DATA, indata)
        case _:
            print("ERROR")


if __name__ == "__main__":
    TA_TEST_DATA=int(sys.argv[1])
    main()
