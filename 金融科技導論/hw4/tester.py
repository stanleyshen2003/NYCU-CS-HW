from mySubmission import compute4G
from mySubmission import compute5G
from mySubmission import double_and_add
from mySubmission import optimized_double_and_add
from mySubmission import GetCurveParameters
from mySubmission import sign_transaction
from mySubmission import verify_signature
from main import *
import sys
import random
from ecdsa import ellipticcurve

def test_mul():
    G = getG()
    for i in range(0, 1000):
        # print(f"Testing {i}G")
        point = double_and_add(i, G, getINFINITY)[0]
        point2 = optimized_double_and_add(i, getG(), getINFINITY)[0]
        point_correct = i * G
        assert point == point_correct, f"double_and_add failed for {i}G"
        assert point2 == point_correct, f"optimized_double_and_add failed for {i}G"
    print("All tests passed for double_and_add and optimized_double_and_add")
    
def test_sign():
    private_key = 0x1E99423A4ED27608A15A2616CFDB24B88504FF82F39A3D84F07E3C5C8E69A9E5
    hashID = "1d0f172a0ecb4a6f8f7e8b3b97c8e9ac6f8e5b3c2d6c6d3c6f8e5b3c2d6c6d3"
    G = getG()
    public_key = private_key * G
    for i in range(1, 1000):
        # print(f"Testing signature {i}")
        false_key = private_key + i
        signature = sign_transaction(false_key, hashID, getG, getN, random.randint)
        assert not verify_signature(public_key, hashID, signature, getG, getN, getINFINITY), "Signature verification failed"
    
    signature = sign_transaction(private_key, hashID, getG, getN, random.randint)
    assert verify_signature(public_key, hashID, signature, getG, getN, getINFINITY), "Signature verification failed"
    print("All tests passed for sign_transaction and verify_signature")
    
    
    
test_mul()
# test_sign()
