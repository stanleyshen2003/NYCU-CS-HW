def egcd(a, b):
    if a == 0:
        return (b, 0, 1)
    g, y, x = egcd(b%a,a)
    return (g, x - (b//a) * y, y)

def modinv(a, m):
    g, x, y = egcd(a, m)
    if g != 1:
        raise Exception('No modular inverse')
    return x%m

#############################################################
# Problem 0: Find base point
def GetCurveParameters():
    # Certicom secp256-k1
    # Hints: https://en.bitcoin.it/wiki/Secp256k1
    _p = 0xFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFEFFFFFC2F
    _a = 0x0000000000000000000000000000000000000000000000000000000000000000
    _b = 0x0000000000000000000000000000000000000000000000000000000000000007
    _Gx = 0x79BE667EF9DCBBAC55A06295CE870B07029BFCDB2DCE28D959F2815B16F81798
    _Gy = 0x483ADA7726A3C4655DA4FBFC0E1108A8FD17B448A68554199C47D08FFB10D4B8
    _Gz = 0x0000000000000000000000000000000000000000000000000000000000000001
    _n = 0xFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFEBAAEDCE6AF48A03BBFD25E8CD0364141
    _h = 0x01
    return _p, _a, _b, _Gx, _Gy, _Gz, _n, _h


#############################################################
# Problem 1: Evaluate 4G
def compute4G(G, callback_get_INFINITY):
    """Compute 4G"""

    """ Your code here """
    # result = callback_get_INFINITY()
    return 4 * G


#############################################################
# Problem 2: Evaluate 5G
def compute5G(G, callback_get_INFINITY):
    """Compute 5G"""
    """ Your code here """
    return 5 * G


#############################################################
# Problem 3: Evaluate dG
# Problem 4: Double-and-Add algorithm
def double_and_add(n, point, callback_get_INFINITY):
    """Calculate n * point using the Double-and-Add algorithm."""

    """ Your code here """
    result = point
    num_doubles = 0
    num_additions = 0
    # result = None
    if n == 0:
        return callback_get_INFINITY(), 0, 0
    binary = bin(n)[2:]
    
    for i in range(1, len(binary)):
        result = result.double()
        num_doubles += 1
        if binary[i] == '1':
            result = result + point
            num_additions += 1

    return result, num_doubles , num_additions


#############################################################
# Problem 5: Optimized Double-and-Add algorithm
def optimized_double_and_add(n, point, callback_get_INFINITY):
    """Optimized Double-and-Add algorithm that simplifies sequences of consecutive 1's."""

    """ Your code here """
    result = callback_get_INFINITY()
    if n == 0:
        return result, 0, 0
    if n == 1:
        return point, 0, 0
    if n == 2:
        return point.double(), 1, 0
    if n == 3:
        return point.double() + point, 1, 1
    num_doubles = 0
    num_additions = 0
    
    binary = bin(n)[2:]
    
    consecutive_ones = [0] * len(binary)
    consecutive_zeros = [0] * len(binary)
    count = 0
    count_zeros = 0
    for i in range(len(binary)):
        if binary[i] == '1':
            count += 1
        elif binary[i] == '0' and count > 0:
            consecutive_ones[i-1] = count
            count = 0
        if binary[i] == '0':
            count_zeros += 1
        elif binary[i] == '1' and count_zeros > 0:
            consecutive_zeros[i-1] = count_zeros
            count_zeros = 0
    if count > 0:
        consecutive_ones[-1] = count
    
    consecutive_ones.reverse()
    consecutive_zeros.reverse()
    
    sequence_zeros = 0
    for i in range(1, len(consecutive_zeros)):
        if consecutive_zeros[i] == 1 and consecutive_ones[i-1] != 1 and consecutive_ones[i+1] != 1:
            sequence_zeros += 1                                         
    additions = []
    for i in range(len(consecutive_ones)):
        if consecutive_ones[i] == 2 and i == len(consecutive_ones) - 2 and consecutive_zeros[i-1] != 1:
            additions.append((i, '+'))
            additions.append((i+1, '+'))
        elif consecutive_ones[i] >= 2:
            additions.append((i, '-'))
            additions.append((i+consecutive_ones[i], '+'))
        elif consecutive_ones[i] == 1:
            additions.append((i, '+'))
    
    addition_store = additions
    addition_new = []
    while addition_new != addition_store:
        addition_store = addition_new
        addition_new = []
        for i in range(len(additions) - 2, 0, -1):
            if additions[i][0] - 1 == additions[i-1][0]:
                if additions[i][1] == '+' and additions[i-1][1] == '-':
                    addition_new.append((additions[i-1][0], '+'))
                    i -= 1
                elif additions[i][1] == '-' and additions[i-1][1] == '+':
                    addition_new.append((additions[i+1][0], '-'))
                    i -= 1
                else:
                    addition_new.append(additions[i])
            else:
                addition_new.append(additions[i])
    step = 0
    additions = addition_new
    for i in range(len(binary)+1):
        if step < len(additions) and i == additions[step][0]:
            if additions[step][1] == '+':
                result = result + point
                num_additions += 1
            elif additions[step][1] == '-':
                result = result + (-point)
                num_additions += 1
            step += 1
        point = point.double()
    num_doubles = 0 if len(additions) == 0 else additions[-1][0]
    num_additions = len(additions) - 1 if len(additions) > 0 else 0     



    return result, num_doubles, num_additions


#############################################################
# Problem 6: Sign a Bitcoin transaction with a random k and private key d
def sign_transaction(private_key, hashID, callback_getG, callback_get_n, callback_randint):
    """Sign a bitcoin transaction using the private key."""

    """ Your code here """
    G = callback_getG()
    n = callback_get_n()
    hashID = int(hashID, 16)
    
    s = 0
    r = 0
    
    while s == 0 or r == 0:
        k = callback_randint(1, n-1)
        x1 = (k * G).x()
        r = x1 % n
        s = (modinv(k, n) * (hashID + r * private_key)) % n
    
    signature = (r, s)
    return signature


##############################################################
# Step 7: Verify the digital signature with the public key Q
def verify_signature(public_key, hashID, signature, callback_getG, callback_get_n, callback_get_INFINITY):
    """Verify the digital signature."""

    """ Your code here """
    hashID = int(hashID, 16)
    G = callback_getG()
    n = callback_get_n()
    infinity_point = callback_get_INFINITY()
    is_valid_signature = True if callback_get_n() > 0 else False
    r = signature[0]
    s = signature[1]
    
    w = modinv(s, n)
    u1 = (hashID * w) % n
    u2 = (r * w) % n
    point = u1 * G + u2 * public_key
    if point == infinity_point:
        is_valid_signature = False
    else:
        v = point.x() % n
        if v != r:
            is_valid_signature = False
    

    return is_valid_signature

