import math
from scipy.special import betainc

def p2l_bound(k, n, delta):
    if k == n:
        return 1
    
    t1 = k/n
    t2 = 1
    while t2 - t1 > 1e-10:
        t = (t1 + t2) / 2

        left = (delta / 2 - delta / 6) * betainc(k+1, n-k, t)
        left += (delta / 6 ) * betainc(k+1, 4*n+1-k, t)
        right = (1+delta / 6 /n) * t * n * (betainc(k, n-k+1, t) - betainc(k+1, n-k, t))
        if left > right:
            t2 = t
        else:
            t1 = t

    return t2


def p2l_upper_bound(k, n, delta):
    bound = k/n + 2 * math.sqrt(k+1) * (math.sqrt(math.log(k+1))+4)/n
    bound += 2 * math.sqrt(k+1) * math.sqrt(math.log(1/delta))/n + math.log(1/delta)/n
    return bound

def compute_all_p2l_bounds(k, n, delta, information_dict):
    information_dict['p2l_bound'] = p2l_bound(k, n, delta)
    print("P2L bound with numerical evaluation :", information_dict['p2l_bound'])
    
    information_dict['p2l_upper_bound'] = p2l_upper_bound(k, n, delta)
    print("P2L upper bound :", information_dict['p2l_upper_bound'])