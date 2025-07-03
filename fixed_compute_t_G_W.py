from decimal import Decimal, getcontext
import itertools

def compute_t_G_W_precise(H, W_block, precision=50):
    """High precision version of compute_t_G_W using Decimal arithmetic"""
    getcontext().prec = precision
    
    n = len(H)
    edges = [(i, j) for i in range(n) for j in range(i + 1, n) if H[i][j] == 1 or H[j][i] == 1]
    num_blocks = len(W_block)
    block_volume = Decimal(1) / Decimal(num_blocks)
    t = Decimal(0)
    
    for assignment in itertools.product(range(num_blocks), repeat=n):
        prob = Decimal(1)
        for (u, v) in edges:
            prob *= Decimal(str(W_block[assignment[u]][assignment[v]]))
        t += prob * (block_volume ** n)
    return float(t)

# Test it
HH = [[0, 1, 0], [1, 0, 1], [0, 1, 0]]
GG = [[0.5, 0.3, 0.2], [0.3, 0.4, 0.1], [0.2, 0.1, 0.6]]

result = compute_t_G_W_precise(HH, GG)
print(f"High precision compute_t_G_W: {result}")
print(f"CountHomLib result:           0.09074074074074075")
print(f"Match: {abs(result - 0.09074074074074075) < 1e-15}")
