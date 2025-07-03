from homlib import Graph, Graphon, countHomGraphon
import itertools
HH = Graph([[0, 1, 0],
 [1, 0, 1],
 [0, 1, 0]])

GG=Graphon([[0.5, 0.3, 0.2],
 [0.3, 0.4, 0.1],
 [0.2, 0.1, 0.6]])

print(countHomGraphon(HH, GG))

HH = [[0, 1, 0],
 [1, 0, 1],
 [0, 1, 0]]

GG = [[0.5, 0.3, 0.2],
 [0.3, 0.4, 0.1],
 [0.2, 0.1, 0.6]]

def compute_t_G_W(H, W_block):
    # Compute t for nxn graphon, brute force n^|V(H)| calculations
    n = len(H)  # Fixed: use len(H) instead of len(H[0])
    edges = [(i, j) for i in range(n) for j in range(i + 1, n) if H[i][j] == 1 or H[j][i] == 1]
    num_blocks = len(W_block)  # Fixed: use len(W_block) instead of len(W_block[0])
    block_volume = 1.0 / num_blocks
    t = 0.0
    for assignment in itertools.product(range(num_blocks), repeat=n):
        prob = 1.0
        for (u, v) in edges:
            prob *= W_block[assignment[u]][assignment[v]]  # Fixed: W_block[i][j] instead of W_block[i, j]
        t += prob * (block_volume ** n)
    return t

print(compute_t_G_W(HH, GG))