import numpy as np
import time

n = 1419
T = np.random.rand(n, n)
T /= T.sum(axis=1, keepdims=True)
log_T = np.log(T + 1e-300)
V = np.random.randn(n)
tau = 1.0

# Benchmark 50 value iteration steps (our new max)
t0 = time.time()
for _ in range(50):
    Q = log_T + V[None, :] / tau
    Q_max = Q.max(axis=1, keepdims=True)
    log_sum_exp = Q_max.squeeze() + np.log(np.exp(Q - Q_max).sum(axis=1))
    V = V + 0.95 * tau * log_sum_exp
t1 = time.time()
print(f"50 value iteration steps: {t1-t0:.2f}s")

# Benchmark 20 mat-vec
visit = np.random.rand(n)
pi = T.copy()
t0 = time.time()
for _ in range(20):
    visit = visit @ pi * 0.95
t1 = time.time()
print(f"20 mat-vec multiplies: {t1-t0:.4f}s")

# Estimated time for 200 IRL iterations
vi_time = 50 * 0.026
sv_time = 20 * 0.0007
per_iter = vi_time + sv_time
total = per_iter * 200
print(f"Estimated per IRL iter: {per_iter:.2f}s")
print(f"Estimated 200 IRL iters: {total:.0f}s ({total/60:.1f}min)")
print(f"Full pipeline (primary + 3 stab + 10 null): {total*14/60:.0f}min")
