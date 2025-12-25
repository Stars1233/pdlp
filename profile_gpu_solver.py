"""
Profile GPU solver to find bottlenecks.
Run this on Colab to see where time is spent.
"""

import torch
import time
from collections import defaultdict

torch.set_default_dtype(torch.float64)


class GPUProfiler:
    """Context manager to profile GPU operations."""
    def __init__(self):
        self.timings = defaultdict(list)

    def __call__(self, name):
        return self._ProfileContext(self, name)

    class _ProfileContext:
        def __init__(self, profiler, name):
            self.profiler = profiler
            self.name = name

        def __enter__(self):
            if torch.cuda.is_available():
                torch.cuda.synchronize()
            self.start = time.perf_counter()
            return self

        def __exit__(self, *args):
            if torch.cuda.is_available():
                torch.cuda.synchronize()
            elapsed = time.perf_counter() - self.start
            self.profiler.timings[self.name].append(elapsed)

    def report(self):
        """Print timing report."""
        print("\n" + "="*70)
        print("GPU PROFILING REPORT")
        print("="*70)

        items = []
        for name, times in self.timings.items():
            total = sum(times)
            avg = total / len(times)
            items.append((total, avg, len(times), name))

        items.sort(reverse=True)

        total_time = sum(t[0] for t in items)

        print(f"{'Operation':<30} {'Total (s)':<12} {'Avg (ms)':<12} {'Count':<8} {'%':<8}")
        print("-"*70)

        for total, avg, count, name in items:
            pct = 100 * total / total_time if total_time > 0 else 0
            print(f"{name:<30} {total:<12.3f} {avg*1000:<12.3f} {count:<8} {pct:<8.1f}")

        print("-"*70)
        print(f"{'TOTAL':<30} {total_time:<12.3f}")
        print()


def create_transportation_problem(n_suppliers, n_customers, device='cpu'):
    """Create sparse transportation problem."""
    torch.manual_seed(42)

    n_vars = n_suppliers * n_customers

    supply = torch.rand(n_suppliers) * 20 + 10
    demand = torch.rand(n_customers) * 15 + 5

    total_demand = demand.sum()
    total_supply = supply.sum()
    if total_supply < total_demand:
        supply = supply * (total_demand / total_supply * 1.2)

    costs = torch.rand(n_suppliers, n_customers) * 5
    c = costs.flatten().to(device)

    # Build sparse constraint matrices
    G_rows, G_cols, G_vals = [], [], []

    # Supply constraints
    for i in range(n_suppliers):
        for j in range(n_customers):
            idx = i * n_customers + j
            G_rows.append(i)
            G_cols.append(idx)
            G_vals.append(-1.0)

    # Demand constraints
    for j in range(n_customers):
        for i in range(n_suppliers):
            idx = i * n_customers + j
            G_rows.append(n_suppliers + j)
            G_cols.append(idx)
            G_vals.append(1.0)

    G = torch.sparse_coo_tensor(
        torch.tensor([G_rows, G_cols]),
        torch.tensor(G_vals),
        (n_suppliers + n_customers, n_vars)
    ).to(device)

    h = torch.cat([-supply, demand]).to(device)

    A = torch.zeros(0, n_vars).to_sparse_coo().to(device)
    b = torch.tensor([]).to(device)

    l = torch.zeros(n_vars).to(device)
    u = torch.ones(n_vars).to(device) * 100

    return G, A, c, h, b, l, u


def profile_solver_iteration(device='cuda'):
    """Profile a single PDLP iteration to find bottlenecks."""
    print(f"\n{'='*70}")
    print(f"PROFILING SOLVER ITERATION ON {device.upper()}")
    print(f"{'='*70}\n")

    # Create problem
    n_suppliers, n_customers = 500, 750
    n_vars = n_suppliers * n_customers

    print(f"Problem: {n_suppliers}×{n_customers} = {n_vars:,} variables")
    print(f"Loading problem...")

    G, A, c, h, b, l, u = create_transportation_problem(n_suppliers, n_customers, device)

    K = torch.cat([G, A], dim=0)
    q = torch.cat([h, b])
    m = K.shape[0]

    print(f"K shape: {K.shape}, nnz: {K._nnz():,}")
    print()

    # Initialize variables
    x = torch.zeros(n_vars, device=device, dtype=torch.float64)
    y = torch.zeros(m, device=device, dtype=torch.float64)
    w = torch.tensor(1.0, device=device, dtype=torch.float64)
    eta = torch.tensor(0.01, device=device, dtype=torch.float64)

    # Profiler
    profiler = GPUProfiler()

    # Warmup
    if device == 'cuda':
        print("Warming up GPU...")
        for _ in range(10):
            y_new = K @ x
            x_new = K.T @ y
        torch.cuda.synchronize()
        print()

    # Profile iterations
    n_iters = 100
    print(f"Profiling {n_iters} iterations...\n")

    total_start = time.perf_counter()
    if device == 'cuda':
        torch.cuda.synchronize()

    for i in range(n_iters):
        # Gradient computation
        with profiler("K.T @ y"):
            grad_x = c - (K.T @ y)

        # Step
        with profiler("x update (projection)"):
            x_new = x - (eta / w) * grad_x
            x_new = torch.clamp(x_new, l, u)  # Box projection

        # Primal residual
        with profiler("K @ x"):
            Kx = K @ x_new

        with profiler("y update (projection)"):
            residual = q - Kx
            y_new = y + (eta * w) * residual
            # Dual projection (>= 0 for inequalities)
            m1 = G.shape[0]
            y_new[:m1] = torch.clamp(y_new[:m1], min=0.0)

        # Compute step size metrics
        with profiler("step size computation"):
            dx = x_new - x
            dy = y_new - y

            num = w * (dx @ dx) + (dy @ dy) / w

            # This involves another K @ dx
            with profiler("K @ dx (in backtrack)"):
                K_dx = K @ dx

            denom = 2 * torch.abs(dy @ K_dx)

            if denom > 1e-12:
                eta_new = num / denom
            else:
                eta_new = eta

        # Update
        with profiler("variable updates"):
            x = x_new
            y = y_new
            eta = eta_new

    if device == 'cuda':
        torch.cuda.synchronize()
    total_time = time.perf_counter() - total_start

    print(f"Total time for {n_iters} iterations: {total_time:.3f}s")
    print(f"Average time per iteration: {total_time/n_iters*1000:.2f} ms")
    print()

    profiler.report()

    # Calculate what percentage SpMV is
    spmv_ops = ['K.T @ y', 'K @ x', 'K @ dx (in backtrack)']
    spmv_time = sum(sum(profiler.timings[op]) for op in spmv_ops)
    spmv_pct = 100 * spmv_time / total_time

    print(f"SpMV operations (K@x, K.T@y, K@dx): {spmv_pct:.1f}% of iteration time")
    print(f"Other operations: {100-spmv_pct:.1f}% of iteration time")
    print()

    if spmv_pct < 50:
        print("⚠️  WARNING: SpMV is NOT the bottleneck!")
        print("   Other operations (projections, vector ops) dominate.")
        print("   GPU may be underutilized due to small vector ops.")
    else:
        print("✓ SpMV operations dominate (expected for large sparse problems)")


if __name__ == "__main__":
    if torch.cuda.is_available():
        print(f"CUDA available: {torch.cuda.get_device_name(0)}")
        device = 'cuda'
    else:
        print("No CUDA available, using CPU")
        device = 'cpu'

    profile_solver_iteration(device)

    print("\n" + "="*70)
    print("INTERPRETATION GUIDE")
    print("="*70)
    print("""
If SpMV (K@x, K.T@y) is < 50% of time:
  → GPU is spending most time on small vector operations
  → These don't parallelize well on GPU
  → CPU might actually be faster for small problems!

If SpMV is > 70% of time:
  → GPU is being used effectively
  → Sparse matrix operations are the bottleneck (expected)
  → GPU speedup should be good

Check the "step size computation" time:
  → If this is large, the adaptive step size is expensive
  → Could try fixed step size for benchmarking
    """)
