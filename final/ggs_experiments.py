import time
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

def make_consistent_problem(m, n, seed=0):
    rng = np.random.default_rng(seed)

    A = rng.standard_normal((m, n))
    x_star = rng.standard_normal(n)
    b = A @ x_star

    return A, b, x_star
def make_inconsistent_problem(m, n, seed=0):
    """
    Creates an inconsistent least-squares problem.

    We choose x_star first, then set

        b = A x_star + b0,

    where b0 is approximately orthogonal to the column space of A.

    This means x_star is still the least-squares solution, but b is generally
    not exactly in the column space of A.
    """
    rng = np.random.default_rng(seed)

    A = rng.standard_normal((m, n))
    x_star = rng.standard_normal(n)

    z = rng.standard_normal(m)

    # Project z onto the column space of A, then subtract.
    # The result b0 is approximately orthogonal to col(A).
    projection_coeffs = np.linalg.lstsq(A, z, rcond=None)[0]
    b0 = z - A @ projection_coeffs

    b = A @ x_star + b0

    return A, b, x_star

def make_sparse_consistent_problem(m, n, density, seed=0):
    """
    Creates a synthetic sparse-looking least-squares problem.

    The matrix is stored as a dense NumPy array, but most entries are zero.
    This lets us test how the algorithms behave as the sparsity pattern changes
    without rewriting the solvers for scipy sparse matrices.

    density = fraction of nonzero entries, e.g. 0.01 means 1 percent nonzero.
    """
    rng = np.random.default_rng(seed)

    while True:
        mask = rng.random((m, n)) < density
        A = rng.standard_normal((m, n)) * mask

        # Make sure no column is entirely zero.
        for j in range(n):
            if np.all(A[:, j] == 0):
                i = rng.integers(0, m)
                A[i, j] = rng.standard_normal()

        # Make sure A has full column rank.
        if np.linalg.matrix_rank(A) == n:
            break

    x_star = rng.standard_normal(n)
    b = A @ x_star

    return A, b, x_star


def relative_solution_error(x, x_star):
    return np.linalg.norm(x - x_star) ** 2 / np.linalg.norm(x_star) ** 2


def ggs(A, b, x_star, tol=1e-6, max_iter=200_000):
    """
    Greedy Gauss-Seidel for

        min_x ||b - A x||_2^2.

    Let r_k = b - A x_k. The normal-equation residual is A^T r_k.
    At each iteration, GGS chooses the coordinate with largest absolute
    normal-equation residual:

        j = argmax_i |A_i^T r_k|.

    Then it performs the exact coordinate update

        x_{k+1} = x_k + (A_j^T r_k / ||A_j||_2^2) e_j.

    This is Algorithm 2 from the paper, with the usual case that the
    maximizing index is unique.
    """
    m, n = A.shape

    x = np.zeros(n)
    r = b.copy()

    col_norm_sq = np.sum(A * A, axis=0)

    errors = []
    times = []

    start = time.perf_counter()

    for k in range(max_iter + 1):
        err = relative_solution_error(x, x_star)
        errors.append(err)
        times.append(time.perf_counter() - start)

        if err <= tol:
            break

        normal_residual = A.T @ r

        j = np.argmax(np.abs(normal_residual))

        step = normal_residual[j] / col_norm_sq[j]

        x[j] += step
        r -= step * A[:, j]

    return {
        "x": x,
        "iterations": k,
        "time": time.perf_counter() - start,
        "errors": np.array(errors),
        "times": np.array(times),
    }
def grcd(A, b, x_star, tol=1e-6, max_iter=200_000, seed=0):
    """
    Greedy randomized coordinate descent.

    This follows Algorithm 1 in the paper. It first builds a set of promising
    coordinates V_k, then randomly selects one coordinate from V_k with
    probability proportional to |A_j^T r_k|^2.

    This is the main comparison method for GGS.
    """
    rng = np.random.default_rng(seed)

    m, n = A.shape

    x = np.zeros(n)
    r = b.copy()

    col_norm_sq = np.sum(A * A, axis=0)
    frob_norm_sq = np.sum(col_norm_sq)

    errors = []
    times = []

    start = time.perf_counter()

    for k in range(max_iter + 1):
        err = relative_solution_error(x, x_star)
        errors.append(err)
        times.append(time.perf_counter() - start)

        if err <= tol:
            break

        normal_residual = A.T @ r
        normal_residual_sq = normal_residual ** 2
        normal_residual_norm_sq = np.sum(normal_residual_sq)

        scores = normal_residual_sq / col_norm_sq

        eps_k = 0.5 * (
            np.max(scores) / normal_residual_norm_sq
            + 1.0 / frob_norm_sq
        )

        threshold = eps_k * normal_residual_norm_sq * col_norm_sq

        V = np.where(normal_residual_sq >= threshold)[0]

        weights = normal_residual_sq[V]
        probs = weights / np.sum(weights)

        j = rng.choice(V, p=probs)

        step = normal_residual[j] / col_norm_sq[j]

        x[j] += step
        r -= step * A[:, j]

    return {
        "x": x,
        "iterations": k,
        "time": time.perf_counter() - start,
        "errors": np.array(errors),
        "times": np.array(times),
    }

def rgs(A, b, x_star, tol=1e-6, max_iter=200_000, seed=0):
    """
    Randomized Gauss-Seidel / randomized coordinate descent.

    This is the basic randomized baseline discussed before GRCD and GGS.
    At each step, we choose column j with probability proportional to
    ||A_j||_2^2, then do the usual exact coordinate update.
    """
    rng = np.random.default_rng(seed)

    m, n = A.shape

    x = np.zeros(n)
    r = b.copy()

    col_norm_sq = np.sum(A * A, axis=0)
    probs = col_norm_sq / np.sum(col_norm_sq)

    errors = []
    times = []

    start = time.perf_counter()

    for k in range(max_iter + 1):
        err = relative_solution_error(x, x_star)
        errors.append(err)
        times.append(time.perf_counter() - start)

        if err <= tol:
            break

        j = rng.choice(n, p=probs)

        normal_residual_j = A[:, j] @ r
        step = normal_residual_j / col_norm_sq[j]

        x[j] += step
        r -= step * A[:, j]

    return {
        "x": x,
        "iterations": k,
        "time": time.perf_counter() - start,
        "errors": np.array(errors),
        "times": np.array(times),
    }


def plot_convergence(ggs_result, grcd_result, filename_prefix="convergence"):
    plt.figure(figsize=(7, 5))

    plt.semilogy(
        ggs_result["errors"],
        linewidth=2.5,
        label="GGS"
    )
    plt.semilogy(
        grcd_result["errors"],
        linewidth=2.5,
        linestyle="--",
        label="GRCD"
    )

    plt.xlabel("Iteration")
    plt.ylabel("Relative solution error")
    plt.title("Convergence by iteration")
    plt.legend()
    plt.grid(True, which="both", alpha=0.3)
    plt.tight_layout()
    plt.savefig(f"{filename_prefix}_iterations.png", dpi=300)
    plt.show()

    plt.figure(figsize=(7, 5))

    plt.semilogy(
        ggs_result["times"],
        ggs_result["errors"],
        linewidth=2.5,
        label="GGS"
    )
    plt.semilogy(
        grcd_result["times"],
        grcd_result["errors"],
        linewidth=2.5,
        linestyle="--",
        label="GRCD"
    )

    plt.xlabel("Time (seconds)")
    plt.ylabel("Relative solution error")
    plt.title("Convergence by runtime")
    plt.legend()
    plt.grid(True, which="both", alpha=0.3)
    plt.tight_layout()
    plt.savefig(f"{filename_prefix}_time.png", dpi=300)
    plt.show()


def plot_three_method_convergence(rgs_result, grcd_result, ggs_result,
                                  filename_prefix="three_method_convergence",
                                  title_prefix=""):
    plt.figure(figsize=(7, 5))

    plt.semilogy(
        rgs_result["errors"],
        linewidth=2.5,
        linestyle=":",
        label="RGS"
    )
    plt.semilogy(
        grcd_result["errors"],
        linewidth=2.5,
        linestyle="--",
        label="GRCD"
    )
    plt.semilogy(
        ggs_result["errors"],
        linewidth=2.5,
        label="GGS"
    )

    plt.xlabel("Iteration")
    plt.ylabel("Relative solution error")
    plt.title(f"{title_prefix}Convergence by iteration")
    plt.legend()
    plt.grid(True, which="both", alpha=0.3)
    plt.tight_layout()
    plt.savefig(f"{filename_prefix}_iterations.png", dpi=300)
    plt.show()

    plt.figure(figsize=(7, 5))

    plt.semilogy(
        rgs_result["times"],
        rgs_result["errors"],
        linewidth=2.5,
        linestyle=":",
        label="RGS"
    )
    plt.semilogy(
        grcd_result["times"],
        grcd_result["errors"],
        linewidth=2.5,
        linestyle="--",
        label="GRCD"
    )
    plt.semilogy(
        ggs_result["times"],
        ggs_result["errors"],
        linewidth=2.5,
        label="GGS"
    )

    plt.xlabel("Time (seconds)")
    plt.ylabel("Relative solution error")
    plt.title(f"{title_prefix}Convergence by runtime")
    plt.legend()
    plt.grid(True, which="both", alpha=0.3)
    plt.tight_layout()
    plt.savefig(f"{filename_prefix}_time.png", dpi=300)
    plt.show()

def run_dense_experiments(sizes, trials=5, tol=1e-6, problem_type="consistent"):
    """
    Runs GGS and GRCD on several dense least-squares problems.

    problem_type can be:
        "consistent"
        "inconsistent"

    Each row averages over a few random trials.
    """
    rows = []

    for m, n in sizes:
        print(f"Running {problem_type} size {m} x {n}...")

        ggs_iterations = []
        grcd_iterations = []
        ggs_times = []
        grcd_times = []

        for trial in range(trials):
            seed = 1000 + trial

            if problem_type == "consistent":
                A, b, x_star = make_consistent_problem(m, n, seed=seed)
            elif problem_type == "inconsistent":
                A, b, x_star = make_inconsistent_problem(m, n, seed=seed)
            else:
                raise ValueError("problem_type must be 'consistent' or 'inconsistent'")

            ggs_result = ggs(A, b, x_star, tol=tol)
            grcd_result = grcd(A, b, x_star, tol=tol, seed=seed)

            ggs_iterations.append(ggs_result["iterations"])
            grcd_iterations.append(grcd_result["iterations"])
            ggs_times.append(ggs_result["time"])
            grcd_times.append(grcd_result["time"])

        ggs_it = np.mean(ggs_iterations)
        grcd_it = np.mean(grcd_iterations)
        ggs_cpu = np.mean(ggs_times)
        grcd_cpu = np.mean(grcd_times)

        row = {
            "size": f"{m} x {n}",
            "GGS IT": ggs_it,
            "GRCD IT": grcd_it,
            "IT speed-up": grcd_it / ggs_it,
            "GGS CPU": ggs_cpu,
            "GRCD CPU": grcd_cpu,
            "CPU speed-up": grcd_cpu / ggs_cpu,
        }

        rows.append(row)

    return rows

def run_sparse_density_experiment(densities, m=2000, n=100, trials=10, tol=1e-6):
    """
    Compares RGS, GRCD, and GGS as the density of A changes.
    """
    rows = []

    for density in densities:
        print(f"Running density {density:.2%}...")

        rgs_iterations = []
        grcd_iterations = []
        ggs_iterations = []

        rgs_times = []
        grcd_times = []
        ggs_times = []

        for trial in range(trials):
            seed = 3000 + trial

            A, b, x_star = make_sparse_consistent_problem(
                m=m,
                n=n,
                density=density,
                seed=seed
            )

            rgs_result = rgs(A, b, x_star, tol=tol, seed=seed)
            grcd_result = grcd(A, b, x_star, tol=tol, seed=seed)
            ggs_result = ggs(A, b, x_star, tol=tol)

            rgs_iterations.append(rgs_result["iterations"])
            grcd_iterations.append(grcd_result["iterations"])
            ggs_iterations.append(ggs_result["iterations"])

            rgs_times.append(rgs_result["time"])
            grcd_times.append(grcd_result["time"])
            ggs_times.append(ggs_result["time"])

        row = {
            "density": density,
            "RGS IT": np.mean(rgs_iterations),
            "GRCD IT": np.mean(grcd_iterations),
            "GGS IT": np.mean(ggs_iterations),
            "RGS CPU": np.mean(rgs_times),
            "GRCD CPU": np.mean(grcd_times),
            "GGS CPU": np.mean(ggs_times),
            "GRCD/GGS speed-up": np.mean(grcd_times) / np.mean(ggs_times),
            "RGS/GGS speed-up": np.mean(rgs_times) / np.mean(ggs_times),
        }

        rows.append(row)

    return rows


def print_experiment_table(rows):
    """
    Prints a clean table in the terminal.
    """
    print()
    print(
        f"{'size':>12} | "
        f"{'GGS IT':>10} | "
        f"{'GRCD IT':>10} | "
        f"{'IT spd':>8} | "
        f"{'GGS CPU':>10} | "
        f"{'GRCD CPU':>10} | "
        f"{'CPU spd':>8}"
    )
    print("-" * 88)

    for row in rows:
        print(
            f"{row['size']:>12} | "
            f"{row['GGS IT']:10.2f} | "
            f"{row['GRCD IT']:10.2f} | "
            f"{row['IT speed-up']:8.3f} | "
            f"{row['GGS CPU']:10.4f} | "
            f"{row['GRCD CPU']:10.4f} | "
            f"{row['CPU speed-up']:8.3f}"
        )
def print_sparse_density_table(rows):
    print()
    print(
        f"{'density':>10} | "
        f"{'RGS IT':>10} | {'GRCD IT':>10} | {'GGS IT':>10} | "
        f"{'RGS CPU':>10} | {'GRCD CPU':>10} | {'GGS CPU':>10} | "
        f"{'GRCD/GGS':>10}"
    )
    print("-" * 105)

    for row in rows:
        print(
            f"{row['density']:10.2%} | "
            f"{row['RGS IT']:10.2f} | "
            f"{row['GRCD IT']:10.2f} | "
            f"{row['GGS IT']:10.2f} | "
            f"{row['RGS CPU']:10.4f} | "
            f"{row['GRCD CPU']:10.4f} | "
            f"{row['GGS CPU']:10.4f} | "
            f"{row['GRCD/GGS speed-up']:10.3f}"
        )

def save_experiment_table(rows, filename="dense_consistent_table.csv"):
    """
    Saves the table as a CSV file.

    This avoids needing pandas.
    """
    with open(filename, "w") as f:
        f.write("size,GGS IT,GRCD IT,IT speed-up,GGS CPU,GRCD CPU,CPU speed-up\n")

        for row in rows:
            f.write(
                f"{row['size']},"
                f"{row['GGS IT']:.2f},"
                f"{row['GRCD IT']:.2f},"
                f"{row['IT speed-up']:.3f},"
                f"{row['GGS CPU']:.4f},"
                f"{row['GRCD CPU']:.4f},"
                f"{row['CPU speed-up']:.3f}\n"
            )

    print(f"\nSaved table to {filename}")

def save_sparse_density_table(rows, filename="sparse_density_table.csv"):
    with open(filename, "w") as f:
        f.write(
            "density,RGS IT,GRCD IT,GGS IT,"
            "RGS CPU,GRCD CPU,GGS CPU,"
            "GRCD/GGS speed-up,RGS/GGS speed-up\n"
        )

        for row in rows:
            f.write(
                f"{row['density']},"
                f"{row['RGS IT']:.2f},"
                f"{row['GRCD IT']:.2f},"
                f"{row['GGS IT']:.2f},"
                f"{row['RGS CPU']:.4f},"
                f"{row['GRCD CPU']:.4f},"
                f"{row['GGS CPU']:.4f},"
                f"{row['GRCD/GGS speed-up']:.3f},"
                f"{row['RGS/GGS speed-up']:.3f}\n"
            )

    print(f"\nSaved table to {filename}")


def plot_cpu_speedups(rows, filename="dense_cpu_speedups.png", title=None):
    """
    Makes a slide-friendly bar chart.

    Values above 1 mean GGS was faster than GRCD.
    """
    sizes = [row["size"] for row in rows]
    speedups = [row["CPU speed-up"] for row in rows]

    if title is None:
        title = "GGS runtime speed-up over GRCD"

    plt.figure(figsize=(9, 5))
    plt.bar(sizes, speedups)

    plt.axhline(1.0, linestyle="--", linewidth=1.5)

    plt.ylabel("CPU speed-up: GRCD time / GGS time")
    plt.xlabel("Matrix size")
    plt.title(title)
    plt.xticks(rotation=35, ha="right")
    plt.tight_layout()

    plt.savefig(filename, dpi=300)
    plt.show()

    print(f"Saved plot to {filename}")

def plot_sparse_density_results(rows, filename_prefix="sparse_density"):
    densities = [100 * row["density"] for row in rows]

    rgs_iterations = [row["RGS IT"] for row in rows]
    grcd_iterations = [row["GRCD IT"] for row in rows]
    ggs_iterations = [row["GGS IT"] for row in rows]

    rgs_times = [row["RGS CPU"] for row in rows]
    grcd_times = [row["GRCD CPU"] for row in rows]
    ggs_times = [row["GGS CPU"] for row in rows]

    speedups = [row["GRCD/GGS speed-up"] for row in rows]

    plt.figure(figsize=(7, 5))
    plt.plot(densities, rgs_iterations, marker="o", linewidth=2.5, linestyle=":", label="RGS")
    plt.plot(densities, grcd_iterations, marker="o", linewidth=2.5, linestyle="--", label="GRCD")
    plt.plot(densities, ggs_iterations, marker="o", linewidth=2.5, label="GGS")
    plt.xlabel("Density (% nonzero)")
    plt.ylabel("Average iterations")
    plt.title("Synthetic sparse systems: iteration count vs density")
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(f"{filename_prefix}_iterations.png", dpi=300)
    plt.show()

    plt.figure(figsize=(7, 5))
    plt.plot(densities, rgs_times, marker="o", linewidth=2.5, linestyle=":", label="RGS")
    plt.plot(densities, grcd_times, marker="o", linewidth=2.5, linestyle="--", label="GRCD")
    plt.plot(densities, ggs_times, marker="o", linewidth=2.5, label="GGS")
    plt.xlabel("Density (% nonzero)")
    plt.ylabel("Average runtime (seconds)")
    plt.title("Synthetic sparse systems: runtime vs density")
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(f"{filename_prefix}_runtime.png", dpi=300)
    plt.show()

    plt.figure(figsize=(7, 5))
    plt.plot(densities, speedups, marker="o", linewidth=2.5)
    plt.axhline(1.0, linestyle="--", linewidth=1.5)
    plt.xlabel("Density (% nonzero)")
    plt.ylabel("CPU speed-up: GRCD time / GGS time")
    plt.title("Synthetic sparse systems: GGS speed-up over GRCD")
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(f"{filename_prefix}_speedup.png", dpi=300)
    plt.show()



# ------------------------------------------------------MAIN--------------------------------------------------
if __name__ == "__main__":
    densities = [
        0.01,
        0.02,
        0.05,
        0.10,
        0.25,
        0.50,
        1.00,
    ]

    rows = run_sparse_density_experiment(
        densities,
        m=2000,
        n=100,
        trials=50,
        tol=1e-6
    )

    print_sparse_density_table(rows)

    save_sparse_density_table(
        rows,
        filename="sparse_density_table.csv"
    )

    plot_sparse_density_results(
        rows,
        filename_prefix="sparse_density"
    )