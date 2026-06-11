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
    rng = np.random.default_rng(seed)

    A = rng.standard_normal((m, n))
    x_star = rng.standard_normal(n)

    z = rng.standard_normal(m)

    projection_coeffs = np.linalg.lstsq(A, z, rcond=None)[0]
    b0 = z - A @ projection_coeffs

    b = A @ x_star + b0

    return A, b, x_star

def make_sparse_consistent_problem(m, n, density, seed=0):
    rng = np.random.default_rng(seed)

    while True:
        mask = rng.random((m, n)) < density
        A = rng.standard_normal((m, n)) * mask

        #check for zero col
        for j in range(n):
            if np.all(A[:, j] == 0):
                i = rng.integers(0, m)
                A[i, j] = rng.standard_normal()

        if np.linalg.matrix_rank(A) == n:
            break

    x_star = rng.standard_normal(n)
    b = A @ x_star

    return A, b, x_star


def relative_solution_error(x, x_star):
    return np.linalg.norm(x - x_star) ** 2 / np.linalg.norm(x_star) ** 2


def ggs(A, b, x_star, tol=1e-6, max_iter=200_000):
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

def run_large_matrix_experiment(sizes, trials=5, tol=1e-6):

    rows = []

    for m, n in sizes:
        print(f"Running large dense size {m} x {n}...")

        ggs_iterations = []
        grcd_iterations = []
        ggs_times = []
        grcd_times = []

        for trial in range(trials):
            seed = 5000 + trial

            A, b, x_star = make_consistent_problem(m, n, seed=seed)

            ggs_result = ggs(A, b, x_star, tol=tol)
            grcd_result = grcd(A, b, x_star, tol=tol, seed=seed)

            ggs_iterations.append(ggs_result["iterations"])
            grcd_iterations.append(grcd_result["iterations"])
            ggs_times.append(ggs_result["time"])
            grcd_times.append(grcd_result["time"])

            print(
                f"  trial {trial + 1}/{trials}: "
                f"GGS {ggs_result['time']:.3f}s, "
                f"GRCD {grcd_result['time']:.3f}s"
            )

        ggs_it = np.mean(ggs_iterations)
        grcd_it = np.mean(grcd_iterations)
        ggs_cpu = np.mean(ggs_times)
        grcd_cpu = np.mean(grcd_times)

        row = {
            "size": f"{m} x {n}",
            "m": m,
            "n": n,
            "GGS IT": ggs_it,
            "GRCD IT": grcd_it,
            "IT speed-up": grcd_it / ggs_it,
            "GGS CPU": ggs_cpu,
            "GRCD CPU": grcd_cpu,
            "CPU speed-up": grcd_cpu / ggs_cpu,
        }

        rows.append(row)

    return rows

def print_large_matrix_table(rows):
    print()
    print(
        f"{'size':>16} | "
        f"{'GGS IT':>10} | "
        f"{'GRCD IT':>10} | "
        f"{'IT spd':>8} | "
        f"{'GGS CPU':>10} | "
        f"{'GRCD CPU':>10} | "
        f"{'CPU spd':>8}"
    )
    print("-" * 95)

    for row in rows:
        print(
            f"{row['size']:>16} | "
            f"{row['GGS IT']:10.2f} | "
            f"{row['GRCD IT']:10.2f} | "
            f"{row['IT speed-up']:8.3f} | "
            f"{row['GGS CPU']:10.4f} | "
            f"{row['GRCD CPU']:10.4f} | "
            f"{row['CPU speed-up']:8.3f}"
        )

def save_large_matrix_table(rows, filename="large_matrix_table.csv"):
    with open(filename, "w") as f:
        f.write("size,m,n,GGS IT,GRCD IT,IT speed-up,GGS CPU,GRCD CPU,CPU speed-up\n")

        for row in rows:
            f.write(
                f"{row['size']},"
                f"{row['m']},"
                f"{row['n']},"
                f"{row['GGS IT']:.2f},"
                f"{row['GRCD IT']:.2f},"
                f"{row['IT speed-up']:.3f},"
                f"{row['GGS CPU']:.4f},"
                f"{row['GRCD CPU']:.4f},"
                f"{row['CPU speed-up']:.3f}\n"
            )

    print(f"\nSaved table to {filename}")

def plot_large_matrix_results(rows, filename_prefix="large_matrix"):
    sizes = [row["size"] for row in rows]
    m_values = [row["m"] for row in rows]

    ggs_iterations = [row["GGS IT"] for row in rows]
    grcd_iterations = [row["GRCD IT"] for row in rows]

    ggs_times = [row["GGS CPU"] for row in rows]
    grcd_times = [row["GRCD CPU"] for row in rows]

    speedups = [row["CPU speed-up"] for row in rows]

    plt.figure(figsize=(8, 5))
    plt.bar(sizes, speedups)
    plt.axhline(1.0, linestyle="--", linewidth=1.5)
    plt.ylabel("CPU speed-up: GRCD time / GGS time")
    plt.xlabel("Matrix size")
    plt.title("Large tall matrices: GGS speed-up over GRCD")
    plt.xticks(rotation=25, ha="right")
    plt.tight_layout()
    plt.savefig(f"{filename_prefix}_speedup.png", dpi=300)
    plt.show()

    plt.figure(figsize=(7, 5))
    plt.plot(m_values, ggs_times, marker="o", linewidth=2.5, label="GGS")
    plt.plot(m_values, grcd_times, marker="o", linewidth=2.5, linestyle="--", label="GRCD")
    plt.xlabel("Number of rows m")
    plt.ylabel("Average runtime (seconds)")
    plt.title("Large tall matrices: runtime scaling")
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(f"{filename_prefix}_runtime.png", dpi=300)
    plt.show()

    plt.figure(figsize=(7, 5))
    plt.plot(m_values, ggs_iterations, marker="o", linewidth=2.5, label="GGS")
    plt.plot(m_values, grcd_iterations, marker="o", linewidth=2.5, linestyle="--", label="GRCD")
    plt.xlabel("Number of rows m")
    plt.ylabel("Average iterations")
    plt.title("Large tall matrices: iteration counts")
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(f"{filename_prefix}_iterations.png", dpi=300)
    plt.show()



# ------------------------------------------------------MAIN--------------------------------------------------
def plot_large_matrix_diagnostics(rows, filename_prefix="large_matrix_diagnostic"):
    m_values = [row["m"] for row in rows]

    ggs_iterations = [row["GGS IT"] for row in rows]
    grcd_iterations = [row["GRCD IT"] for row in rows]

    ggs_times = [row["GGS CPU"] for row in rows]
    grcd_times = [row["GRCD CPU"] for row in rows]

    speedups = [row["CPU speed-up"] for row in rows]
    iteration_ratios = [row["IT speed-up"] for row in rows]

    time_gaps = [
        row["GRCD CPU"] - row["GGS CPU"]
        for row in rows
    ]

    #speed-up ratio near 1
    plt.figure(figsize=(7, 5))
    plt.semilogx(m_values, speedups, marker="o", linewidth=2.5)
    plt.axhline(1.0, linestyle="--", linewidth=1.5)

    plt.xlabel("Number of rows m")
    plt.ylabel("CPU speed-up: GRCD time / GGS time")
    plt.title("Large tall matrices: speed-up approaches 1")
    plt.grid(True, which="both", alpha=0.3)
    plt.tight_layout()
    plt.savefig(f"{filename_prefix}_speedup_ratio.png", dpi=300)
    plt.show()

    #absolute runtime gap
    plt.figure(figsize=(7, 5))
    plt.semilogx(m_values, time_gaps, marker="o", linewidth=2.5)
    plt.axhline(0.0, linestyle="--", linewidth=1.5)

    plt.xlabel("Number of rows m")
    plt.ylabel("Runtime gap: GRCD time - GGS time (seconds)")
    plt.title("Large tall matrices: runtime gap")
    plt.grid(True, which="both", alpha=0.3)
    plt.tight_layout()
    plt.savefig(f"{filename_prefix}_runtime_gap.png", dpi=300)
    plt.show()

    #iteration ratio near 1
    plt.figure(figsize=(7, 5))
    plt.semilogx(m_values, iteration_ratios, marker="o", linewidth=2.5)
    plt.axhline(1.0, linestyle="--", linewidth=1.5)

    plt.xlabel("Number of rows m")
    plt.ylabel("Iteration ratio: GRCD IT / GGS IT")
    plt.title("Large tall matrices: iteration counts become nearly identical")
    plt.grid(True, which="both", alpha=0.3)
    plt.tight_layout()
    plt.savefig(f"{filename_prefix}_iteration_ratio.png", dpi=300)
    plt.show()

    #runtime scaling of both methods
    plt.figure(figsize=(7, 5))
    plt.loglog(m_values, ggs_times, marker="o", linewidth=2.5, label="GGS")
    plt.loglog(m_values, grcd_times, marker="o", linewidth=2.5, linestyle="--", label="GRCD")

    plt.xlabel("Number of rows m")
    plt.ylabel("Average runtime (seconds)")
    plt.title("Large tall matrices: both methods scale similarly")
    plt.legend()
    plt.grid(True, which="both", alpha=0.3)
    plt.tight_layout()
    plt.savefig(f"{filename_prefix}_runtime_loglog.png", dpi=300)
    plt.show()

if __name__ == "__main__":
    sizes = [
        (10_000, 100),
        (25_000, 100),
        (50_000, 100),
        (100_000, 100),
        (250_000, 100),
        (500_000, 100),
    ]

    rows = run_large_matrix_experiment(
        sizes,
        trials=10,
        tol=1e-6
    )

    print_large_matrix_table(rows)

    save_large_matrix_table(
        rows,
        filename="large_matrix_table.csv"
    )

    plot_large_matrix_results(
        rows,
        filename_prefix="large_matrix"
    )

    plot_large_matrix_diagnostics(
        rows,
        filename_prefix="large_matrix_diagnostic"
    )