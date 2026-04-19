import numpy as np
import scipy.sparse as sp
import scipy.sparse.linalg as spla
import matplotlib.pyplot as plt


def main():
    # Discretization parameters
    h = 1 / 20
    n = int(1 / h) - 1                
    x = np.arange(1, n + 1) * h
    tol = h**2
    maxit = 100000

    main_diag = 2.0 * np.ones(n)
    off_diag = -1.0 * np.ones(n - 1)
    A = (1 / h**2) * sp.diags(
        [off_diag, main_diag, off_diag],
        offsets=[-1, 0, 1],
        format="csr"
    )

    # Right-hand side
    f = -4 * np.pi**2 * np.cos(np.pi * x) * np.sin(np.pi * x)

    # Direct solve
    xtrue = spla.spsolve(A, f)

    # Jacobi 
    u = np.zeros(n)
    Dinv = (h**2 / 2.0) * np.ones(n)   

    residual_history = []
    error_history = []

    f_norm = np.linalg.norm(f)
    xtrue_norm = np.linalg.norm(xtrue)

    # Jacobi iteration
    for k in range(maxit):
        r = f - A @ u
        rel_res = np.linalg.norm(r) / f_norm
        rel_err = np.linalg.norm(u - xtrue) / xtrue_norm

        residual_history.append(rel_res)
        error_history.append(rel_err)

        if rel_res < tol:
            break

        u = u + Dinv * r

    print(f"Jacobi stopped after {k + 1} iterations")
    print(f"Final relative residual: {residual_history[-1]:.6e}")
    print(f"Final relative error:    {error_history[-1]:.6e}")

    # relative residual
    plt.figure(figsize=(6, 4))
    plt.semilogy(residual_history, linewidth=2)
    plt.xlabel("Iteration")
    plt.ylabel("Relative residual")
    plt.title("Jacobi iteration: relative residual")
    plt.grid(True, which="both", linestyle="--", alpha=0.6)
    plt.tight_layout()
    plt.savefig("jacobi_residual.png", dpi=300, bbox_inches="tight")
    plt.close()

    # relative error
    plt.figure(figsize=(6, 4))
    plt.semilogy(error_history, linewidth=2)
    plt.xlabel("Iteration")
    plt.ylabel("Relative error")
    plt.title("Jacobi iteration: relative error")
    plt.grid(True, which="both", linestyle="--", alpha=0.6)
    plt.tight_layout()
    plt.savefig("jacobi_error.png", dpi=300, bbox_inches="tight")
    plt.close()

    # final computed solution against direct solution
    plt.figure(figsize=(6, 4))
    plt.plot(x, xtrue, linewidth=2, label="Direct solve")
    plt.plot(x, u, "--", linewidth=2, label="Jacobi")
    plt.xlabel("x")
    plt.ylabel("u")
    plt.title("Computed solution")
    plt.legend()
    plt.grid(True, linestyle="--", alpha=0.6)
    plt.tight_layout()
    plt.savefig("jacobi_solution.png", dpi=300, bbox_inches="tight")
    plt.close()


if __name__ == "__main__":
    main()