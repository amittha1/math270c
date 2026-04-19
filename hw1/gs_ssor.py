import numpy as np
import scipy.sparse as sp
import scipy.sparse.linalg as spla
import matplotlib.pyplot as plt


def build_problem(h):
    n = int(1 / h) - 1
    x = np.arange(1, n + 1) * h

    main_diag = 2.0 * np.ones(n)
    off_diag = -1.0 * np.ones(n - 1)
    A = (1 / h**2) * sp.diags(
        [off_diag, main_diag, off_diag],
        offsets=[-1, 0, 1],
        format="csr"
    )

    f = -4 * np.pi**2 * np.cos(np.pi * x) * np.sin(np.pi * x)
    xtrue = spla.spsolve(A, f)

    return A, f, xtrue, x


def jacobi(A, f, xtrue, h, tol, maxit=100000):
    n = len(f)
    u = np.zeros(n)
    Dinv = (h**2 / 2.0) * np.ones(n)

    residual_history = []
    error_history = []

    f_norm = np.linalg.norm(f)
    xtrue_norm = np.linalg.norm(xtrue)

    for k in range(maxit):
        r = f - A @ u
        rel_res = np.linalg.norm(r) / f_norm
        rel_err = np.linalg.norm(u - xtrue) / xtrue_norm

        residual_history.append(rel_res)
        error_history.append(rel_err)

        if rel_res < tol:
            break

        u = u + Dinv * r

    return u, residual_history, error_history, k + 1


def gauss_seidel(A, f, xtrue, h, tol, maxit=100000):
    n = len(f)
    u = np.zeros(n)

    residual_history = []
    error_history = []

    f_norm = np.linalg.norm(f)
    xtrue_norm = np.linalg.norm(xtrue)

    for k in range(maxit):
        r = f - A @ u
        rel_res = np.linalg.norm(r) / f_norm
        rel_err = np.linalg.norm(u - xtrue) / xtrue_norm

        residual_history.append(rel_res)
        error_history.append(rel_err)

        if rel_res < tol:
            break

        u_old = u.copy()

        for i in range(n):
            left = u[i - 1] if i > 0 else 0.0
            right = u_old[i + 1] if i < n - 1 else 0.0
            u[i] = 0.5 * (h**2 * f[i] + left + right)

    return u, residual_history, error_history, k + 1


def ssor(A, f, xtrue, h, tol, omega, maxit=100000):
    n = len(f)
    u = np.zeros(n)

    residual_history = []
    error_history = []

    f_norm = np.linalg.norm(f)
    xtrue_norm = np.linalg.norm(xtrue)

    for k in range(maxit):
        r = f - A @ u
        rel_res = np.linalg.norm(r) / f_norm
        rel_err = np.linalg.norm(u - xtrue) / xtrue_norm

        residual_history.append(rel_res)
        error_history.append(rel_err)

        if rel_res < tol:
            break

        # Forward SOR sweep
        y = u.copy()
        u_old = u.copy()
        for i in range(n):
            left = y[i - 1] if i > 0 else 0.0
            right = u_old[i + 1] if i < n - 1 else 0.0
            gs_value = 0.5 * (h**2 * f[i] + left + right)
            y[i] = (1 - omega) * u_old[i] + omega * gs_value

        # Backward SOR sweep
        u_new = y.copy()
        y_old = y.copy()
        for i in range(n - 1, -1, -1):
            left = y[i - 1] if i > 0 else 0.0
            right = u_new[i + 1] if i < n - 1 else 0.0
            gs_value = 0.5 * (h**2 * f[i] + left + right)
            u_new[i] = (1 - omega) * y_old[i] + omega * gs_value

        u = u_new

    return u, residual_history, error_history, k + 1


def main():
    h = 1 / 20
    tol = h**2
    A, f, xtrue, x = build_problem(h)

    # Jacobi
    u_jac, res_jac, err_jac, it_jac = jacobi(A, f, xtrue, h, tol)

    # Gauss-Seidel
    u_gs, res_gs, err_gs, it_gs = gauss_seidel(A, f, xtrue, h, tol)

    # SSOR grid search over omega in (0, 2)
    omega_values = np.linspace(0.1, 1.95, 38)

    best_omega = None
    best_iters = None
    best_result = None

    for omega in omega_values:
        u_ssor, res_ssor, err_ssor, it_ssor = ssor(A, f, xtrue, h, tol, omega)

        if best_iters is None or it_ssor < best_iters:
            best_iters = it_ssor
            best_omega = omega
            best_result = (u_ssor, res_ssor, err_ssor, it_ssor)

    u_ssor, res_ssor, err_ssor, it_ssor = best_result

    print(f"Jacobi iterations:       {it_jac}")
    print(f"Gauss-Seidel iterations: {it_gs}")
    print(f"Best SSOR omega:         {best_omega:.3f}")
    print(f"Best SSOR iterations:    {it_ssor}")

    # Comparison plot: relative residual
    plt.figure(figsize=(7, 5))
    plt.semilogy(res_jac, linewidth=2, label="Jacobi")
    plt.semilogy(res_gs, linewidth=2, label="Gauss-Seidel")
    plt.semilogy(res_ssor, linewidth=2, label=f"SSOR ($\\omega={best_omega:.2f}$)")
    plt.xlabel("Iteration")
    plt.ylabel("Relative residual")
    plt.title("Relative residual comparison")
    plt.legend()
    plt.grid(True, which="both", linestyle="--", alpha=0.6)
    plt.tight_layout()
    plt.savefig("comparison_residual.png", dpi=300, bbox_inches="tight")
    plt.close()

    # Comparison plot: relative error
    plt.figure(figsize=(7, 5))
    plt.semilogy(err_jac, linewidth=2, label="Jacobi")
    plt.semilogy(err_gs, linewidth=2, label="Gauss-Seidel")
    plt.semilogy(err_ssor, linewidth=2, label=f"SSOR ($\\omega={best_omega:.2f}$)")
    plt.xlabel("Iteration")
    plt.ylabel("Relative error")
    plt.title("Relative error comparison")
    plt.legend()
    plt.grid(True, which="both", linestyle="--", alpha=0.6)
    plt.tight_layout()
    plt.savefig("comparison_error.png", dpi=300, bbox_inches="tight")
    plt.close()


if __name__ == "__main__":
    main()