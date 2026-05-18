import numpy as np
import scipy.sparse as sp
import scipy.sparse.linalg as spla
import matplotlib.pyplot as plt


def build_2d_laplacian(m):
    h = 1.0 / (m + 1)

    main = 2.0 * np.ones(m)
    off = -1.0 * np.ones(m - 1)

    T = sp.diags([off, main, off], [-1, 0, 1], format="csr")
    I = sp.eye(m, format="csr")

    A = (sp.kron(I, T) + sp.kron(T, I)) / h**2
    return A.tocsr(), h


def relative_residual(A, x, b, b_norm):
    r = b - A @ x
    return np.linalg.norm(r) / b_norm


def relative_error(x, xtrue, xtrue_norm):
    return np.linalg.norm(x - xtrue) / xtrue_norm


def cg_naive(A, b, xtrue, tol=1e-8, maxit=5000):
    #alpha_k = <b,p_k> / <p_k, A p_k>
    #beta_k  = - <r_k, A p_k> / <p_k, A p_k>
    n = len(b)
    x = np.zeros(n)

    r = b - A @ x
    p = r.copy()
    b_norm = np.linalg.norm(b)
    xtrue_norm = np.linalg.norm(xtrue)
    res_hist = [np.linalg.norm(r) / b_norm]
    err_hist = [relative_error(x, xtrue, xtrue_norm)]

    for k in range(maxit):
        Ap = A @ p
        pAp = np.dot(p, Ap)
        alpha = np.dot(b, p) / pAp
        x = x + alpha * p
        r_new = r - alpha * Ap
        res_hist.append(np.linalg.norm(r_new) / b_norm)
        err_hist.append(relative_error(x, xtrue, xtrue_norm))

        if res_hist[-1] < tol:
            break

        beta = -np.dot(r_new, Ap) / pAp

        p = r_new + beta * p
        r = r_new

    return x, res_hist, err_hist


def cg_book(A, b, xtrue, tol=1e-8, maxit=5000):
    #alpha_k = <r_{k-1}, r_{k-1}> / <p_k, A p_k>
    #beta_k  = <r_k, r_k> / <r_{k-1}, r_{k-1}>

    n = len(b)
    x = np.zeros(n)

    r = b - A @ x
    p = r.copy()

    b_norm = np.linalg.norm(b)
    xtrue_norm = np.linalg.norm(xtrue)
    res_hist = [np.linalg.norm(r) / b_norm]
    err_hist = [relative_error(x, xtrue, xtrue_norm)]
    rr_old = np.dot(r, r)

    for k in range(maxit):
        Ap = A @ p
        alpha = rr_old / np.dot(p, Ap)

        x = x + alpha * p
        r = r - alpha * Ap

        rr_new = np.dot(r, r)

        res_hist.append(np.sqrt(rr_new) / b_norm)
        err_hist.append(relative_error(x, xtrue, xtrue_norm))

        if res_hist[-1] < tol:
            break

        beta = rr_new / rr_old
        p = r + beta * p
        rr_old = rr_new

    return x, res_hist, err_hist


def jacobi_2d(m, b, A, xtrue, tol=1e-8, maxit=5000):
    h = 1.0 / (m + 1)

    u = np.zeros((m, m))
    f = b.reshape((m, m))

    b_norm = np.linalg.norm(b)
    xtrue_norm = np.linalg.norm(xtrue)

    res_hist = [relative_residual(A, u.ravel(), b, b_norm)]
    err_hist = [relative_error(u.ravel(), xtrue, xtrue_norm)]

    for k in range(maxit):
        old = u.copy()

        for i in range(m):
            for j in range(m):
                up = old[i - 1, j] if i > 0 else 0.0
                down = old[i + 1, j] if i < m - 1 else 0.0
                left = old[i, j - 1] if j > 0 else 0.0
                right = old[i, j + 1] if j < m - 1 else 0.0

                u[i, j] = 0.25 * (h**2 * f[i, j] + up + down + left + right)

        u_flat = u.ravel()
        res_hist.append(relative_residual(A, u_flat, b, b_norm))
        err_hist.append(relative_error(u_flat, xtrue, xtrue_norm))

        if res_hist[-1] < tol:
            break

    return u.ravel(), res_hist, err_hist


def gauss_seidel_2d(m, b, A, xtrue, tol=1e-8, maxit=5000):
    h = 1.0 / (m + 1)

    u = np.zeros((m, m))
    f = b.reshape((m, m))

    b_norm = np.linalg.norm(b)
    xtrue_norm = np.linalg.norm(xtrue)

    res_hist = [relative_residual(A, u.ravel(), b, b_norm)]
    err_hist = [relative_error(u.ravel(), xtrue, xtrue_norm)]

    for k in range(maxit):
        for i in range(m):
            for j in range(m):
                up = u[i - 1, j] if i > 0 else 0.0
                down = u[i + 1, j] if i < m - 1 else 0.0
                left = u[i, j - 1] if j > 0 else 0.0
                right = u[i, j + 1] if j < m - 1 else 0.0

                u[i, j] = 0.25 * (h**2 * f[i, j] + up + down + left + right)

        u_flat = u.ravel()
        res_hist.append(relative_residual(A, u_flat, b, b_norm))
        err_hist.append(relative_error(u_flat, xtrue, xtrue_norm))

        if res_hist[-1] < tol:
            break

    return u.ravel(), res_hist, err_hist


def ssor_2d(m, b, A, xtrue, omega, tol=1e-8, maxit=5000):
    h = 1.0 / (m + 1)

    u = np.zeros((m, m))
    f = b.reshape((m, m))

    b_norm = np.linalg.norm(b)
    xtrue_norm = np.linalg.norm(xtrue)

    res_hist = [relative_residual(A, u.ravel(), b, b_norm)]
    err_hist = [relative_error(u.ravel(), xtrue, xtrue_norm)]

    for k in range(maxit):
        # Forward SOR sweep
        for i in range(m):
            for j in range(m):
                up = u[i - 1, j] if i > 0 else 0.0
                down = u[i + 1, j] if i < m - 1 else 0.0
                left = u[i, j - 1] if j > 0 else 0.0
                right = u[i, j + 1] if j < m - 1 else 0.0

                gs_value = 0.25 * (h**2 * f[i, j] + up + down + left + right)
                u[i, j] = (1 - omega) * u[i, j] + omega * gs_value

        # Backward SOR sweep
        for i in range(m - 1, -1, -1):
            for j in range(m - 1, -1, -1):
                up = u[i - 1, j] if i > 0 else 0.0
                down = u[i + 1, j] if i < m - 1 else 0.0
                left = u[i, j - 1] if j > 0 else 0.0
                right = u[i, j + 1] if j < m - 1 else 0.0

                gs_value = 0.25 * (h**2 * f[i, j] + up + down + left + right)
                u[i, j] = (1 - omega) * u[i, j] + omega * gs_value

        u_flat = u.ravel()
        res_hist.append(relative_residual(A, u_flat, b, b_norm))
        err_hist.append(relative_error(u_flat, xtrue, xtrue_norm))

        if res_hist[-1] < tol:
            break

    return u.ravel(), res_hist, err_hist


def save_cg_comparison(res_naive, err_naive, res_book, err_book):
    plt.figure(figsize=(7, 5))
    plt.semilogy(res_naive, linewidth=2, label="Naive CG")
    plt.semilogy(res_book, linewidth=2, linestyle="--", label="Book CG")
    plt.xlabel("Iteration")
    plt.ylabel("Relative residual")
    plt.title("CG comparison: relative residual")
    plt.legend()
    plt.grid(True, which="both", linestyle="--", alpha=0.6)
    plt.tight_layout()
    plt.savefig("cg_compare_residual.png", dpi=300, bbox_inches="tight")
    plt.close()

    plt.figure(figsize=(7, 5))
    plt.semilogy(err_naive, linewidth=2, label="Naive CG")
    plt.semilogy(err_book, linewidth=2, linestyle="--", label="Book CG")
    plt.xlabel("Iteration")
    plt.ylabel("Relative error")
    plt.title("CG comparison: relative error")
    plt.legend()
    plt.grid(True, which="both", linestyle="--", alpha=0.6)
    plt.tight_layout()
    plt.savefig("cg_compare_error.png", dpi=300, bbox_inches="tight")
    plt.close()


def save_all_methods_comparison(
    res_book,
    err_book,
    res_gs,
    err_gs,
    res_ssor,
    err_ssor,
    res_jacobi,
    err_jacobi,
    best_omega
):
    plt.figure(figsize=(7, 5))
    plt.semilogy(res_book, linewidth=2, label="CG")
    plt.semilogy(res_gs, linewidth=2, label="Gauss-Seidel")
    plt.semilogy(res_ssor, linewidth=2, label=f"SSOR, omega={best_omega:.2f}")
    plt.semilogy(res_jacobi, linewidth=2, label="Jacobi")
    plt.xlabel("Iteration")
    plt.ylabel("Relative residual")
    plt.title("Method comparison: relative residual")
    plt.legend()
    plt.grid(True, which="both", linestyle="--", alpha=0.6)
    plt.tight_layout()
    plt.savefig("all_methods_residual.png", dpi=300, bbox_inches="tight")
    plt.close()

    plt.figure(figsize=(7, 5))
    plt.semilogy(err_book, linewidth=2, label="CG")
    plt.semilogy(err_gs, linewidth=2, label="Gauss-Seidel")
    plt.semilogy(err_ssor, linewidth=2, label=f"SSOR, omega={best_omega:.2f}")
    plt.semilogy(err_jacobi, linewidth=2, label="Jacobi")
    plt.xlabel("Iteration")
    plt.ylabel("Relative error")
    plt.title("Method comparison: relative error")
    plt.legend()
    plt.grid(True, which="both", linestyle="--", alpha=0.6)
    plt.tight_layout()
    plt.savefig("all_methods_error.png", dpi=300, bbox_inches="tight")
    plt.close()




def main():
    m = 20 # as assigned in the problem statement
    A, h = build_2d_laplacian(m)

    n = m * m
    b = np.ones(n)

    tol = 1e-8
    maxit = 5000

    print(f"Grid: {m} by {m} interior points")
    print(f"Matrix size: {A.shape}")
    print(f"h = {h:.6f}")

    # Direct solution for reference
    utrue = spla.spsolve(A, b)

    # CG methods
    u_naive, res_naive, err_naive = cg_naive(A, b, utrue, tol=tol, maxit=maxit)
    u_book, res_book, err_book = cg_book(A, b, utrue, tol=tol, maxit=maxit)

    print()
    print("CG comparison")
    print(f"Naive CG iterations: {len(res_naive) - 1}")
    print(f"Book CG iterations:  {len(res_book) - 1}")

    save_cg_comparison(res_naive, err_naive, res_book, err_book)

    # Stationary methods
    u_jacobi, res_jacobi, err_jacobi = jacobi_2d(
        m, b, A, utrue, tol=tol, maxit=maxit
    )

    u_gs, res_gs, err_gs = gauss_seidel_2d(
        m, b, A, utrue, tol=tol, maxit=maxit
    )

    print()
    print("Stationary methods")
    print(f"Jacobi iterations:       {len(res_jacobi) - 1}")
    print(f"Gauss-Seidel iterations: {len(res_gs) - 1}")

    # SSOR grid search
    omega_values = np.linspace(0.1, 1.95, 38)

    best_omega = None
    best_iters = None
    best_ssor_data = None

    omega_iters = []

    for omega in omega_values:
        u_ssor, res_ssor, err_ssor = ssor_2d(
            m, b, A, utrue, omega=omega, tol=tol, maxit=maxit
        )

        iters = len(res_ssor) - 1
        omega_iters.append(iters)

        if best_iters is None or iters < best_iters:
            best_iters = iters
            best_omega = omega
            best_ssor_data = (u_ssor, res_ssor, err_ssor)

    u_ssor, res_ssor, err_ssor = best_ssor_data

    print(f"Best SSOR omega:         {best_omega:.4f}")
    print(f"Best SSOR iterations:    {best_iters}")



    save_all_methods_comparison(
        res_book,
        err_book,
        res_gs,
        err_gs,
        res_ssor,
        err_ssor,
        res_jacobi,
        err_jacobi,
        best_omega
    )


    print()
    print("Saved figures:")
    print("  cg_compare_residual.png")
    print("  cg_compare_error.png")
    print("  all_methods_residual.png")
    print("  all_methods_error.png")
    print("  solution_direct.png")


if __name__ == "__main__":
    main()