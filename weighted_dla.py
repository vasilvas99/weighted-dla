import marimo

__generated_with = "0.25.2"
app = marimo.App(width="medium")


@app.cell
def _():
    import matplotlib.pyplot as plt
    import numpy as np
    from numba import njit

    return njit, np, plt


@app.cell
def _():
    EMPTY = 0
    DIFFUSING = 1
    SOLID = 2
    return DIFFUSING, EMPTY, SOLID


@app.cell
def _(SOLID, njit, np):
    @njit
    def solve_laplace(dla_matrix):
        N = dla_matrix.shape[0]
        A = np.zeros((N**2, N**2))
        b = np.zeros(N**2)

        for i in range(N):
            for j in range(N):
                m = i * N + j

                if i == 0 or i == N - 1 or j == 0 or j == N - 1:
                    A[m, m] = 1
                    b[m] = 0
                elif dla_matrix[i, j] == SOLID:
                    A[m, m] = 1
                    b[m] = 1
                else:
                    A[m, m] = 4
                    A[m, m + 1] = -1
                    A[m, m - 1] = -1
                    A[m, m + N] = -1
                    A[m, m - N] = -1
                    b[m] = 0

        return np.linalg.solve(A, b).reshape(N, N)

    return (solve_laplace,)


@app.cell
def _(SOLID, njit, np):
    @njit
    def estimate_normalized_boundary_fluxes(p, dla_matrix):
        N = dla_matrix.shape[0]
        solid_nodes = np.argwhere(dla_matrix == SOLID)
        fluxes = np.zeros_like(dla_matrix)

        nbs = [(1, 0), (-1, 0), (0, 1), (0, -1)]
        for i, j in solid_nodes:
            flux = 0
            for nb in nbs:
                i_nb, j_nb = i + nb[0], j + nb[0]
                if i_nb == 0 or i_nb == N - 1 or j_nb == 0 or j_nb == N - 1:
                    continue

                if dla_matrix[i_nb, j_nb] != SOLID:
                    flux += 1 - p[i_nb, j_nb]
            fluxes[i, j] = np.abs(flux)
        return fluxes / np.sum(fluxes)

    return (estimate_normalized_boundary_fluxes,)


@app.cell
def _(
    DIFFUSING,
    EMPTY,
    SOLID,
    estimate_normalized_boundary_fluxes,
    njit,
    np,
    solve_laplace,
):
    @njit
    def dla_step(mtx, r_birth, nbs, probs=None):
        if probs is None:
            probs = np.ones_like(mtx)
        n_dla = mtx.shape[0]

        while True:
            # spawn a walker
            theta = np.random.uniform(0, 2 * np.pi)
            i = int(n_dla // 2 + r_birth * np.cos(theta))
            j = int(n_dla // 2 + r_birth * np.sin(theta))

            if i == n_dla // 2 and j == n_dla // 2:
                i -= 1

            if mtx[i, j] != EMPTY:
                continue

            mtx[i, j] = DIFFUSING
            break

        while True:
            jump_rng = np.random.randint(0, 4)
            di, dj = nbs[jump_rng]
            i_new, j_new = i + di, j + dj

            if i_new == 0 or i_new == n_dla - 1 or j_new == 0 or j_new == n_dla - 1:
                # kill and restart
                mtx[i, j] = EMPTY
                break

            # don't overwrite a solid cell
            if mtx[i_new, j_new] == SOLID:
                if np.random.rand() < probs[i_new, j_new]:
                    mtx[i, j] = SOLID
                    break
                continue

            # become solid if next to a solid
            attached = False
            for nb in nbs:
                if mtx[i_new + nb[0], j_new + nb[1]] == SOLID:
                    if np.random.rand() < probs[i_new + nb[0], j_new + nb[1]]:
                        mtx[i_new, j_new] = SOLID
                        mtx[i, j] = EMPTY
                        attached = True
                        break
            if attached:
                break

            mtx[i, j] = EMPTY
            mtx[i_new, j_new] = DIFFUSING
            i, j = i_new, j_new

        # increase spawning radius as the cluster grows
        solid_positions = np.argwhere(mtx == SOLID)
        max_radius = np.sqrt(
            np.max(
                (solid_positions[:, 0] - n_dla // 2) ** 2
                + (solid_positions[:, 1] - n_dla // 2) ** 2
            )
        )
        r_birth = min(int(max_radius) + 5, n_dla // 2 - 2)

        return r_birth

    def run_dla(n_dla=50, n_steps=1000):
        r_birth = n_dla - 30

        mtx = np.zeros((n_dla, n_dla))
        mtx[n_dla // 2, n_dla // 2] = SOLID
        nbs = [(1, 0), (-1, 0), (0, 1), (0, -1)]

        for t in range(n_steps):
            r_birth = dla_step(mtx, r_birth, nbs)

        return mtx

    def run_weighted_dla(n_dla=50, n_steps=1000):
        r_birth = n_dla - 30
        nbs = [(1, 0), (-1, 0), (0, 1), (0, -1)]

        # start from a solid "cross"
        mtx = np.zeros((n_dla, n_dla))
        mtx[n_dla // 2, n_dla // 2] = SOLID
        for nb in nbs:
            (dx, dy) = nb
            mtx[n_dla // 2 + dx, n_dla // 2 + dy] = SOLID

        for t in range(n_steps):
            p_loc = solve_laplace(mtx)
            fluxes_loc = estimate_normalized_boundary_fluxes(p_loc, mtx)
            r_birth = dla_step(mtx, r_birth, nbs, probs=1 - fluxes_loc)

        return mtx

    return run_dla, run_weighted_dla


@app.cell
def _(run_dla, solve_laplace):
    mtx_reguar = run_dla(n_dla=150, n_steps=1500)
    p_regular = solve_laplace(mtx_reguar)
    return mtx_reguar, p_regular


@app.cell
def _(p_regular, plt):
    plt.imshow(p_regular)


@app.cell
def _(estimate_normalized_boundary_fluxes, mtx_reguar, p_regular):
    fluxes_regular = estimate_normalized_boundary_fluxes(p_regular, mtx_reguar)
    return (fluxes_regular,)


@app.cell
def _(fluxes_regular, mtx_reguar, plt):
    plt.imshow(mtx_reguar, cmap="gray")
    plt.imshow(fluxes_regular, cmap="hot", alpha=0.8)
    plt.show()


@app.cell
def _(mtx_reguar, p_regular, plt):
    plt.imshow(mtx_reguar, cmap="gray")
    plt.imshow(p_regular, cmap="hot", alpha=0.8)
    plt.show()


@app.cell
def _(plt, run_weighted_dla):
    mtx_weighted = run_weighted_dla(n_dla=50, n_steps=2000)
    plt.imshow(mtx_weighted)
    return (mtx_weighted,)


@app.cell
def _(estimate_normalized_boundary_fluxes, mtx_weighted, plt, solve_laplace):
    pw = solve_laplace(mtx_weighted)
    pwfl = estimate_normalized_boundary_fluxes(pw, mtx_weighted)
    plt.imshow(mtx_weighted, cmap="gray")
    plt.imshow(1 - pwfl, cmap="hot", alpha=0.8)
    plt.show()


if __name__ == "__main__":
    app.run()
