import marimo

__generated_with = "0.25.2"
app = marimo.App(width="medium")


@app.cell
def _():
    import matplotlib.pyplot as plt
    import numpy as np
    from numba import njit
    from scipy.sparse import lil_matrix, csr_array
    from scipy.sparse.linalg import spsolve
    from functools import partial
    import marimo as mo

    from matplotlib.animation import FuncAnimation
    from scipy.sparse import issparse


    return (
        FuncAnimation,
        csr_array,
        issparse,
        lil_matrix,
        mo,
        njit,
        np,
        partial,
        plt,
        spsolve,
    )


@app.cell
def _():
    EMPTY = 0
    DIFFUSING = 1
    SOLID = 2
    return DIFFUSING, EMPTY, SOLID


@app.cell
def _(SOLID, lil_matrix, np, spsolve):
    def solve_laplace(dla_matrix):
        N = dla_matrix.shape[0]
        A = lil_matrix((N**2, N**2), dtype=np.float64)
        b = np.zeros(N**2, dtype=np.float64)

        for i in range(N):
            for j in range(N):
                m = i * N + j

                if i == 0 or i == N - 1 or j == 0 or j == N - 1:
                    A[m, m] = 1
                    # b[m] = 0 is implicit 
                elif dla_matrix[i, j] == SOLID:
                    A[m, m] = 1
                    b[m] = 1
                else:
                    A[m, m] = 4
                    A[m, m + 1] = -1
                    A[m, m - 1] = -1
                    A[m, m + N] = -1
                    A[m, m - N] = -1
                    # b[m] = 0 # is implicit 
        A = A.tocsr()
        return spsolve(A, b).reshape(N, N)

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
                i_nb, j_nb = i + nb[0], j + nb[1]
                if i_nb == 0 or i_nb == N - 1 or j_nb == 0 or j_nb == N - 1:
                    continue

                if dla_matrix[i_nb, j_nb] != SOLID:
                    flux += 1 - p[i_nb, j_nb]
            fluxes[i, j] = np.abs(flux)
        return fluxes / np.sum(fluxes)

    return (estimate_normalized_boundary_fluxes,)


@app.cell
def _(EMPTY, SOLID, njit, np):
    @njit
    def estimate_empty_boundary_fluxes(p, mtx):
        N = mtx.shape[0]
        fluxes = np.zeros((N, N), dtype=np.float64)

        for i in range(1, N - 1):
            for j in range(1, N - 1):
                if mtx[i, j] != EMPTY:
                    continue

                n_solid = 0

                if mtx[i + 1, j] == SOLID:
                    n_solid += 1
                if mtx[i - 1, j] == SOLID:
                    n_solid += 1
                if mtx[i, j + 1] == SOLID:
                    n_solid += 1
                if mtx[i, j - 1] == SOLID:
                    n_solid += 1

                if n_solid > 0:
                    fluxes[i, j] = n_solid * max(0.0, 1.0 - p[i, j])

        total = np.sum(fluxes)

        if total > 0:
            fluxes /= total

        return fluxes

    return


@app.cell
def _(
    DIFFUSING,
    EMPTY,
    SOLID,
    csr_array,
    estimate_normalized_boundary_fluxes,
    njit,
    np,
    solve_laplace,
):
    @njit
    def dla_step(mtx, r_birth, nbs, probs=None):
        attached = False

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
                    attached = True
                    break
                continue

            # become solid if next to a solid
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

        return r_birth, attached

    def run_dla(n_dla=50, n_steps=1000):
        r_birth = 7

        mtx = np.zeros((n_dla, n_dla))
        mtx[n_dla // 2, n_dla // 2] = SOLID
        nbs = [(1, 0), (-1, 0), (0, 1), (0, -1)]

        for t in range(n_steps):
            r_birth, attached = dla_step(mtx, r_birth, nbs)

        return mtx

    def run_weighted_dla(probability_calc, n_dla=50, n_steps=1000, sequence = False):
        r_birth = 7
        nbs = [(1, 0), (-1, 0), (0, 1), (0, -1)]

        frames = []

        # start from a solid "cross"
        mtx = np.zeros((n_dla, n_dla))
        mtx[n_dla // 2, n_dla // 2] = SOLID
        for nb in nbs:
            (dx, dy) = nb
            mtx[n_dla // 2 + dx, n_dla // 2 + dy] = SOLID
        attached = True
        for t in range(n_steps):
            if attached:
                p_loc = solve_laplace(mtx)
                if sequence:
                    frames.append(csr_array(mtx))
            fluxes_loc = estimate_normalized_boundary_fluxes(p_loc, mtx)
            r_birth, attached = dla_step(
                mtx, r_birth, nbs, probs=probability_calc(fluxes_loc)
            )
    
        if sequence:
            return frames
        else:
            return mtx

    return run_dla, run_weighted_dla


@app.cell
def _(np):
    def nonlinear_inversion(
        fluxes: np.ndarray,
        k: float = 12.0,
        cutoff: float = 0.35,
        eps: float = 1e-4,
    ) -> np.ndarray:
        """
        Logistic sticking probability:

            P(q) = eps + (1 - eps) / (1 + exp(k * (q - cutoff)))

        Parameters
        ----------
        fluxes : np.ndarray
            Array of local boundary fluxes.
        k : float, default=12.0
            Logistic steepness. Higher values more strongly penalize
            high-flux attachment sites.
        cutoff : float, default=0.35
            Relative flux at the logistic midpoint.
        eps : float, default=1e-4
            Minimum sticking probability.

        Returns
        -------
        np.ndarray
            Sticking probabilities with the same shape as `fluxes`.
            Values lie in [eps, 1].
        """
        positive = fluxes[fluxes > 0]

        if positive.size == 0:
            return np.ones_like(fluxes, dtype=np.float64)

        q_med = np.median(positive)
        q = fluxes / q_med

        exponent = np.clip(k * (q - cutoff), -60, 60)

        return eps + (1.0 - eps) / (1.0 + np.exp(exponent))

    return (nonlinear_inversion,)


@app.cell
def _(run_dla, solve_laplace):
    mtx_reguar = run_dla(n_dla=50, n_steps=300)
    p_regular = solve_laplace(mtx_reguar)
    return mtx_reguar, p_regular


@app.cell
def _(p_regular, plt):
    plt.imshow(p_regular)
    return


@app.cell
def _(estimate_normalized_boundary_fluxes, mtx_reguar, p_regular):
    fluxes_regular = estimate_normalized_boundary_fluxes(p_regular, mtx_reguar)
    return (fluxes_regular,)


@app.cell
def _(fluxes_regular, mtx_reguar, plt):
    plt.imshow(mtx_reguar, cmap="gray")
    plt.imshow(fluxes_regular, cmap="hot", alpha=0.8)
    plt.show()
    return


@app.cell
def _(mtx_reguar, p_regular, plt):
    plt.imshow(mtx_reguar, cmap="gray")
    plt.imshow(p_regular, cmap="hot", alpha=0.8)
    plt.show()
    return


@app.cell
def _():
    # moderate_sigmoid = partial(
    #     nonlinear_inversion,
    #     k=12,
    #     cutoff=0.35,
    #     eps=1e-4,
    # )
    # mtx_weighted = run_weighted_dla(moderate_sigmoid, n_dla=100, n_steps=50_000)
    # plt.imshow(mtx_weighted)
    return


@app.cell
def _():
    # pw = solve_laplace(mtx_weighted)
    # pwfl = estimate_normalized_boundary_fluxes(pw, mtx_weighted)
    # plt.imshow(mtx_weighted, cmap="gray")
    # plt.imshow(moderate_sigmoid(pwfl), cmap="hot", alpha=0.8)
    # plt.show()
    return


@app.cell
def _(nonlinear_inversion, partial, run_weighted_dla):
    moderate_sigmoid = partial(
        nonlinear_inversion,
        k=8,
        cutoff=0.35,
        eps=1e-4,
    )
    seq = run_weighted_dla(moderate_sigmoid, n_dla=100, n_steps=50_000, sequence=True)
    return (seq,)


@app.cell
def _():
    from matplotlib.animation import PillowWriter


    return (PillowWriter,)


@app.cell
def _(plt):
    plt.show()
    return


@app.cell
def _(FuncAnimation, PillowWriter, issparse, mo, plt, seq):
    k = 50
    fps = 20        # Playback speed
    output = "dla_animation.mp4"

    def dense(frame):
        return frame.toarray() if issparse(frame) else frame

    fig, ax = plt.subplots(figsize=(7, 7))

    im = ax.imshow(
        dense(seq[0]),
        cmap="viridis",
        interpolation="nearest",
        vmin=0,
        vmax=1,
    )

    ax.axis("off")

    def update(i):
        im.set_data(dense(seq[i]))
        ax.set_title(f"DLA — Frame {i}")
        return (im,)

    # Select only every k-th frame
    frame_indices = range(0, len(seq), k)

    anim = FuncAnimation(
        fig,
        update,
        frames=frame_indices,
        interval=1000 / fps,
        blit=False,
    )

    anim.save(
        "dla_animation.gif",
        writer=PillowWriter(fps=fps),
        dpi=120,
    )


    plt.close(fig)

    mo.Html(anim.to_jshtml(fps=fps))
    return


@app.cell
def _():
    return


if __name__ == "__main__":
    app.run()
