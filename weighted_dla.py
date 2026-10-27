import marimo

__generated_with = "0.25.2"
app = marimo.App(width="medium")


@app.cell
def _():
    import numpy as np
    import matplotlib.pyplot as plt
    from concave_hull import concave_hull
    from scipy import linalg

    from numba import njit

    return np, plt


@app.cell
def _():
    EMPTY = 0
    DIFFUSING = 1
    SOLID = 2
    return DIFFUSING, EMPTY, SOLID


@app.cell
def _(SOLID, np):
    def solve_laplace(dla_matrix):
      N = dla_matrix.shape[0]
      A = np.zeros((N**2, N**2))
      b = np.zeros(N**2)

      for i in range(N):
        for j in range(N):
          m = i*N + j

          if i == 0 or i == N-1 or j == 0 or j == N-1:
            A[m, m] = 1
            b[m] = 0
          elif dla_matrix[i, j] == SOLID: 
            A[m, m] = 1
            b[m] = 1
          else:
            A[m, m] = 4
            A[m, m+1] = -1
            A[m, m-1] = -1
            A[m, m+N] = -1
            A[m, m-N] = -1
            b[m] = 0

      return np.linalg.solve(A, b).reshape(N, N)

    return (solve_laplace,)


@app.cell
def _(SOLID, np):
    def estimate_normalized_boundary_fluxes(p, dla_matrix):
        N = dla_matrix.shape[0]
        solid_nodes = np.argwhere(dla_matrix == SOLID)
        fluxes = np.zeros_like(dla_matrix)
    
        nbs = [(1, 0), (-1, 0), (0, 1), (0, -1)]
        for (i, j) in solid_nodes:
            flux = 0
            for nb in nbs:
                i_nb, j_nb = i + nb[0], j + nb[0]
                if i_nb == 0 or i_nb == N-1 or j_nb == 0 or j_nb == N-1:
                    continue
           
                if dla_matrix[i_nb, j_nb] != SOLID:
                    flux += 1 - p[i_nb, j_nb]
            fluxes[i, j] = np.abs(flux)    
        return fluxes/np.sum(fluxes)

    return (estimate_normalized_boundary_fluxes,)


@app.cell
def _():
    # def dla_step(mtx, r_birth):
    return


@app.cell
def _(DIFFUSING, EMPTY, SOLID, np):
    def dla_step(mtx, n_dla, r_birth, nbs):
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

            if i_new == 0 or i_new == n_dla-1 or j_new == 0 or j_new == n_dla-1:
                # kill and restart
                mtx[i, j] = EMPTY
                break

            # don't overwrite a solid cell
            if mtx[i_new, j_new] == SOLID:
                mtx[i, j] = SOLID
                break

            # become solid if next to a solid
            attached = False
            for nb in nbs:
                if mtx[i_new + nb[0], j_new + nb[1]] == SOLID:
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
        max_radius = np.max(np.linalg.norm(solid_positions - n_dla//2, axis=1))
        r_birth = min(int(max_radius) + 5, n_dla//2 - 2)

        return r_birth


    def run_dla(n_dla=50, n_steps=1000):
        r_birth = n_dla - 30

        mtx = np.zeros((n_dla, n_dla))
        mtx[n_dla//2, n_dla//2] = SOLID
        nbs = [(1, 0), (-1, 0), (0, 1), (0, -1)]

        for t in range(n_steps):
            r_birth = dla_step(mtx, n_dla, r_birth, nbs)

        return mtx




    return (run_dla,)


@app.cell
def _(run_dla, solve_laplace):
    mtx = run_dla()
    p = solve_laplace(mtx)
    return mtx, p


@app.cell
def _(p, plt):
    plt.imshow(p)
    return


@app.cell
def _(estimate_normalized_boundary_fluxes, mtx, p):
    fluxes = estimate_normalized_boundary_fluxes(p, mtx)
    return (fluxes,)


@app.cell
def _(fluxes, mtx, plt):
    plt.imshow(mtx, cmap="gray")
    plt.imshow(fluxes, cmap="hot", alpha=0.8)
    plt.show()
    return


@app.cell
def _(mtx, p, plt):
    plt.imshow(mtx, cmap="gray")
    plt.imshow(p, cmap="hot", alpha=0.8)
    plt.show()
    return


@app.cell
def _():
    return


if __name__ == "__main__":
    app.run()
