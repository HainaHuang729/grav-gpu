#include <math.h>
#include <stddef.h>
#include <stdint.h>

static inline void cic_axis(double coord, double box_len, int grid,
                            int *ix0_out, int *ix1_out,
                            double *w0_out, double *w1_out) {
    /* Match np.remainder(coord / box_len, 1.0) * grid. */
    double q = coord / box_len;
    double r = q - floor(q);
    double p = r * (double)grid;
    int ix0 = (int)p;
    double f = p - (double)ix0;
    if (ix0 >= grid) {
        ix0 %= grid;
    } else if (ix0 < 0) {
        ix0 = (ix0 % grid + grid) % grid;
    }
    int ix1 = (ix0 + 1 == grid) ? 0 : ix0 + 1;
    *ix0_out = ix0;
    *ix1_out = ix1;
    *w0_out = 1.0 - f;
    *w1_out = f;
}

static inline void deposit_cic_impl(int64_t n, int grid, const double *coords,
                                    const double *vels, const double *vel_adj,
                                    const double *box, double *density,
                                    double *mom0, double *mom1, double *mom2,
                                    double *ma0, double *ma1, double *ma2,
                                    int has_adj) {
    const int64_t g = (int64_t)grid;
    const int64_t g2 = g * g;
    const double bx = box[0];
    const double by = box[1];
    const double bz = box[2];

    for (int64_t i = 0; i < n; ++i) {
        int ix0, ix1, iy0, iy1, iz0, iz1;
        double wx0, wx1, wy0, wy1, wz0, wz1;
        const double *c = coords + 3 * i;
        cic_axis(c[0], bx, grid, &ix0, &ix1, &wx0, &wx1);
        cic_axis(c[1], by, grid, &iy0, &iy1, &wy0, &wy1);
        cic_axis(c[2], bz, grid, &iz0, &iz1, &wz0, &wz1);

        const double vx = vels[3 * i + 0];
        const double vy = vels[3 * i + 1];
        const double vz = vels[3 * i + 2];
        const double vax = has_adj ? vel_adj[3 * i + 0] : 0.0;
        const double vay = has_adj ? vel_adj[3 * i + 1] : 0.0;
        const double vaz = has_adj ? vel_adj[3 * i + 2] : 0.0;

        const int64_t x0 = (int64_t)ix0 * g2;
        const int64_t x1 = (int64_t)ix1 * g2;
        const int64_t y0 = (int64_t)iy0 * g;
        const int64_t y1 = (int64_t)iy1 * g;

        const int64_t idx[8] = {
            x0 + y0 + iz0, x0 + y0 + iz1,
            x0 + y1 + iz0, x0 + y1 + iz1,
            x1 + y0 + iz0, x1 + y0 + iz1,
            x1 + y1 + iz0, x1 + y1 + iz1
        };
        const double w[8] = {
            wx0 * wy0 * wz0, wx0 * wy0 * wz1,
            wx0 * wy1 * wz0, wx0 * wy1 * wz1,
            wx1 * wy0 * wz0, wx1 * wy0 * wz1,
            wx1 * wy1 * wz0, wx1 * wy1 * wz1
        };

        for (int k = 0; k < 8; ++k) {
            const int64_t m = idx[k];
            const double wk = w[k];
            density[m] += wk;
            mom0[m] += wk * vx;
            mom1[m] += wk * vy;
            mom2[m] += wk * vz;
            if (has_adj) {
                ma0[m] += wk * vax;
                ma1[m] += wk * vay;
                ma2[m] += wk * vaz;
            }
        }
    }
}

void deposit_cic_c(int64_t n, int grid, const double *coords, const double *vels,
                   const double *vel_adj, const double *box, double *density,
                   double *mom0, double *mom1, double *mom2,
                   double *ma0, double *ma1, double *ma2) {
    if (vel_adj != NULL && ma0 != NULL && ma1 != NULL && ma2 != NULL) {
        deposit_cic_impl(n, grid, coords, vels, vel_adj, box, density,
                         mom0, mom1, mom2, ma0, ma1, ma2, 1);
    } else {
        deposit_cic_impl(n, grid, coords, vels, NULL, box, density,
                         mom0, mom1, mom2, NULL, NULL, NULL, 0);
    }
}
