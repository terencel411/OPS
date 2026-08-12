/*
 * eddy_kernels.h -- oSEM_3d's eddies as OPS particles
 *
 * ACC<T>&  is the grid accessor:     c(component, i, j, k)
 * ACCP<T>& is the particle accessor: p(component) -- already positioned on the
 *          current particle, so there is no (i,j,k).
 *
 * These replace ../oSEM_3d's instantiate_eddies and convect_eddies
 * (opensbliblock00_kernels.h:39-59), which run as ops_par_loops over a
 * {eddies,1,1} strip of the fluid block.
 *
 * ------------------------------------------------------------------
 * TWO POSITIONS, AND WHY
 * ------------------------------------------------------------------
 * An eddy carries its coordinates in `e`, an ordinary particle dat, NOT in the
 * particle position dat. The position dat `pos` is the ownership handle: OPS
 * used it once, at seeding, to hand each eddy to a rank, and nothing reads it
 * afterwards.
 *
 * They have to be separate because the eddy box is not inside the fluid block.
 * It overhangs on all three low faces -- x, y and z all start at -radius while
 * the block spans [0,375] x [0,100] x [0,40] -- and it is wider than the domain
 * in z (44.68 against 40). The bounding box OPS derives from a coordinate dat
 * is exactly [first node .. last node], so an eddy placed at its true
 * coordinates would frequently fall outside it, and outside means owned by no
 * rank. `pos` is therefore the true position clamped into the block, which
 * moves an eddy by at most one radius and only in the overhang, and hands it to
 * the rank holding the piece of inlet it actually influences.
 *
 * The eddies could not be given a block of their own instead: OPS partitions
 * the ranks BETWEEN blocks (ops_mpi_partition.cpp:152-154 gives each block
 * nproc/nblocks of them), so a second block would take half the ranks away from
 * the solver.
 */

#ifndef _EDDY_KERNELS_H_
#define _EDDY_KERNELS_H_

/* The three coordinate dats packed into one 3-component dat, which is what
   ops_create_bounding_box and ops_decl_mapping want. */
void KerPackCoords(ACC<double> &c, const ACC<double> &x0, const ACC<double> &x1,
                   const ACC<double> &x2) {
  c(0, 0, 0, 0) = x0(0, 0, 0);
  c(1, 0, 0, 0) = x1(0, 0, 0);
  c(2, 0, 0, 0) = x2(0, 0, 0);
}

/* ------------------------------------------------------------------ *
 * instantiate_eddies
 * ------------------------------------------------------------------ *
 * The coordinates are seeded on the host (seed_eddies in opensbli.cpp) because
 * the particle count has to be set there; this kernel writes the rest.
 *
 * `rnd` carries six independent uniforms in [0,1) per eddy, pre-filled by the
 * driver -- the same fill-then-read structure ../oSEM_3d uses with
 * ops_fill_random_uniform, and the only one available, since no OPS generator
 * can be called from inside a kernel.
 *
 * THE SIGN TEST DIFFERS FROM ../oSEM_3d, DELIBERATELY. There it is
 * `(rng < 0) ? -1 : 1` on an int dat filled by ops_fill_random_uniform, which
 * draws from uniform_int_distribution<int>(0, INT_MAX) (ops_lib_core.cpp:2604)
 * and so is never negative -- every sign comes out +1. Drawing doubles and
 * thresholding at 0.5 sidesteps the question. See the README.
 */
void KerInitEddy(ACCP<double> &r, ACCP<double> &eps, const ACCP<double> &rnd) {
  r(0) = radius;

  eps(0) = (rnd(3) < 0.5) ? -1.0 : 1.0;
  eps(1) = (rnd(4) < 0.5) ? -1.0 : 1.0;
  eps(2) = (rnd(5) < 0.5) ? -1.0 : 1.0;
}

/* ------------------------------------------------------------------ *
 * convect_eddies
 * ------------------------------------------------------------------ *
 * ../oSEM_3d/opensbliblock00_kernels.h:50-59, statement for statement: advance
 * x by one step, and if the eddy has left the box put it back at the inlet with
 * a fresh y, z and signs. The re-injection lines are the instantiation's lines
 * unchanged, which is what convect_eddies is.
 *
 * The teleport is kept. Ownership is by eddy id and was fixed at seeding, so no
 * migration runs and a jump to an arbitrary point in the box costs nothing. The
 * gather sums by global id, so it stays complete and exact whatever the
 * coordinates do.
 */
void KerConvectEddies(ACCP<double> &e, ACCP<double> &eps,
                      const ACCP<double> &rnd) {

  e(0) = e(0) + increment;

  if (e(0) > eddy_x_max) {
    e(0) = eddy_x_min;
    e(1) = eddy_y_min + rnd(1) * (eddy_y_max - eddy_y_min);
    e(2) = eddy_z_min + rnd(2) * (eddy_z_max - eddy_z_min);

    eps(0) = (rnd(3) < 0.5) ? -1.0 : 1.0;
    eps(1) = (rnd(4) < 0.5) ? -1.0 : 1.0;
    eps(2) = (rnd(5) < 0.5) ? -1.0 : 1.0;
  }
}

/* ------------------------------------------------------------------ *
 * The gather -- what replaces the MPI_Allgatherv
 * ------------------------------------------------------------------ *
 * ../oSEM_3d fetches seven eddy dats with ops_dat_fetch_data, which returns
 * only this rank's slice, and repairs that with a hand-rolled MPI_Allgatherv
 * over raw MPI_COMM_WORLD every timestep (opensbli.cpp:416-491).
 *
 * Here `all` is an ops_arg_reduce declared OPS_INC: every eddy writes into the
 * slot its global id picks out and every other rank contributes 0.0 there, so
 * the MPI_Allreduce inside ops_reduction_result is an allgather. It runs over
 * OPS_MPI_GLOBAL (ops_mpi_rt_support.cpp:1285), not the block communicator, so
 * every rank ends up with the complete list in id order -- the array shape
 * Kernel030 already wanted.
 *
 * Must be OPS_PARTICLE_ITERATE_LOCAL: a ghost eddy carries its owner's id and
 * would be added into the same slot twice.
 */
void KerPublishEddy(const ACCP<double> &e, const ACCP<double> &r,
                    const ACCP<double> &eps, const ACCP<int> &id, double *all) {
  const int s = NCOMP * id(0);
  all[s + E_X] += e(0);
  all[s + E_Y] += e(1);
  all[s + E_Z] += e(2);
  all[s + E_R] += r(0);
  all[s + E_SX] += eps(0);
  all[s + E_SY] += eps(1);
  all[s + E_SZ] += eps(2);
}

/* Population check. Eddies recycle, they are never created or destroyed, so
   anything other than the seeded total means one was lost. */
void KerCountEddies(const ACCP<int> &id, int *count) {
  (void)id;
  *count += 1;
}

#endif /* _EDDY_KERNELS_H_ */
