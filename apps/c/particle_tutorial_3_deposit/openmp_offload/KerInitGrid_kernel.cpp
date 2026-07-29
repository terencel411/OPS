// Auto-generated at 2026-07-29 00:37:16.441180 by ops-translator


//  ==================
//  Host stub function
//  ==================
#ifndef OPS_LAZY
void ops_par_loop_KerInitGrid(
    const char *name,
    ops_block block,
    int dim,
    int *range,
    ops_arg arg0,
    ops_arg arg1,
    ops_arg arg2
)
{ 
#else
void ops_par_loop_KerInitGrid_execute(ops_kernel_descriptor *desc)
{
    ops_block block = desc->block;
    int dim = desc->dim;
    int *range = desc->range;
    ops_arg arg0 = desc->args[0];
    ops_arg arg1 = desc->args[1];
    ops_arg arg2 = desc->args[2];
#endif

//  ======
//  Timing
//  ======
    double __t1, __t2, __c1, __c2;

    ops_arg args[3];

    args[0] = arg0;
    args[1] = arg1;
    args[2] = arg2;

#if defined(CHECKPOINTING) && !defined(OPS_LAZY)
    if (!ops_checkpointing_before(args, 3, range, 1)) return;
#endif

    if (block->instance->OPS_diags > 1)
    {
        ops_timing_realloc(block->instance, 1, "KerInitGrid");
        block->instance->OPS_kernels[1].count++;
        ops_timers_core(&__c1, &__t1);
    }

#ifdef OPS_DEBUG
    ops_register_args(block->instance, args, "KerInitGrid");
#endif

//  =================================================
//  compute locally allocated range for the sub-block
//  =================================================
    int start_indx[2];
    int end_indx[2];
    int arg_idx[2];

#if defined(OPS_LAZY) || !defined(OPS_MPI)
    for (int n = 0; n < 2; n++) {
        start_indx[n] = range[2*n];
        end_indx[n]   = range[2*n+1];
    }
#else
    if (compute_ranges(args, 3, block, range, start_indx, end_indx, arg_idx) < 0) return;
#endif

    int start_indx0 = start_indx[0], end_indx0 = end_indx[0];
    int start_indx1 = start_indx[1], end_indx1 = end_indx[1];

#if defined(OPS_MPI)
#if defined(OPS_LAZY)
    sub_block_list sb = OPS_sub_block_list[block->index];
    arg_idx[0] = sb->decomp_disp[0];
    arg_idx[1] = sb->decomp_disp[1];
#else
    arg_idx[0] -= start_indx[0];
    arg_idx[1] -= start_indx[1];
#endif  //OPS_LAZY
#else //OPS_MPI
    arg_idx[0] = 0;
    arg_idx[1] = 0;
#endif //OPS_MPI

//  ======================================================
//  Initialize global variable with the dimensions of dats
//  ======================================================
    int xdim0_KerInitGrid = args[0].dat->size[0];
    int ydim0_KerInitGrid = args[0].dat->size[1];

//  =======================================================
//  Set up initial pointers and exchange halos if necessary
//  =======================================================
    int base0 = args[0].dat->base_offset;
    double * __restrict__ xf_p = (double *)(args[0].data_d + base0);

    int consts_bytes = 0;

    double *arg1h = (double *)args[1].data;
    int dx_dim = args[1].dim;

    consts_bytes += ROUND_UP(dx_dim*sizeof(double));

    reallocConstArrays(block->instance, consts_bytes);
    consts_bytes = 0;

    args[1].data = block->instance->OPS_consts_h + consts_bytes;
    args[1].data_d = block->instance->OPS_consts_d + consts_bytes;
    for (int d = 0; d < dx_dim; d++)
        ((double *)args[1].data)[d] = arg1h[d];

    consts_bytes += ROUND_UP(dx_dim*sizeof(double));

    mvConstArraysToDevice(block->instance, consts_bytes);

    double * __restrict__ dx = (double *)args[1].data_d;

//  ==============
//  Halo exchanges
//  ==============
#ifndef OPS_LAZY
    ops_H_D_exchanges_device(args, 3);
    ops_halo_exchanges(args, 3, range);
#endif //OPS_LAZY

    if (block->instance->OPS_diags > 1) {
        ops_timers_core(&__c2, &__t2);
        block->instance->OPS_kernels[1].mpi_time += __t2 - __t1;
    }

    #pragma omp target teams distribute parallel for collapse(2)
        for (int n_y = start_indx1; n_y < end_indx1; n_y++)
        {
            for (int n_x = start_indx0; n_x < end_indx0; n_x++)
            {
                int idx[] = {arg_idx[0] + n_x, arg_idx[1] + n_y};
#ifdef OPS_SOA
                 ACC<double> xf(2,xdim0_KerInitGrid, ydim0_KerInitGrid, xf_p + (n_x * 1) + (n_y * xdim0_KerInitGrid * 1));
#else
                 ACC<double> xf(2,xdim0_KerInitGrid, ydim0_KerInitGrid, xf_p + 2 * ((n_x * 1) + (n_y * xdim0_KerInitGrid * 1)));
#endif

  xf(0, 0, 0) = (*dx) * static_cast<double>(idx[0]);
  xf(1, 0, 0) = (*dx) * static_cast<double>(idx[1]);

            }
        }

    if (block->instance->OPS_diags > 1)
    {
        ops_timers_core(&__c1, &__t1);
        block->instance->OPS_kernels[1].time += __t1 - __t2;
    }

#ifndef OPS_LAZY
    ops_set_dirtybit_device(args, 3);
    ops_set_halo_dirtybit3(&args[0], range);
#endif

    if (block->instance->OPS_diags > 1)
    {
//      ====================
//      Update kernel record
//      ====================
        ops_timers_core(&__c2, &__t2);
        block->instance->OPS_kernels[1].mpi_time += __t2 -__t1;
        block->instance->OPS_kernels[1].transfer += ops_compute_transfer(dim, start_indx, end_indx, &arg0);
    }
}

#ifdef OPS_LAZY
void ops_par_loop_KerInitGrid(
    const char *name,
    ops_block block,
    int dim,
    int *range,
    ops_arg arg0,
    ops_arg arg1,
    ops_arg arg2
    )
{
    ops_arg args[3];

    args[0] = arg0;
    args[1] = arg1;
    args[2] = arg2;

    create_kerneldesc_and_enque("KerInitGrid", args, 3, 1, dim, 1, range, block, ops_par_loop_KerInitGrid_execute);
}
#endif
