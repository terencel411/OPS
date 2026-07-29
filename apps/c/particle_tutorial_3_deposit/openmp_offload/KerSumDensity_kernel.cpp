// Auto-generated at 2026-07-29 00:37:16.447299 by ops-translator


//  ==================
//  Host stub function
//  ==================
#ifndef OPS_LAZY
void ops_par_loop_KerSumDensity(
    const char *name,
    ops_block block,
    int dim,
    int *range,
    ops_arg arg0,
    ops_arg arg1
)
{ 
#else
void ops_par_loop_KerSumDensity_execute(ops_kernel_descriptor *desc)
{
    ops_block block = desc->block;
    int dim = desc->dim;
    int *range = desc->range;
    ops_arg arg0 = desc->args[0];
    ops_arg arg1 = desc->args[1];
#endif

//  ======
//  Timing
//  ======
    double __t1, __t2, __c1, __c2;

    ops_arg args[2];

    args[0] = arg0;
    args[1] = arg1;

#if defined(CHECKPOINTING) && !defined(OPS_LAZY)
    if (!ops_checkpointing_before(args, 2, range, 3)) return;
#endif

    if (block->instance->OPS_diags > 1)
    {
        ops_timing_realloc(block->instance, 3, "KerSumDensity");
        block->instance->OPS_kernels[3].count++;
        ops_timers_core(&__c1, &__t1);
    }

#ifdef OPS_DEBUG
    ops_register_args(block->instance, args, "KerSumDensity");
#endif

//  =================================================
//  compute locally allocated range for the sub-block
//  =================================================
    int start_indx[2];
    int end_indx[2];
#if defined(OPS_MPI) && !defined(OPS_LAZY)
    int arg_idx[2];
#endif

#if defined(OPS_LAZY) || !defined(OPS_MPI)
    for (int n = 0; n < 2; n++) {
        start_indx[n] = range[2*n];
        end_indx[n]   = range[2*n+1];
    }
#else
    if (compute_ranges(args, 2, block, range, start_indx, end_indx, arg_idx) < 0) return;
#endif

    int start_indx0 = start_indx[0], end_indx0 = end_indx[0];
    int start_indx1 = start_indx[1], end_indx1 = end_indx[1];

//  ======================================================
//  Initialize global variable with the dimensions of dats
//  ======================================================
    int xdim0_KerSumDensity = args[0].dat->size[0];

//  =======================================================
//  Set up initial pointers and exchange halos if necessary
//  =======================================================
    int base0 = args[0].dat->base_offset;
    double * __restrict__ rho_p = (double *)(args[0].data_d + base0);
#ifdef OPS_MPI
    double * __restrict__ p_a1 = (double *)(((ops_reduction)args[1].data)->data + ((ops_reduction)args[1].data)->size * block->index);
#else //OPS_MPI
    double * __restrict__ p_a1 = (double *)((ops_reduction)args[1].data)->data;
#endif //OPS_MPI

//  ==============
//  Halo exchanges
//  ==============
#ifndef OPS_LAZY
    ops_H_D_exchanges_device(args, 2);
    ops_halo_exchanges(args, 2, range);
#endif //OPS_LAZY

    if (block->instance->OPS_diags > 1) {
        ops_timers_core(&__c2, &__t2);
        block->instance->OPS_kernels[3].mpi_time += __t2 - __t1;
    }

    double p_a1_0 = p_a1[0];

    #pragma omp target teams distribute parallel for reduction(+:p_a1_0)
        for (int n_y = start_indx1; n_y < end_indx1; n_y++)
        {
            for (int n_x = start_indx0; n_x < end_indx0; n_x++)
            {

                const  ACC<double> rho(xdim0_KerSumDensity, rho_p + (n_x * 1) + (n_y * xdim0_KerSumDensity * 1));

                double sum[1];
                sum[0] = ZERO_double;

     *sum += rho(0, 0); 

                p_a1_0 += sum[0];

            }
        }

    p_a1[0] = p_a1_0;

    if (block->instance->OPS_diags > 1)
    {
        ops_timers_core(&__c1, &__t1);
        block->instance->OPS_kernels[3].time += __t1 - __t2;
    }

#ifndef OPS_LAZY
    ops_set_dirtybit_device(args, 2);
#endif

    if (block->instance->OPS_diags > 1)
    {
//      ====================
//      Update kernel record
//      ====================
        ops_timers_core(&__c2, &__t2);
        block->instance->OPS_kernels[3].mpi_time += __t2 -__t1;
        block->instance->OPS_kernels[3].transfer += ops_compute_transfer(dim, start_indx, end_indx, &arg0);
    }
}

#ifdef OPS_LAZY
void ops_par_loop_KerSumDensity(
    const char *name,
    ops_block block,
    int dim,
    int *range,
    ops_arg arg0,
    ops_arg arg1
    )
{
    ops_arg args[2];

    args[0] = arg0;
    args[1] = arg1;

    create_kerneldesc_and_enque("KerSumDensity", args, 2, 3, dim, 1, range, block, ops_par_loop_KerSumDensity_execute);
}
#endif
