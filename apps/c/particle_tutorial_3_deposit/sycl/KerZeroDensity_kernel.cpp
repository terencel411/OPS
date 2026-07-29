// Auto-generated at 2026-07-29 00:37:16.542935 by ops-translator


//  ==================
//  Host stub function
//  ==================
#ifndef OPS_LAZY
void ops_par_loop_KerZeroDensity(
    const char *name,
    ops_block block,
    int dim,
    int *range,
    ops_arg arg0
)
{ 
#else
void ops_par_loop_KerZeroDensity_execute(ops_kernel_descriptor *desc)
{
    ops_block block = desc->block;
    int dim = desc->dim;
    int *range = desc->range;
    ops_arg arg0 = desc->args[0];
#endif

//  ======
//  Timing
//  ======
    double __t1, __t2, __c1, __c2;

    ops_arg args[1];

    args[0] = arg0;

#if defined(CHECKPOINTING) && !defined(OPS_LAZY)
    if (!ops_checkpointing_before(args, 1, range, 2)) return;
#endif

    if (block->instance->OPS_diags > 1)
    {
        ops_timing_realloc(block->instance, 2, "KerZeroDensity");
        block->instance->OPS_kernels[2].count++;
        ops_timers_core(&__c1, &__t1);
    }

#ifdef OPS_DEBUG
    ops_register_args(block->instance, args, "KerZeroDensity");
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
    if (compute_ranges(args, 1, block, range, start_indx, end_indx, arg_idx) < 0) return;
#endif

//  ======================================================
//  Initialize global variable with the dimensions of dats
//  ======================================================
    int xdim0_KerZeroDensity = args[0].dat->size[0];

//  =======================================================
//  Set up initial pointers and exchange halos if necessary
//  =======================================================
    int base0 = args[0].dat->base_offset;
    double * __restrict__ rho_p = (double *)(args[0].data_d + base0);

//  =============
//  Halo exchange
//  =============
#ifndef OPS_LAZY
    ops_H_D_exchanges_device(args, 1);
    ops_halo_exchanges(args, 1, range);
#endif

    if (block->instance->OPS_diags > 1) {
        ops_timers_core(&__c2, &__t2);
        block->instance->OPS_kernels[2].mpi_time += __t2 - __t1;
    }

    int start_0 = start_indx[0];
    int end_0 = end_indx[0];
    int start_1 = start_indx[1];
    int end_1 = end_indx[1];

    if ((end_indx[0]-start_indx[0])>0 && (end_indx[1]-start_indx[1])>0) {
        block->instance->sycl_instance->queue->submit([&](cl::sycl::handler &cgh) {

            cgh.parallel_for<class KerZeroDensity_kernel>(cl::sycl::nd_range<2>(cl::sycl::range<2>(
                ((end_indx[1]-start_indx[1]-1)/block->instance->OPS_block_size_y+1)*block->instance->OPS_block_size_y,
                ((end_indx[0]-start_indx[0]-1)/block->instance->OPS_block_size_x+1)*block->instance->OPS_block_size_x)
                , cl::sycl::range<2>(
                    block->instance->OPS_block_size_y,
                    block->instance->OPS_block_size_x)
            )
            , [=](cl::sycl::nd_item<2> item
            ) [[intel::kernel_args_restrict]] {

                int n_y = item.get_global_id()[0]+start_1;
                int n_x = item.get_global_id()[1]+start_0;

                 ACC<double> rho(xdim0_KerZeroDensity, rho_p + (n_x * 1) + (n_y * xdim0_KerZeroDensity * 1));

// =========
// User code
// =========
                if (n_x < end_0 && n_y < end_1) {

                     rho(0, 0) = 0.0; 
                }

            });
        });
    }

    if (block->instance->OPS_diags > 1) {
        block->instance->sycl_instance->queue->wait();
        ops_timers_core(&__c1, &__t1);
        block->instance->OPS_kernels[2].time += __t1 - __t2;
    }

#ifndef OPS_LAZY
    ops_set_dirtybit_device(args, 1);
    ops_set_halo_dirtybit3(&args[0], range);
#endif

    if (block->instance->OPS_diags > 1) {
//      ====================
//      Update kernel record
//      ====================
        ops_timers_core(&__c2, &__t2);
        block->instance->OPS_kernels[2].mpi_time += __t2 -__t1;
        block->instance->OPS_kernels[2].transfer += ops_compute_transfer(dim, start_indx, end_indx, &arg0);
    }
}

#ifdef OPS_LAZY
void ops_par_loop_KerZeroDensity(
    const char *name,
    ops_block block,
    int dim,
    int *range,
    ops_arg arg0
    )
{
    ops_arg args[1];

    args[0] = arg0;

    create_kerneldesc_and_enque("KerZeroDensity", args, 1, 2, dim, 1, range, block, ops_par_loop_KerZeroDensity_execute);
}
#endif
