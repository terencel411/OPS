// Auto-generated at 2026-07-29 00:37:16.546058 by ops-translator


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

    int maxblocks = (end_indx[0]-start_indx[0]-1)/block->instance->OPS_block_size_x+1;
    maxblocks *= (end_indx[1]-start_indx[1]-1)/block->instance->OPS_block_size_y+1;
    int reduct_bytes = 0;
    size_t reduct_size = 0;

    reduct_bytes += ROUND_UP(maxblocks*1*sizeof(double));
    reduct_size = MAX(reduct_size,1*sizeof(double));

    reallocReductArrays(block->instance, reduct_bytes);
    reduct_bytes = 0;

    arg1.data = block->instance->OPS_reduct_h + reduct_bytes;
    double *arg1_data_d = (double*)(block->instance->OPS_reduct_d + reduct_bytes);
    for (int b = 0; b < maxblocks; b++) {
        for (int d = 0; d < 1; d++)   ((double *)arg1.data)[d+b*1] = ZERO_double;
    }
    reduct_bytes += ROUND_UP(maxblocks*1*sizeof(double));

    mvReductArraysToDevice(block->instance, reduct_bytes);

//  =============
//  Halo exchange
//  =============
#ifndef OPS_LAZY
    ops_H_D_exchanges_device(args, 2);
    ops_halo_exchanges(args, 2, range);
#endif

    if (block->instance->OPS_diags > 1) {
        ops_timers_core(&__c2, &__t2);
        block->instance->OPS_kernels[3].mpi_time += __t2 - __t1;
    }

    int start_0 = start_indx[0];
    int end_0 = end_indx[0];
    int start_1 = start_indx[1];
    int end_1 = end_indx[1];

    if ((end_indx[0]-start_indx[0])>0 && (end_indx[1]-start_indx[1])>0) {
        block->instance->sycl_instance->queue->submit([&](cl::sycl::handler &cgh) {

            cl::sycl::accessor<char, 1, cl::sycl::access::mode::read_write, 
            cl::sycl::access::target::local> local_mem(reduct_size * cl::sycl::range<1>(block->instance->OPS_block_size_x * block->instance->OPS_block_size_y), cgh);

            cgh.parallel_for<class KerSumDensity_kernel>(cl::sycl::nd_range<2>(cl::sycl::range<2>(
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

                const  ACC<double> rho(xdim0_KerSumDensity, rho_p + (n_x * 1) + (n_y * xdim0_KerSumDensity * 1));

                double sum[1];
                sum[0] = ZERO_double;

// =========
// User code
// =========
                if (n_x < end_0 && n_y < end_1) {

                     *sum += rho(0, 0); 
                }

                int group_size = item.get_local_range(0);
                    group_size *= item.get_local_range(1);
                for (int d = 0; d < 1; d++) {
                    ops_reduction_sycl<OPS_INC>(arg1_data_d + d + item.get_group_linear_id()*1, sum[d], (double*)&local_mem[0], item, group_size);
                }
            });
        });
    }

//  ==============================
//  Reduction across blocks
//  ==============================
    mvReductArraysToHost(block->instance, reduct_bytes);

    for (int b = 0; b < maxblocks; b++)
        for (int d = 0; d < 1; d++)
            p_a1[d] = p_a1[d] + ((double *)arg1.data)[d+b*1];

    if (block->instance->OPS_diags > 1) {
        block->instance->sycl_instance->queue->wait();
        ops_timers_core(&__c1, &__t1);
        block->instance->OPS_kernels[3].time += __t1 - __t2;
    }

#ifndef OPS_LAZY
    ops_set_dirtybit_device(args, 2);
#endif

    if (block->instance->OPS_diags > 1) {
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
