#ifndef ACC_UTILITIES_H_
#define ACC_UTILITIES_H_

// #include "src/acc/acc_ptr.h"
// #include "src/acc/data_types.h"
// #include "src/error.h"
// #ifdef _CUDA_ENABLED
// #include "src/acc/cuda/cuda_kernels/helper.cuh"
// #include "src/acc/cuda/cuda_kernels/wavg.cuh"
// #include "src/acc/cuda/cuda_kernels/diff2.cuh"
// #include "src/acc/cuda/cuda_fft.h"
// #else
// #include "src/acc/cpu/cpu_kernels/helper.h"
// #include "src/acc/cpu/cpu_kernels/wavg.h"
// #include "src/acc/cpu/cpu_kernels/diff2.h"
// #endif

#include "./diff2.cuh"
#include "./common.h"

namespace AccUtilities
{


template<bool REF3D, bool DATA3D, int block_sz, int eulers_per_block, int prefetch_fraction>
void diff2_coarse(
		unsigned long grid_size,
		int block_size,
		XFLOAT *g_eulers,
		XFLOAT *trans_x,
		XFLOAT *trans_y,
		XFLOAT *trans_z,
		XFLOAT *g_real,
		XFLOAT *g_imag,
		AccProjectorKernel projector,
		XFLOAT *g_corr,
		XFLOAT *g_diff2s,
		unsigned long translation_num,
		unsigned long image_size,
		cudaStream_t stream
		)
{
// #ifdef _CUDA_ENABLED
cuda_kernel_diff2_coarse<REF3D, DATA3D, block_sz, eulers_per_block, prefetch_fraction>
		<<<grid_size,block_size,0,stream>>>(
			g_eulers,
			trans_x,
			trans_y,
			trans_z,
			g_real,
			g_imag,
			projector,
			g_corr,
			g_diff2s,
			translation_num,
			image_size);
// #else
// 	#if 1
// 		CpuKernels::diff2_coarse<REF3D, DATA3D, block_sz, eulers_per_block, prefetch_fraction>(
// 			grid_size,
// 			g_eulers,
// 			trans_x,
// 			trans_y,
// 			trans_z,
// 			g_real,
// 			g_imag,
// 			projector,
// 			g_corr,
// 			g_diff2s,
// 			translation_num,
// 			image_size
// 		);
// 	#else
// 		if (DATA3D)
// 			CpuKernels::diff2_coarse_3D<eulers_per_block>(
// 			grid_size,
// 			g_eulers,
// 			trans_x,
// 			trans_y,
// 			trans_z,
// 			g_real,
// 			g_imag,
// 			projector,
// 			g_corr,
// 			g_diff2s,
// 			translation_num,
// 			image_size);
// 		else
// 			CpuKernels::diff2_coarse_2D<REF3D, eulers_per_block>(
// 			grid_size,
// 			g_eulers,
// 			trans_x,
// 			trans_y,
// 			trans_z,
// 			g_real,
// 			g_imag,
// 			projector,
// 			g_corr,
// 			g_diff2s,
// 			translation_num,
// 			image_size);
// 	#endif
// #endif
}




template<bool REF3D, bool DATA3D, int block_sz, int eulers_per_block, int prefetch_fraction>
void diff2_coarse_double(
		unsigned long grid_size,
		int block_size,
		XFLOAT *g_eulers,
		XFLOAT *trans_x,
		XFLOAT *trans_y,
		XFLOAT *trans_z,
		XFLOAT *g_real,
		XFLOAT *g_imag,
		AccProjectorKernel projector,
		XFLOAT *g_corr,
		double *g_diff2s,
		unsigned long translation_num,
		unsigned long image_size,
		cudaStream_t stream,
		bool filter_img = false
		)
{
cuda_kernel_diff2_coarse_double<REF3D, DATA3D, block_sz, eulers_per_block, prefetch_fraction>
		<<<grid_size,block_size,0,stream>>>(
			g_eulers,
			trans_x,
			trans_y,
			trans_z,
			g_real,
			g_imag,
			projector,
			g_corr,
			g_diff2s,
			translation_num,
			image_size,
			filter_img);
}

template<bool REF3D, bool DATA3D, int block_sz>
void diff2_CC_coarse(
		unsigned long grid_size,
		int block_size,
		XFLOAT *g_eulers,
		XFLOAT *g_imgs_real,
		XFLOAT *g_imgs_imag,
		XFLOAT *g_trans_x,
		XFLOAT *g_trans_y,
		XFLOAT *g_trans_z,
		AccProjectorKernel projector,
		XFLOAT *g_corr_img,
		XFLOAT *g_diff2s,
		unsigned long translation_num,
		unsigned long image_size,
		XFLOAT exp_local_sqrtXi2,
		cudaStream_t stream
		)
{
// #ifdef _CUDA_ENABLED
dim3 CCblocks(grid_size,translation_num);
	cuda_kernel_diff2_CC_coarse<REF3D,DATA3D,block_sz>
		<<<CCblocks,block_size,0,stream>>>(
			g_eulers,
			g_imgs_real,
			g_imgs_imag,
			g_trans_x,
			g_trans_y,
			g_trans_z,
			projector,
			g_corr_img,
			g_diff2s,
			translation_num,
			image_size,
			exp_local_sqrtXi2);
// #else
// 	if (DATA3D)
// 		CpuKernels::diff2_CC_coarse_3D(
// 			grid_size,
// 			g_eulers,
// 			g_imgs_real,
// 			g_imgs_imag,
// 			g_trans_x,
// 			g_trans_y,
// 			g_trans_z,
// 			projector,
// 			g_corr_img,
// 			g_diff2s,
// 			translation_num,
// 			image_size,
// 			exp_local_sqrtXi2);
// 	else
// 		CpuKernels::diff2_CC_coarse_2D<REF3D>(
// 			grid_size,
// 			g_eulers,
// 			g_imgs_real,
// 			g_imgs_imag,
// 			g_trans_x,
// 			g_trans_y,
// 			projector,
// 			g_corr_img,
// 			g_diff2s,
// 			translation_num,
// 			image_size,
// 			exp_local_sqrtXi2);
// #endif
}

// template<bool REF3D, bool DATA3D, int block_sz, int chunk_sz>
// void diff2_fine(
// 		unsigned long grid_size,
// 		int block_size,
// 		XFLOAT *g_eulers,
// 		XFLOAT *g_imgs_real,
// 		XFLOAT *g_imgs_imag,
// 		XFLOAT *trans_x,
// 		XFLOAT *trans_y,
// 		XFLOAT *trans_z,
// 		AccProjectorKernel projector,
// 		XFLOAT *g_corr_img,
// 		XFLOAT *g_diff2s,
// 		unsigned long image_size,
// 		XFLOAT sum_init,
// 		unsigned long orientation_num,
// 		unsigned long translation_num,
// 		unsigned long todo_blocks,
// 		unsigned long *d_rot_idx,
// 		unsigned long *d_trans_idx,
// 		unsigned long *d_job_idx,
// 		unsigned long *d_job_num,
// 		cudaStream_t stream
// 		)
// {
// #ifdef _CUDA_ENABLED
// dim3 block_dim = grid_size;
// 		cuda_kernel_diff2_fine<REF3D,DATA3D, block_sz, chunk_sz>
// 				<<<block_dim,block_size,0,stream>>>(
// 					g_eulers,
// 					g_imgs_real,
// 					g_imgs_imag,
// 					trans_x,
// 					trans_y,
// 					trans_z,
// 					projector,
// 					g_corr_img,    // in these non-CC kernels this is effectively an adjusted MinvSigma2
// 					g_diff2s,
// 					image_size,
// 					sum_init,
// 					orientation_num,
// 					translation_num,
// 					todo_blocks, //significant_num,
// 					d_rot_idx,
// 					d_trans_idx,
// 					d_job_idx,
// 					d_job_num);
// #else
// 		// TODO - make use of orientation_num, translation_num,todo_blocks on
// 		// CPU side if CUDA starts to use
// 	if (DATA3D)
// 		CpuKernels::diff2_fine_3D(
// 			grid_size,
// 			g_eulers,
// 			g_imgs_real,
// 			g_imgs_imag,
// 			trans_x,
// 			trans_y,
// 			trans_z,
// 			projector,
// 			g_corr_img,    // in these non-CC kernels this is effectively an adjusted MinvSigma2
// 			g_diff2s,
// 			image_size,
// 			sum_init,
// 			orientation_num,
// 			translation_num,
// 			todo_blocks, //significant_num,
// 			d_rot_idx,
// 			d_trans_idx,
// 			d_job_idx,
// 			d_job_num);
// 	else
// 		CpuKernels::diff2_fine_2D<REF3D>(
// 			grid_size,
// 			g_eulers,
// 			g_imgs_real,
// 			g_imgs_imag,
// 			trans_x,
// 			trans_y,
// 			trans_z,
// 			projector,
// 			g_corr_img,    // in these non-CC kernels this is effectively an adjusted MinvSigma2
// 			g_diff2s,
// 			image_size,
// 			sum_init,
// 			orientation_num,
// 			translation_num,
// 			todo_blocks, //significant_num,
// 			d_rot_idx,
// 			d_trans_idx,
// 			d_job_idx,
// 			d_job_num);
// #endif
// }

// template<bool REF3D, bool DATA3D, int block_sz,int chunk_sz>
// void diff2_CC_fine(
// 		unsigned long grid_size,
// 		int block_size,
// 		XFLOAT *g_eulers,
// 		XFLOAT *g_imgs_real,
// 		XFLOAT *g_imgs_imag,
// 		XFLOAT *g_trans_x,
// 		XFLOAT *g_trans_y,
// 		XFLOAT *g_trans_z,
// 		AccProjectorKernel &projector,
// 		XFLOAT *g_corr_img,
// 		XFLOAT *g_diff2s,
// 		unsigned long image_size,
// 		XFLOAT sum_init,
// 		XFLOAT exp_local_sqrtXi2,
// 		unsigned long orientation_num,
// 		unsigned long translation_num,
// 		unsigned long todo_blocks,
// 		unsigned long *d_rot_idx,
// 		unsigned long *d_trans_idx,
// 		unsigned long *d_job_idx,
// 		unsigned long *d_job_num,
// 		cudaStream_t stream
// 		)
// {
// #ifdef _CUDA_ENABLED
// dim3 block_dim = grid_size;
// 	cuda_kernel_diff2_CC_fine<REF3D,DATA3D,block_sz,chunk_sz>
// 			<<<block_dim,block_size,0,stream>>>(
// 				g_eulers,
// 				g_imgs_real,
// 				g_imgs_imag,
// 				g_trans_x,
// 				g_trans_y,
// 				g_trans_z,
// 				projector,
// 				g_corr_img,
// 				g_diff2s,
// 				image_size,
// 				sum_init,
// 				exp_local_sqrtXi2,
// 				orientation_num,
// 				translation_num,
// 				todo_blocks,
// 				d_rot_idx,
// 				d_trans_idx,
// 				d_job_idx,
// 				d_job_num);
// #else
// 		// TODO - Make use of orientation_num, translation_num, todo_blocks on
// 		// CPU side if CUDA starts to use
// 	if (DATA3D)
// 		CpuKernels::diff2_CC_fine_3D(
// 			grid_size,
// 			g_eulers,
// 			g_imgs_real,
// 			g_imgs_imag,
// 			g_trans_x,
// 			g_trans_y,
// 			g_trans_z,
// 			projector,
// 			g_corr_img,
// 			g_diff2s,
// 			image_size,
// 			sum_init,
// 			exp_local_sqrtXi2,
// 			orientation_num,
// 			translation_num,
// 			todo_blocks,
// 			d_rot_idx,
// 			d_trans_idx,
// 			d_job_idx,
// 			d_job_num);
// 	else
// 		CpuKernels::diff2_CC_fine_2D<REF3D>(
// 			grid_size,
// 			g_eulers,
// 			g_imgs_real,
// 			g_imgs_imag,
// 			g_trans_x,
// 			g_trans_y,
// 			projector,
// 			g_corr_img,
// 			g_diff2s,
// 			image_size,
// 			sum_init,
// 			exp_local_sqrtXi2,
// 			orientation_num,
// 			translation_num,
// 			todo_blocks,
// 			d_rot_idx,
// 			d_trans_idx,
// 			d_job_idx,
// 			d_job_num);
// #endif
// }

// template<typename T>
// void kernel_weights_exponent_coarse(
// 		unsigned long num_classes,
// 		AccPtr<T> &g_pdf_orientation,
// 		AccPtr<bool> &g_pdf_orientation_zeros,
// 		AccPtr<T> &g_pdf_offset,
// 		AccPtr<bool> &g_pdf_offset_zeros,
// 		AccPtr<T> &g_Mweight,
// 		T g_min_diff2,
// 		unsigned long nr_coarse_orient,
// 		unsigned long  nr_coarse_trans)
// {
// 	long int block_num = ceilf( ((double)nr_coarse_orient*nr_coarse_trans*num_classes) / (double)SUMW_BLOCK_SIZE );

// #ifdef _CUDA_ENABLED
// cuda_kernel_weights_exponent_coarse<T>
// 	<<<block_num,SUMW_BLOCK_SIZE,0,g_Mweight.getStream()>>>(
// 			~g_pdf_orientation,
// 			~g_pdf_orientation_zeros,
// 			~g_pdf_offset,
// 			~g_pdf_offset_zeros,
// 			~g_Mweight,
// 			g_min_diff2,
// 			nr_coarse_orient,
// 			nr_coarse_trans,
// 			nr_coarse_orient*nr_coarse_trans*num_classes);
// #else
// 	CpuKernels::weights_exponent_coarse(
// 			~g_pdf_orientation,
// 			~g_pdf_orientation_zeros,
// 			~g_pdf_offset,
// 			~g_pdf_offset_zeros,
// 			~g_Mweight,
// 			g_min_diff2,
// 			nr_coarse_orient,
// 			nr_coarse_trans,
// 			((size_t)nr_coarse_orient)*((size_t)nr_coarse_trans)*((size_t)num_classes));
// #endif
// }

// template<typename T>
// void kernel_exponentiate(
// 		AccPtr<T> &array,
// 		T add)
// {
// 	int blockDim = (int) ceilf( (double)array.getSize() / (double)BLOCK_SIZE );
// #ifdef _CUDA_ENABLED
// cuda_kernel_exponentiate<T>
// 	<<< blockDim,BLOCK_SIZE,0,array.getStream()>>>
// 	(~array, add, array.getSize());
// #else
// 	CpuKernels::exponentiate<T>
// 	(~array, add, array.getSize());
// #endif
// }

// void kernel_exponentiate_weights_fine(	int grid_size,
// 										int block_size,
// 										XFLOAT *g_pdf_orientation,
// 										XFLOAT *g_pdf_offset,
// 										XFLOAT *g_weights,
// 										unsigned long  oversamples_orient,
// 										unsigned long  oversamples_trans,
// 										unsigned long *d_rot_id,
// 										unsigned long *d_trans_idx,
// 										unsigned long *d_job_idx,
// 										unsigned long *d_job_num,
// 										long int job_num,
// 										cudaStream_t stream);

};  // namespace AccUtilities


#endif //ACC_UTILITIES_H_

