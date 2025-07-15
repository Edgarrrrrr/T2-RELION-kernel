#ifndef CUDA_DIFF2_KERNELS_CUH_
#define CUDA_DIFF2_KERNELS_CUH_

#include <cuda_runtime.h>
#include <vector>
#include <iostream>
#include <fstream>

#include "./acc_projector.h"
#include "./acc_projectorkernel_impl.h"
#include "./cuda_device_utils.cuh"
#include "./mma_utils.cuh"
#include "./coarse_scheduler.cuh"
#include "./reg_bitmap.cuh"
#include "./warp_layout.cuh"
#include "./orientation_matrix_handler.cuh"
#include "./translation_matrix_handler.cuh"

/*
 *   	DIFFERNECE-BASED KERNELS
 */

/*
 * Assuming block_sz % prefetch_fraction == 0 and prefetch_fraction < block_sz
 * Assuming block_sz % eulers_per_block == 0
 * Assuming eulers_per_block * 3 < block_sz
 */

template<typename Ta, typename Tb,  int kBLOCK_SZ>
__global__ void check_array_equal(Ta *a, Tb *b, int size) {
	int tid = threadIdx.x;
	assert(blockIdx.y == 0 && blockIdx.z == 0 && threadIdx.y == 0 && threadIdx.z == 0);
	assert(blockIdx.x == 0);

	bool equal = true;

	for (int i = tid; i < size; i += kBLOCK_SZ) {
		// if (a[i] != b[i]) {
		if (a[i] == 0 && b[i] == 0) continue;
		if (fabs((double)a[i] - (double)b[i]) >= 9e-6 * fabs((double)b[i])) {
			// if (tid == 0)
				printf("!!!ERR!!! a[%d] = %f, b[%d] = %f, epsilon = %e\n", i, (double)a[i], i, (double)b[i], fabs((double)a[i] - (double)b[i]) / fabs((double)b[i]));
			// assert(false);
			equal = false;
		}
	}

	__syncthreads();
	// assert(equal);
}

template<int kBLOCK_SZ>
__global__ void cuda_kernel_coarse_setup_trans_matrix(
		XFLOAT *trans_x,
		XFLOAT *trans_y,
		XFLOAT *trans_z,
		XFLOAT *g_real,
		XFLOAT *g_imag,
		AccProjectorKernel projector,
		XFLOAT *g_trans_real_m,       // translation_num * image_size matrix, column-major
		XFLOAT *g_trans_imag_m,  
		XFLOAT *g_corr,     
		int translation_num,
		int image_size) {
	int tid = threadIdx.x;     // tid is the index of pixel
	int block_id = blockIdx.x; // block_id is the index of translation
	int trans_idx = block_id;

	assert(blockIdx.y == 0 && blockIdx.z == 0 && threadIdx.y == 0 && threadIdx.z == 0);

	for (int i = tid; i < image_size; i += kBLOCK_SZ) {
		int x = i % projector.imgX;
		int y = floorfracf(i, projector.imgX);
		if (y > projector.maxR) {
			y -= projector.imgY;
		}
		XFLOAT real = g_real[i] * sqrt(g_corr[i]) / sqrt(2.);
		XFLOAT imag = g_imag[i] * sqrt(g_corr[i]) / sqrt(2.);

		XFLOAT trans_real, trans_imag;

		XFLOAT tx = trans_x[trans_idx];
		XFLOAT ty = trans_y[trans_idx];
		translatePixel(x, y, tx, ty, real, imag, trans_real, trans_imag);
		int matrix_index = trans_idx + i * translation_num;

		g_trans_real_m[matrix_index] = trans_real;
		g_trans_imag_m[matrix_index] = trans_imag;
	}
}

template<int kBLOCK_SZ>
__global__ void cuda_kernel_coarse_setup_orient_matrix(
	XFLOAT *g_eulers,
	AccProjectorKernel projector,
	XFLOAT *g_orient_real_m,       // image_size * orientation_num matrix, column-major
	XFLOAT *g_orient_imag_m,
	XFLOAT *g_corr,
	int orientation_num,
	int image_size) {
	
	int tid = threadIdx.x;     // tid is the index of pixel
	int block_id = blockIdx.x; // block_id is the index of orientation

	assert(blockIdx.y == 0 && blockIdx.z == 0 && threadIdx.y == 0 && threadIdx.z == 0);
	
	XFLOAT e0 = g_eulers[block_id * 9];
	XFLOAT e1 = g_eulers[block_id * 9 + 1];
	XFLOAT e2 = g_eulers[block_id * 9 + 2];
	XFLOAT e3 = g_eulers[block_id * 9 + 3];
	XFLOAT e4 = g_eulers[block_id * 9 + 4];
	XFLOAT e5 = g_eulers[block_id * 9 + 5];
	XFLOAT e6 = g_eulers[block_id * 9 + 6];
	XFLOAT e7 = g_eulers[block_id * 9 + 7];
	XFLOAT e8 = g_eulers[block_id * 9 + 8];
	
	for (int i = tid; i < image_size; i += kBLOCK_SZ) {
		int x = i % projector.imgX;
		int y = i / projector.imgX;
		if (y > projector.maxR) {
			y -= projector.imgY;
		}

		XFLOAT orient_real, orient_imag;

		projector.project3Dmodel(x, y, e0, e1, e3, e4, e6, e7, orient_real, orient_imag);

		int matrix_index = i + block_id * image_size;
		g_orient_real_m[matrix_index] = orient_real * sqrt(g_corr[i]) / sqrt(2.);
		g_orient_imag_m[matrix_index] = orient_imag * sqrt(g_corr[i]) / sqrt(2.);
	}
}		

template<int kBLOCK_SZ>
__global__ void cuda_kernel_coarse_naive(
		XFLOAT *g_trans_real_m,
		XFLOAT *g_trans_imag_m,
		XFLOAT *g_orient_real_m,
		XFLOAT *g_orient_imag_m,
		XFLOAT *g_corr,
		XFLOAT *g_diff2s,
		int translation_num,
		int orientation_num,
		int image_size) {
	
	int tid = threadIdx.x;
	int block_id = blockIdx.x;

	int trans_idx = block_id % translation_num;
	int orient_idx = block_id / translation_num;

	assert(blockIdx.y == 0 && blockIdx.z == 0 && threadIdx.y == 0 && threadIdx.z == 0);

	XFLOAT diff2 = 0.0;
	XFLOAT diff2_positive = 0.0;
	XFLOAT diff2_negative = 0.0;
	for (int i = tid; i < image_size; i += kBLOCK_SZ) {
		XFLOAT trans_real = g_trans_real_m[i * translation_num + trans_idx];
		XFLOAT trans_imag = g_trans_imag_m[i * translation_num + trans_idx];

		XFLOAT orient_real = g_orient_real_m[i + orient_idx * image_size];
		XFLOAT orient_imag = g_orient_imag_m[i + orient_idx * image_size];
		diff2_positive += (trans_real * trans_real + orient_real * orient_real + trans_imag * trans_imag + orient_imag * orient_imag);
		diff2_negative += (2 * trans_real * orient_real + 2 * trans_imag * orient_imag);
	}
	diff2 = diff2_positive - diff2_negative;

	__shared__ XFLOAT s_diff2[kBLOCK_SZ];
	s_diff2[tid] = diff2;
	__syncthreads();
	for (int j = kBLOCK_SZ / 2; j > 0; j /= 2) {
		if (tid < j) {
			s_diff2[tid] += s_diff2[tid + j];
		}
		__syncthreads();
	}
	if (tid == 0) {
		g_diff2s[block_id] += s_diff2[0];
	}
}

__host__ __device__ inline void pixel_index2coor(int idx, int imgX, int imgY, int maxR, int &x, int &y) {
	x = idx % imgX;
	y = idx / imgX;
	if (y > maxR) {
		y -= imgY;
	}
}

__device__ __forceinline__ XFLOAT warpReduceSum(XFLOAT sum){
    sum += __shfl_down_sync(0xffffffff, sum, 16);
    sum += __shfl_down_sync(0xffffffff, sum, 8);
    sum += __shfl_down_sync(0xffffffff, sum, 4);
    sum += __shfl_down_sync(0xffffffff, sum, 2);
    sum += __shfl_down_sync(0xffffffff, sum, 1);
	sum = __shfl_sync(0xffffffff, sum, 0);
    return sum;
}

template<int kWarpTileM, int kWarpTileN>
__inline__ __device__ int get_warp_tile_offset(int warp_id, int warp_num, int tile_size) {
	return warp_id * tile_size;
}

template<bool REF3D, bool DATA3D, int block_sz, int eulers_per_block, int prefetch_fraction>
__global__ void cuda_kernel_diff2_coarse_double(
		XFLOAT *g_eulers,
		XFLOAT *trans_x,
		XFLOAT *trans_y,
		XFLOAT *trans_z,
		XFLOAT *g_real,
		XFLOAT *g_imag,
		AccProjectorKernel projector,
		XFLOAT *g_corr,
		double *g_diff2s,
		int translation_num,
		int image_size,
		bool filter_img = false
		)
{
	int tid = threadIdx.x;

	//Prefetch euler matrices
	__shared__ double s_eulers[eulers_per_block * 9];

	int max_block_pass_euler( ceilfracf(eulers_per_block*9, block_sz) * block_sz);

	for (int i = tid; i < max_block_pass_euler; i += block_sz)
		if (i < eulers_per_block * 9)
			s_eulers[i] = g_eulers[blockIdx.x * eulers_per_block * 9 + i];


	//Setup variables
	__shared__ double s_ref_real[block_sz/prefetch_fraction * eulers_per_block];
	__shared__ double s_ref_imag[block_sz/prefetch_fraction * eulers_per_block];

	__shared__ double s_real[block_sz];
	__shared__ double s_imag[block_sz];
	__shared__ double s_corr[block_sz];

	double diff2s[eulers_per_block] = {0.f};

	double tx = trans_x[tid%translation_num];
	double ty = trans_y[tid%translation_num];
	// XFLOAT tz = trans_z[tid%translation_num];
	double tz = 0;

	//Step through data
	int max_block_pass_pixel( ceilfracf(image_size,block_sz) * block_sz );
	
	for (int init_pixel = 0; init_pixel < max_block_pass_pixel; init_pixel += block_sz/prefetch_fraction)
	{
		__syncthreads();

		//Prefetch block-fraction-wise
		if(init_pixel + tid/prefetch_fraction < image_size)
		{
			int x,y,z,xy;
			if(DATA3D)
			{
				assert(false);
				z =  floorfracf(init_pixel + tid/prefetch_fraction, projector.imgX*projector.imgY);
				xy = (init_pixel + tid/prefetch_fraction) % (projector.imgX*projector.imgY);
				x =             xy  % projector.imgX;
				y = floorfracf( xy,   projector.imgX);
				if (z > projector.maxR)
					z -= projector.imgZ;
			}
			else
			{
				x =           ( init_pixel + tid/prefetch_fraction) % projector.imgX;
				y = floorfracf( init_pixel + tid/prefetch_fraction  , projector.imgX);
			}
			if (y > projector.maxR)
				y -= projector.imgY;

//			#pragma unroll
			for (int i = tid%prefetch_fraction; i < eulers_per_block; i += prefetch_fraction)
			{
				if(DATA3D) // if DATA3D, then REF3D as well.
					assert(false);
				else if(REF3D)
					projector.project3DmodelDouble(
						x,y,
						s_eulers[i*9  ],
						s_eulers[i*9+1],
						s_eulers[i*9+3],
						s_eulers[i*9+4],
						s_eulers[i*9+6],
						s_eulers[i*9+7],
						s_ref_real[eulers_per_block * (tid/prefetch_fraction) + i],
						s_ref_imag[eulers_per_block * (tid/prefetch_fraction) + i]);
				else
					assert(false);
			}
		}
		//Prefetch block-wise
		if (init_pixel % block_sz == 0 && init_pixel + tid < image_size)
		{
			s_real[tid] = (double)g_real[init_pixel + tid];
			s_imag[tid] = (double)g_imag[init_pixel + tid];
			s_corr[tid] = (double)g_corr[init_pixel + tid] / 2;
		}

		__syncthreads();

		if (tid/translation_num < block_sz/translation_num) // NOTE int division A/B==C/B !=> A==C
		for (int i = tid / translation_num;
				i < block_sz/prefetch_fraction;
				i += block_sz/translation_num)
		{
			if((init_pixel + i) >= image_size) break;

			int x,y,z,xy;
			if(DATA3D)
			{
				z =  floorfracf( init_pixel + i   ,  projector.imgX*projector.imgY); //TODO optimize index extraction.
				xy =           ( init_pixel + i ) % (projector.imgX*projector.imgY);
				x =             xy  % projector.imgX;
				y = floorfracf( xy,   projector.imgX);
				if (z > projector.maxR)
					z -= projector.imgZ;
			}
			else
			{
				x =           ( init_pixel + i ) % projector.imgX;
				y = floorfracf( init_pixel + i   , projector.imgX);
			}
			if (y > projector.maxR)
				y -= projector.imgY;

			double real, imag;

			if(DATA3D)
				translatePixelDouble(x, y, z, tx, ty, tz, s_real[i + init_pixel % block_sz], s_imag[i + init_pixel % block_sz], real, imag);
			else
				translatePixelDouble(x, y,    tx, ty,     s_real[i + init_pixel % block_sz], s_imag[i + init_pixel % block_sz], real, imag);

			if (filter_img) {
				double r2 = (x * x + y * y) * projector.padding_factor * projector.padding_factor;
				if (r2 > projector.maxR2_padded) {
					real = 0.;
					imag = 0.;
				}
			}

			#pragma unroll
			for (int j = 0; j < eulers_per_block; j ++)
			{
				double diff_real =  s_ref_real[eulers_per_block * i + j] - real;
				double diff_imag =  s_ref_imag[eulers_per_block * i + j] - imag;

				double real1 = s_ref_real[eulers_per_block * i + j];
				double imag1 = s_ref_imag[eulers_per_block * i + j];
				
				diff2s[j] += (diff_real * diff_real + diff_imag * diff_imag) * s_corr[i + init_pixel % block_sz];
			}
		}
	}

	//Set global
	#pragma unroll
	for (int i = 0; i < eulers_per_block; i ++)
		atomicAdd(&g_diff2s[(blockIdx.x * eulers_per_block + i) * translation_num + tid % translation_num], diff2s[i]);
	}



template<bool REF3D, bool DATA3D, int block_sz, int eulers_per_block, int prefetch_fraction>
__global__ void cuda_kernel_diff2_coarse(
		XFLOAT *g_eulers,
		XFLOAT *trans_x,
		XFLOAT *trans_y,
		XFLOAT *trans_z,
		XFLOAT *g_real,
		XFLOAT *g_imag,
		AccProjectorKernel projector,
		XFLOAT *g_corr,
		XFLOAT *g_diff2s,
		int translation_num,
		int image_size
		)
{
	int tid = threadIdx.x;

	//Prefetch euler matrices
	__shared__ XFLOAT s_eulers[eulers_per_block * 9];

	int max_block_pass_euler( ceilfracf(eulers_per_block*9, block_sz) * block_sz);

	for (int i = tid; i < max_block_pass_euler; i += block_sz)
		if (i < eulers_per_block * 9)
			s_eulers[i] = g_eulers[blockIdx.x * eulers_per_block * 9 + i];


	//Setup variables
	__shared__ XFLOAT s_ref_real[block_sz/prefetch_fraction * eulers_per_block];
	__shared__ XFLOAT s_ref_imag[block_sz/prefetch_fraction * eulers_per_block];

	__shared__ XFLOAT s_real[block_sz];
	__shared__ XFLOAT s_imag[block_sz];
	__shared__ XFLOAT s_corr[block_sz];

	XFLOAT diff2s[eulers_per_block] = {0.f};

	XFLOAT tx = trans_x[tid%translation_num];
	XFLOAT ty = trans_y[tid%translation_num];
	// XFLOAT tz = trans_z[tid%translation_num];
	XFLOAT tz = 0;

	//Step through data
	int max_block_pass_pixel( ceilfracf(image_size,block_sz) * block_sz );

	for (int init_pixel = 0; init_pixel < max_block_pass_pixel; init_pixel += block_sz/prefetch_fraction)
	{
		__syncthreads();

		//Prefetch block-fraction-wise
		if(init_pixel + tid/prefetch_fraction < image_size)
		{
			int x,y,z,xy;
			if(DATA3D)
			{
				assert(false);
				z =  floorfracf(init_pixel + tid/prefetch_fraction, projector.imgX*projector.imgY);
				xy = (init_pixel + tid/prefetch_fraction) % (projector.imgX*projector.imgY);
				x =             xy  % projector.imgX;
				y = floorfracf( xy,   projector.imgX);
				if (z > projector.maxR)
					z -= projector.imgZ;
			}
			else
			{
				x =           ( init_pixel + tid/prefetch_fraction) % projector.imgX;
				y = floorfracf( init_pixel + tid/prefetch_fraction  , projector.imgX);
			}
			if (y > projector.maxR)
				y -= projector.imgY;

//			#pragma unroll
			for (int i = tid%prefetch_fraction; i < eulers_per_block; i += prefetch_fraction)
			{
				if(DATA3D) // if DATA3D, then REF3D as well.
					projector.project3Dmodel(
						x,y,z,
						s_eulers[i*9  ],
						s_eulers[i*9+1],
						s_eulers[i*9+2],
						s_eulers[i*9+3],
						s_eulers[i*9+4],
						s_eulers[i*9+5],
						s_eulers[i*9+6],
						s_eulers[i*9+7],
						s_eulers[i*9+8],
						s_ref_real[eulers_per_block * (tid/prefetch_fraction) + i],
						s_ref_imag[eulers_per_block * (tid/prefetch_fraction) + i]);
				else if(REF3D)
					projector.project3Dmodel(
						x,y,
						s_eulers[i*9  ],
						s_eulers[i*9+1],
						s_eulers[i*9+3],
						s_eulers[i*9+4],
						s_eulers[i*9+6],
						s_eulers[i*9+7],
						s_ref_real[eulers_per_block * (tid/prefetch_fraction) + i],
						s_ref_imag[eulers_per_block * (tid/prefetch_fraction) + i]);
				else
					projector.project2Dmodel(
						x,y,
						s_eulers[i*9  ],
						s_eulers[i*9+1],
						s_eulers[i*9+3],
						s_eulers[i*9+4],
						s_ref_real[eulers_per_block * (tid/prefetch_fraction) + i],
						s_ref_imag[eulers_per_block * (tid/prefetch_fraction) + i]);
			}
		}

		//Prefetch block-wise
		if (init_pixel % block_sz == 0 && init_pixel + tid < image_size)
		{
			s_real[tid] = g_real[init_pixel + tid];
			s_imag[tid] = g_imag[init_pixel + tid];
			s_corr[tid] = g_corr[init_pixel + tid] / 2;
		}

		__syncthreads();

		if (tid/translation_num < block_sz/translation_num) // NOTE int division A/B==C/B !=> A==C
		for (int i = tid / translation_num;
				i < block_sz/prefetch_fraction;
				i += block_sz/translation_num)
		{
			if((init_pixel + i) >= image_size) break;

			int x,y,z,xy;
			if(DATA3D)
			{
				z =  floorfracf( init_pixel + i   ,  projector.imgX*projector.imgY); //TODO optimize index extraction.
				xy =           ( init_pixel + i ) % (projector.imgX*projector.imgY);
				x =             xy  % projector.imgX;
				y = floorfracf( xy,   projector.imgX);
				if (z > projector.maxR)
					z -= projector.imgZ;
			}
			else
			{
				x =           ( init_pixel + i ) % projector.imgX;
				y = floorfracf( init_pixel + i   , projector.imgX);
			}
			if (y > projector.maxR)
				y -= projector.imgY;

			XFLOAT real, imag;

			if(DATA3D)
				translatePixel(x, y, z, tx, ty, tz, s_real[i + init_pixel % block_sz], s_imag[i + init_pixel % block_sz], real, imag);
			else
				translatePixel(x, y,    tx, ty,     s_real[i + init_pixel % block_sz], s_imag[i + init_pixel % block_sz], real, imag);


			#pragma unroll
			for (int j = 0; j < eulers_per_block; j ++)
			{
				XFLOAT diff_real =  s_ref_real[eulers_per_block * i + j] - real;
				XFLOAT diff_imag =  s_ref_imag[eulers_per_block * i + j] - imag;
				diff2s[j] += (diff_real * diff_real + diff_imag * diff_imag) * s_corr[i + init_pixel % block_sz];
			}
		}
	}

	//Set global
	#pragma unroll
	for (int i = 0; i < eulers_per_block; i ++)
		cuda_atomic_add(&g_diff2s[(blockIdx.x * eulers_per_block + i) * translation_num + tid % translation_num], diff2s[i]);
}


template<bool REF3D, bool DATA3D, int block_sz, int chunk_sz>
__global__ void cuda_kernel_diff2_fine(
		XFLOAT *g_eulers,
		XFLOAT *g_imgs_real,
		XFLOAT *g_imgs_imag,
		XFLOAT *trans_x,
		XFLOAT *trans_y,
		XFLOAT *trans_z,
		AccProjectorKernel projector,
		XFLOAT *g_corr_img,
		XFLOAT *g_diff2s,
		unsigned image_size,
		XFLOAT sum_init,
		unsigned long orientation_num,
		unsigned long translation_num,
		unsigned long todo_blocks,
		unsigned long *d_rot_idx,
		unsigned long *d_trans_idx,
		unsigned long *d_job_idx,
		unsigned long *d_job_num
		)
{
	unsigned long bid = blockIdx.x;
	unsigned long tid = threadIdx.x;

//    // Specialize BlockReduce for a 1D block of 128 threads on type XFLOAT
//    typedef cub::BlockReduce<XFLOAT, 128> BlockReduce;
//    // Allocate shared memory for BlockReduce
//    __shared__ typename BlockReduce::TempStorage temp_storage;

	unsigned long pixel;
	XFLOAT ref_real, ref_imag,
		shifted_real, shifted_imag,
		diff_real, diff_imag;

	__shared__ XFLOAT s[block_sz*chunk_sz]; //We MAY have to do up to chunk_sz translations in each block
	__shared__ XFLOAT s_outs[chunk_sz];
	// inside the padded 2D orientation gri
//	if( bid < todo_blocks ) // we only need to make
	{
		unsigned trans_num  = (unsigned)d_job_num[bid]; //how many transes we have for this rot
		for (int itrans=0; itrans<trans_num; itrans++)
		{
			s[itrans*block_sz+tid] = (XFLOAT)0.0;
		}
		// index of comparison
		unsigned long int ix = d_rot_idx[d_job_idx[bid]];
		unsigned long int iy;
		unsigned pass_num(ceilfracf(image_size,block_sz));

		for (unsigned pass = 0; pass < pass_num; pass++) // finish an entire ref image each block
		{
			pixel = (pass * block_sz) + tid;

			if(pixel < image_size)
			{
				int x,y,z,xy;
				if(DATA3D)
				{
					z =  floorfracf(pixel, projector.imgX*projector.imgY);
					xy = pixel % (projector.imgX*projector.imgY);
					x =             xy  % projector.imgX;
					y = floorfracf( xy,   projector.imgX);
					if (z > projector.maxR)
					{
						if (z >= projector.imgZ - projector.maxR)
							z = z - projector.imgZ;
						else
							x = projector.maxR;
					}
				}
				else
				{
					x =             pixel % projector.imgX;
					y = floorfracf( pixel , projector.imgX);
				}
				if (y > projector.maxR)
				{
					if (y >= projector.imgY - projector.maxR)
						y = y - projector.imgY;
					else
						x = projector.maxR;
				}

				if(DATA3D)
					projector.project3Dmodel(
						x,y,z,
						__ldg(&g_eulers[ix*9  ]), __ldg(&g_eulers[ix*9+1]), __ldg(&g_eulers[ix*9+2]),
						__ldg(&g_eulers[ix*9+3]), __ldg(&g_eulers[ix*9+4]), __ldg(&g_eulers[ix*9+5]),
						__ldg(&g_eulers[ix*9+6]), __ldg(&g_eulers[ix*9+7]), __ldg(&g_eulers[ix*9+8]),
						ref_real, ref_imag);
				else if(REF3D)
					projector.project3Dmodel(
						x,y,
						__ldg(&g_eulers[ix*9  ]), __ldg(&g_eulers[ix*9+1]),
						__ldg(&g_eulers[ix*9+3]), __ldg(&g_eulers[ix*9+4]),
						__ldg(&g_eulers[ix*9+6]), __ldg(&g_eulers[ix*9+7]),
						ref_real, ref_imag);
				else
					projector.project2Dmodel(
						x,y,
						__ldg(&g_eulers[ix*9  ]), __ldg(&g_eulers[ix*9+1]),
						__ldg(&g_eulers[ix*9+3]), __ldg(&g_eulers[ix*9+4]),
						ref_real, ref_imag);

				for (int itrans=0; itrans<trans_num; itrans++) // finish all translations in each partial pass
				{
					iy = d_trans_idx[d_job_idx[bid]] + itrans;

					if(DATA3D)
						translatePixel(x, y, z, trans_x[iy], trans_y[iy], trans_z[iy], g_imgs_real[pixel], g_imgs_imag[pixel], shifted_real, shifted_imag);
					else
						translatePixel(x, y, trans_x[iy], trans_y[iy], g_imgs_real[pixel], g_imgs_imag[pixel], shifted_real, shifted_imag);

					diff_real =  ref_real - shifted_real;
					diff_imag =  ref_imag - shifted_imag;
					s[itrans*block_sz + tid] += (diff_real * diff_real + diff_imag * diff_imag) * (XFLOAT)0.5 * __ldg(&g_corr_img[pixel]);
				}
			}
			__syncthreads();
		}
		for(int j=(block_sz/2); j>0; j/=2)
		{
			if(tid<j)
			{
				for (int itrans=0; itrans<trans_num; itrans++) // finish all translations in each partial pass
				{
					s[itrans*block_sz+tid] += s[itrans*block_sz+tid+j];
				}
			}
			__syncthreads();
		}
		if (tid < trans_num)
		{
			s_outs[tid]=s[tid*block_sz]+sum_init;
		}
		if (tid < trans_num)
		{
			iy=d_job_idx[bid]+tid;
			g_diff2s[iy] = s_outs[tid];
		}
	}
}




/*
 *   	CROSS-CORRELATION-BASED KERNELS
 */

template<bool REF3D, bool DATA3D, int block_sz>
__global__ void cuda_kernel_diff2_CC_coarse(
		XFLOAT *g_eulers,
		XFLOAT *g_imgs_real,
		XFLOAT *g_imgs_imag,
		XFLOAT *g_trans_x,
		XFLOAT *g_trans_y,
		XFLOAT *g_trans_z,
		AccProjectorKernel projector,
		XFLOAT *g_corr_img,
		XFLOAT *g_diff2s,
		unsigned translation_num,
		int image_size,
		XFLOAT exp_local_sqrtXi2
		)
{

	int iorient = blockIdx.x;
	int itrans =  blockIdx.y;
	int tid = threadIdx.x;

    __shared__ XFLOAT s_weight[block_sz];
    s_weight[tid] = (XFLOAT)0.0;
	__shared__ XFLOAT s_norm[block_sz];
	s_norm[tid] = (XFLOAT)0.0;

	XFLOAT real, imag, ref_real, ref_imag;

	XFLOAT e0,e1,e2,e3,e4,e5,e6,e7,e8;
	e0 = __ldg(&g_eulers[iorient*9  ]);
	e1 = __ldg(&g_eulers[iorient*9+1]);
	e2 = __ldg(&g_eulers[iorient*9+2]);
	e3 = __ldg(&g_eulers[iorient*9+3]);
	e4 = __ldg(&g_eulers[iorient*9+4]);
	e5 = __ldg(&g_eulers[iorient*9+5]);
	e6 = __ldg(&g_eulers[iorient*9+6]);
	e7 = __ldg(&g_eulers[iorient*9+7]);
	e8 = __ldg(&g_eulers[iorient*9+8]);

	__syncthreads();

	unsigned pixel_pass_num( ceilfracf(image_size,block_sz) );
	for (unsigned pass = 0; pass < pixel_pass_num; pass++)
	{
		unsigned pixel = (pass * block_sz) + tid;

		if(pixel < image_size)
		{
			int x,y,z,xy;
			if(DATA3D)
			{
				z =  floorfracf(pixel, projector.imgX*projector.imgY);
				xy = pixel % (projector.imgX*projector.imgY);
				x =             xy  % projector.imgX;
				y = floorfracf( xy,   projector.imgX);
				if (z > projector.maxR)
				{
					if (z >= projector.imgZ - projector.maxR)
						z = z - projector.imgZ;
					else
						x = projector.maxR;
				}
			}
			else
			{
				x =             pixel % projector.imgX;
				y = floorfracf( pixel , projector.imgX);
			}
			if (y > projector.maxR)
			{
				if (y >= projector.imgY - projector.maxR)
					y = y - projector.imgY;
				else
					x = projector.maxR;
			}

			if(DATA3D)
				projector.project3Dmodel(
					x,y,z,
					e0,e1,e2,e3,e4,e5,e6,e7,e8,
					ref_real, ref_imag);
			else if(REF3D)
				projector.project3Dmodel(
					x,y,
					e0,e1,e3,e4,e6,e7,
					ref_real, ref_imag);
			else
				projector.project2Dmodel(
					x,y,
					e0,e1,e3,e4,
					ref_real, ref_imag);

			if(DATA3D)
				translatePixel(x, y, z, g_trans_x[itrans], g_trans_y[itrans], g_trans_z[itrans], g_imgs_real[pixel], g_imgs_imag[pixel], real, imag);
			else
				translatePixel(x, y,    g_trans_x[itrans], g_trans_y[itrans],                    g_imgs_real[pixel], g_imgs_imag[pixel], real, imag);

			s_weight[tid] += (ref_real * real     + ref_imag * imag)      * __ldg(&g_corr_img[pixel]);
			s_norm[tid]   += (ref_real * ref_real + ref_imag * ref_imag ) * __ldg(&g_corr_img[pixel]);
		}
		__syncthreads();
	}


	for(int j=(block_sz/2); j>0; j/=2)
	{
		if(tid<j)
		{
			s_weight[tid] += s_weight[tid+j];
			s_norm[tid]   += s_norm[tid+j];
		}
		__syncthreads();
	}
#ifdef ACC_DOUBLE_PRECISION
	g_diff2s[iorient * translation_num + itrans] = - ( s_weight[0] / sqrt(s_norm[0]));
#else
	g_diff2s[iorient * translation_num + itrans] = - ( s_weight[0] / sqrtf(s_norm[0]));
#endif
}

template<bool REF3D, bool DATA3D, int block_sz,int chunk_sz>
__global__ void cuda_kernel_diff2_CC_fine(
		XFLOAT *g_eulers,
		XFLOAT *g_imgs_real,
		XFLOAT *g_imgs_imag,
		XFLOAT *g_trans_x,
		XFLOAT *g_trans_y,
		XFLOAT *g_trans_z,
		AccProjectorKernel projector,
		XFLOAT *g_corr_img,
		XFLOAT *g_diff2s,
		unsigned image_size,
		XFLOAT sum_init,
		XFLOAT exp_local_sqrtXi2,
		unsigned long orientation_num,
		unsigned long translation_num,
		unsigned long todo_blocks,
		unsigned long *d_rot_idx,
		unsigned long *d_trans_idx,
		unsigned long *d_job_idx,
		unsigned long *d_job_num
		)
{
	int bid = blockIdx.y * gridDim.x + blockIdx.x;
	int tid = threadIdx.x;

//    // Specialize BlockReduce for a 1D block of 128 threads on type XFLOAT
//    typedef cub::BlockReduce<XFLOAT, 128> BlockReduce;
//    // Allocate shared memory for BlockReduce
//    __shared__ typename BlockReduce::TempStorage temp_storage;

	int pixel;
	XFLOAT ref_real, ref_imag, shifted_real, shifted_imag;

	__shared__ XFLOAT      s[block_sz*chunk_sz]; //We MAY have to do up to chunk_sz translations in each block
	__shared__ XFLOAT   s_cc[block_sz*chunk_sz];
	__shared__ XFLOAT s_outs[chunk_sz];

	if( bid < todo_blocks ) // we only need to make
	{
		unsigned trans_num   = d_job_num[bid]; //how many transes we have for this rot
		for (int itrans=0; itrans<trans_num; itrans++)
		{
			s[   itrans*block_sz+tid] = 0.0f;
			s_cc[itrans*block_sz+tid] = 0.0f;
		}
		__syncthreads();
		// index of comparison
		unsigned long int ix = d_rot_idx[d_job_idx[bid]];
		unsigned long int iy;
		unsigned pass_num( ceilfracf(image_size,block_sz) );

		for (unsigned pass = 0; pass < pass_num; pass++) // finish an entire ref image each block
		{
			pixel = (pass * block_sz) + tid;

			if(pixel < image_size)
			{
				int x,y,z,xy;
				if(DATA3D)
				{
					z =  floorfracf(pixel, projector.imgX*projector.imgY);
					xy = pixel % (projector.imgX*projector.imgY);
					x =             xy  % projector.imgX;
					y = floorfracf( xy,   projector.imgX);
					if (z > projector.maxR)
					{
						if (z >= projector.imgZ - projector.maxR)
							z = z - projector.imgZ;
						else
							x = projector.maxR;
					}
				}
				else
				{
					x =             pixel % projector.imgX;
					y = floorfracf( pixel , projector.imgX);
				}

				if (y > projector.maxR)
				{
					if (y >= projector.imgY - projector.maxR)
						y = y - projector.imgY;
					else
						x = projector.maxR;
				}

				if(DATA3D)
					projector.project3Dmodel(
						x,y,z,
						__ldg(&g_eulers[ix*9  ]), __ldg(&g_eulers[ix*9+1]), __ldg(&g_eulers[ix*9+2]),
						__ldg(&g_eulers[ix*9+3]), __ldg(&g_eulers[ix*9+4]), __ldg(&g_eulers[ix*9+5]),
						__ldg(&g_eulers[ix*9+6]), __ldg(&g_eulers[ix*9+7]), __ldg(&g_eulers[ix*9+8]),
						ref_real, ref_imag);
				else if(REF3D)
					projector.project3Dmodel(
						x,y,
						__ldg(&g_eulers[ix*9  ]), __ldg(&g_eulers[ix*9+1]),
						__ldg(&g_eulers[ix*9+3]), __ldg(&g_eulers[ix*9+4]),
						__ldg(&g_eulers[ix*9+6]), __ldg(&g_eulers[ix*9+7]),
						ref_real, ref_imag);
				else
					projector.project2Dmodel(
						x,y,
						__ldg(&g_eulers[ix*9  ]), __ldg(&g_eulers[ix*9+1]),
						__ldg(&g_eulers[ix*9+3]), __ldg(&g_eulers[ix*9+4]),
						ref_real, ref_imag);

				for (int itrans=0; itrans<trans_num; itrans++) // finish all translations in each partial pass
				{
					iy = d_trans_idx[d_job_idx[bid]] + itrans;

					if(DATA3D)
						translatePixel(x, y, z, g_trans_x[iy], g_trans_y[iy], g_trans_z[iy], g_imgs_real[pixel], g_imgs_imag[pixel], shifted_real, shifted_imag);
					else
						translatePixel(x, y,    g_trans_x[iy], g_trans_y[iy],                g_imgs_real[pixel], g_imgs_imag[pixel], shifted_real, shifted_imag);

					s[   itrans*block_sz + tid] += (ref_real * shifted_real + ref_imag * shifted_imag) * __ldg(&g_corr_img[pixel]);
					s_cc[itrans*block_sz + tid] += (ref_real*ref_real + ref_imag*ref_imag) * __ldg(&g_corr_img[pixel]);
				}
			}
			__syncthreads();
		}
		for(int j=(block_sz/2); j>0; j/=2)
		{
			if(tid<j)
			{
				for (int itrans=0; itrans<trans_num; itrans++) // finish all translations in each partial pass
				{
					s[   itrans*block_sz+tid] += s[   itrans*block_sz+tid+j];
					s_cc[itrans*block_sz+tid] += s_cc[itrans*block_sz+tid+j];
				}
			}
			__syncthreads();
		}
		if (tid < trans_num)
		{
#ifdef ACC_DOUBLE_PRECISION
			s_outs[tid]= - s[tid*block_sz] / (sqrt(s_cc[tid*block_sz]));
#else
			s_outs[tid]= - s[tid*block_sz] / (sqrtf(s_cc[tid*block_sz]));
#endif
		}
		if (tid < trans_num)
		{
			iy=d_job_idx[bid]+tid;
			g_diff2s[iy] = s_outs[tid];
		}
	}
}
#endif /* CUDA_DIFF2_KERNELS_CUH_ */
