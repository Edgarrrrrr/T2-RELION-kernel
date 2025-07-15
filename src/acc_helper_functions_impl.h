/*
#undef ALTCPU
#include <cuda_runtime.h>
#include "src/acc/cuda/cuda_settings.h"
#include "src/acc/cuda/cuda_kernels/BP.cuh"
#include "src/macros.h"
#include "src/error.h"
*/
#ifndef _ACC_HELPER_FUNCTIONS_IMPL_H_
#define _ACC_HELPER_FUNCTIONS_IMPL_H_

#include "./common.h"
#include "./utilities.h"

void runDiff2KernelCoarse(
		AccProjectorKernel &projector,
		XFLOAT *trans_x,
		XFLOAT *trans_y,
		XFLOAT *trans_z,
		XFLOAT *corr_img,
		XFLOAT *Fimg_real,
		XFLOAT *Fimg_imag,
		XFLOAT *d_eulers,
		XFLOAT *diff2s,
		XFLOAT local_sqrtXi2,
		long unsigned orientation_num,
		long unsigned translation_num,
		long unsigned image_size,
		cudaStream_t stream,
		bool do_CC,
		bool data_is_3D)//内部可能多次调用kernel
{
	const long unsigned blocks3D = (data_is_3D? D2C_BLOCK_SIZE_DATA3D : D2C_BLOCK_SIZE_REF3D);
	// std::cout<<"xjl debug"<<blocks3D<< " "<<orientation_num<<std::endl;
	if(!do_CC)
	{
		if(projector.mdlZ!=0)
		{

// #ifdef ACC_DOUBLE_PRECISION
// 			if (translation_num > blocks3D*4)
// 				CRITICAL(ERR_TRANSLIM);
// #else
// 			if (translation_num > blocks3D*8)
// 				CRITICAL(ERR_TRANSLIM);
// #endif

			long unsigned rest = orientation_num % blocks3D;
			long unsigned even_orientation_num = orientation_num - rest;
// TODO - find a more compact way to represent these combinations resulting in
// a single call to diff2_course?
			if (translation_num <= blocks3D)
			{
				if (even_orientation_num != 0)
				{
					if(data_is_3D)
						AccUtilities::diff2_coarse<true,true, D2C_BLOCK_SIZE_DATA3D, D2C_EULERS_PER_BLOCK_DATA3D, 4>(
							even_orientation_num/(unsigned long)D2C_EULERS_PER_BLOCK_DATA3D,
							D2C_BLOCK_SIZE_DATA3D,
							d_eulers,
							trans_x,
							trans_y,
							trans_z,
							Fimg_real,
							Fimg_imag,
							projector,
							corr_img,
							diff2s,
							translation_num,
							image_size,
							stream);
					else
						AccUtilities::diff2_coarse<true,false, D2C_BLOCK_SIZE_REF3D, D2C_EULERS_PER_BLOCK_REF3D, 4>(
							even_orientation_num/(unsigned long)D2C_EULERS_PER_BLOCK_REF3D,
							D2C_BLOCK_SIZE_REF3D,
							d_eulers,
							trans_x,
							trans_y,
							trans_z,
							Fimg_real,
							Fimg_imag,
							projector,
							corr_img,
							diff2s,
							translation_num,
							image_size,
							stream);
				}

				if (rest != 0)
				{
					if(data_is_3D)
						AccUtilities::diff2_coarse<true,true, D2C_BLOCK_SIZE_DATA3D, 1, 4>(
							rest,
							D2C_BLOCK_SIZE_DATA3D,
							&d_eulers[9*even_orientation_num],
							trans_x,
							trans_y,
							trans_z,
							Fimg_real,
							Fimg_imag,
							projector,
							corr_img,
							&diff2s[translation_num*even_orientation_num],
							translation_num,
							image_size,
							stream);
					else
						AccUtilities::diff2_coarse<true,false, D2C_BLOCK_SIZE_REF3D, 1, 4>(
							rest,
							D2C_BLOCK_SIZE_REF3D,
							&d_eulers[9*even_orientation_num],
							trans_x,
							trans_y,
							trans_z,
							Fimg_real,
							Fimg_imag,
							projector,
							corr_img,
							&diff2s[translation_num*even_orientation_num],
							translation_num,
							image_size,
							stream);
				}
			}
			else if (translation_num <= blocks3D*2)
			{
				if (even_orientation_num != 0)
				{
					if(data_is_3D)
						AccUtilities::diff2_coarse<true,true, D2C_BLOCK_SIZE_DATA3D*2, D2C_EULERS_PER_BLOCK_DATA3D, 4>(
							even_orientation_num/(unsigned long)D2C_EULERS_PER_BLOCK_DATA3D,
							D2C_BLOCK_SIZE_DATA3D*2,
							d_eulers,
							trans_x,
							trans_y,
							trans_z,
							Fimg_real,
							Fimg_imag,
							projector,
							corr_img,
							diff2s,
							translation_num,
							image_size,
							stream);
					else
						AccUtilities::diff2_coarse<true,false, D2C_BLOCK_SIZE_REF3D*2, D2C_EULERS_PER_BLOCK_REF3D, 4>(
							even_orientation_num/(unsigned long)D2C_EULERS_PER_BLOCK_REF3D,
							D2C_BLOCK_SIZE_REF3D*2,
							d_eulers,
							trans_x,
							trans_y,
							trans_z,
							Fimg_real,
							Fimg_imag,
							projector,
							corr_img,
							diff2s,
							translation_num,
							image_size,
							stream);

				}

				if (rest != 0)
				{
					if(data_is_3D)
						AccUtilities::diff2_coarse<true, true, D2C_BLOCK_SIZE_DATA3D*2, 1, 4>(
							rest,
							D2C_BLOCK_SIZE_DATA3D*2,
							&d_eulers[9*even_orientation_num],
							trans_x,
							trans_y,
							trans_z,
							Fimg_real,
							Fimg_imag,
							projector,
							corr_img,
							&diff2s[translation_num*even_orientation_num],
							translation_num,
							image_size,
							stream);
					else
						AccUtilities::diff2_coarse<true,false, D2C_BLOCK_SIZE_REF3D*2, 1, 4>(
							rest,
							D2C_BLOCK_SIZE_REF3D*2,
							&d_eulers[9*even_orientation_num],
							trans_x,
							trans_y,
							trans_z,
							Fimg_real,
							Fimg_imag,
							projector,
							corr_img,
							&diff2s[translation_num*even_orientation_num],
							translation_num,
							image_size,
							stream);
				}
			}
			else if (translation_num <= blocks3D*4)
			{
				if (even_orientation_num != 0)
				{
					if(data_is_3D)
						AccUtilities::diff2_coarse<true,true, D2C_BLOCK_SIZE_DATA3D*4, D2C_EULERS_PER_BLOCK_DATA3D, 4>(
							even_orientation_num/(unsigned long)D2C_EULERS_PER_BLOCK_DATA3D,
							D2C_BLOCK_SIZE_DATA3D*4,
							d_eulers,
							trans_x,
							trans_y,
							trans_z,
							Fimg_real,
							Fimg_imag,
							projector,
							corr_img,
							diff2s,
							translation_num,
							image_size,
							stream);
					else
						AccUtilities::diff2_coarse<true,false, D2C_BLOCK_SIZE_REF3D*4, D2C_EULERS_PER_BLOCK_REF3D, 4>(
							even_orientation_num/(unsigned long)D2C_EULERS_PER_BLOCK_REF3D,
							D2C_BLOCK_SIZE_REF3D*4,
							d_eulers,
							trans_x,
							trans_y,
							trans_z,
							Fimg_real,
							Fimg_imag,
							projector,
							corr_img,
							diff2s,
							translation_num,
							image_size,
							stream);
				}

				if (rest != 0)
				{
					if(data_is_3D)
						AccUtilities::diff2_coarse<true,true, D2C_BLOCK_SIZE_DATA3D*4, 1, 4>(
							rest,
							D2C_BLOCK_SIZE_DATA3D*4,
							&d_eulers[9*even_orientation_num],
							trans_x,
							trans_y,
							trans_z,
							Fimg_real,
							Fimg_imag,
							projector,
							corr_img,
							&diff2s[translation_num*even_orientation_num],
							translation_num,
							image_size,
							stream);
					else
						AccUtilities::diff2_coarse<true,false, D2C_BLOCK_SIZE_REF3D*4, 1, 4>(
							rest,
							D2C_BLOCK_SIZE_REF3D*4,
							&d_eulers[9*even_orientation_num],
							trans_x,
							trans_y,
							trans_z,
							Fimg_real,
							Fimg_imag,
							projector,
							corr_img,
							&diff2s[translation_num*even_orientation_num],
							translation_num,
							image_size,
							stream);
				}
			}
#ifndef ACC_DOUBLE_PRECISION
			else
			{
				if (even_orientation_num != 0)
				{
					if(data_is_3D)
						AccUtilities::diff2_coarse<true,true, D2C_BLOCK_SIZE_DATA3D*8, D2C_EULERS_PER_BLOCK_DATA3D, 4>(
							even_orientation_num/(unsigned long)D2C_EULERS_PER_BLOCK_DATA3D,
							D2C_BLOCK_SIZE_DATA3D*8,
							d_eulers,
							trans_x,
							trans_y,
							trans_z,
							Fimg_real,
							Fimg_imag,
							projector,
							corr_img,
							diff2s,
							translation_num,
							image_size,
							stream);
					else
						AccUtilities::diff2_coarse<true,false, D2C_BLOCK_SIZE_REF3D*8, D2C_EULERS_PER_BLOCK_REF3D, 4>(
							even_orientation_num/(unsigned long)D2C_EULERS_PER_BLOCK_REF3D,
							D2C_BLOCK_SIZE_REF3D*8,
							d_eulers,
							trans_x,
							trans_y,
							trans_z,
							Fimg_real,
							Fimg_imag,
							projector,
							corr_img,
							diff2s,
							translation_num,
							image_size,
							stream);
				}

				if (rest != 0)
				{
					if(data_is_3D)
						AccUtilities::diff2_coarse<true,true, D2C_BLOCK_SIZE_DATA3D*8, 1, 4>(
							rest,
							D2C_BLOCK_SIZE_DATA3D*8,
							&d_eulers[9*even_orientation_num],
							trans_x,
							trans_y,
							trans_z,
							Fimg_real,
							Fimg_imag,
							projector,
							corr_img,
							&diff2s[translation_num*even_orientation_num],
							translation_num,
							image_size,
							stream);
					else
						AccUtilities::diff2_coarse<true,false, D2C_BLOCK_SIZE_REF3D*8, 1, 4>(
							rest,
							D2C_BLOCK_SIZE_REF3D*8,
							&d_eulers[9*even_orientation_num],
							trans_x,
							trans_y,
							trans_z,
							Fimg_real,
							Fimg_imag,
							projector,
							corr_img,
							&diff2s[translation_num*even_orientation_num],
							translation_num,
							image_size,
							stream);
				}
			}
#endif
		}  // projector.mdlZ!=0
		else
		{

			if (translation_num > D2C_BLOCK_SIZE_2D)
			{
				printf("Number of coarse translations larger than %d on the GPU not supported.\n", D2C_BLOCK_SIZE_2D);
				fflush(stdout);
				exit(1);
			}


			long unsigned rest = orientation_num % (unsigned long)D2C_EULERS_PER_BLOCK_2D;
			long unsigned even_orientation_num = orientation_num - rest;

			if (even_orientation_num != 0)
			{
				if(data_is_3D)
					AccUtilities::diff2_coarse<false,true, D2C_BLOCK_SIZE_2D, D2C_EULERS_PER_BLOCK_2D, 2>(
						even_orientation_num/(unsigned long)D2C_EULERS_PER_BLOCK_2D,
						D2C_BLOCK_SIZE_2D,
						d_eulers,
						trans_x,
						trans_y,
						trans_z,
						Fimg_real,
						Fimg_imag,
						projector,
						corr_img,
						diff2s,
						translation_num,
						image_size,
						stream);
				else
					AccUtilities::diff2_coarse<false,false, D2C_BLOCK_SIZE_2D, D2C_EULERS_PER_BLOCK_2D, 2>(
						even_orientation_num/(unsigned long)D2C_EULERS_PER_BLOCK_2D,
						D2C_BLOCK_SIZE_2D,
						d_eulers,
						trans_x,
						trans_y,
						trans_z,
						Fimg_real,
						Fimg_imag,
						projector,
						corr_img,
						diff2s,
						translation_num,
						image_size,
						stream);
				HANDLE_ERROR(cudaGetLastError());
			}

			if (rest != 0)
			{
				if(data_is_3D)
					AccUtilities::diff2_coarse<false,true, D2C_BLOCK_SIZE_2D, 1, 2>(
						rest,
						D2C_BLOCK_SIZE_2D,
						&d_eulers[9*even_orientation_num],
						trans_x,
						trans_y,
						trans_z,
						Fimg_real,
						Fimg_imag,
						projector,
						corr_img,
						&diff2s[translation_num*even_orientation_num],
						translation_num,
						image_size,
						stream);
				else
					AccUtilities::diff2_coarse<false,false, D2C_BLOCK_SIZE_2D, 1, 2>(
						rest,
						D2C_BLOCK_SIZE_2D,
						&d_eulers[9*even_orientation_num],
						trans_x,
						trans_y,
						trans_z,
						Fimg_real,
						Fimg_imag,
						projector,
						corr_img,
						&diff2s[translation_num*even_orientation_num],
						translation_num,
						image_size,
						stream);
				HANDLE_ERROR(cudaGetLastError());
			}
		}  // projector.mdlZ==0
	}  // !do_CC
	else
	{  // do_CC
// TODO - find a more compact way to represent these combinations resulting in
// a single call to diff2_CC_course?
		// dim3 CCblocks(orientation_num,translation_num);
		if(data_is_3D)
			AccUtilities::diff2_CC_coarse<true,true,D2C_BLOCK_SIZE_DATA3D>(
				orientation_num,
				D2C_BLOCK_SIZE_DATA3D,
				d_eulers,
				Fimg_real,
				Fimg_imag,
				trans_x,
				trans_y,
				trans_z,
				projector,
				corr_img,
				diff2s,
				translation_num,
				image_size,
				local_sqrtXi2,
				stream);
		else if(projector.mdlZ!=0)
			AccUtilities::diff2_CC_coarse<true,false,D2C_BLOCK_SIZE_REF3D>(
				orientation_num,
				D2C_BLOCK_SIZE_REF3D,
				d_eulers,
				Fimg_real,
				Fimg_imag,
				trans_x,
				trans_y,
				trans_z,
				projector,
				corr_img,
				diff2s,
				translation_num,
				image_size,
				local_sqrtXi2,
				stream);
		else
			AccUtilities::diff2_CC_coarse<false,false,D2C_BLOCK_SIZE_2D>(
				orientation_num,
				D2C_BLOCK_SIZE_2D,
				d_eulers,
				Fimg_real,
				Fimg_imag,
				trans_x,
				trans_y,
				trans_z,
				projector,
				corr_img,
				diff2s,
				translation_num,
				image_size,
				local_sqrtXi2,
				stream);
		HANDLE_ERROR(cudaGetLastError());
	} // do_CC
}



void runDiff2KernelCoarseDouble(
	AccProjectorKernel &projector,
	XFLOAT *trans_x,
	XFLOAT *trans_y,
	XFLOAT *trans_z,
	XFLOAT *corr_img,
	XFLOAT *Fimg_real,
	XFLOAT *Fimg_imag,
	XFLOAT *d_eulers,
	double *diff2s,
	XFLOAT local_sqrtXi2,
	long unsigned orientation_num,
	long unsigned translation_num,
	long unsigned image_size,
	cudaStream_t stream,
	bool do_CC,
	bool data_is_3D,
	bool filter_img = false)//内部可能多次调用kernel
{
const long unsigned blocks3D = (data_is_3D? D2C_BLOCK_SIZE_DATA3D_D : D2C_BLOCK_SIZE_REF3D_D);
	if(projector.mdlZ!=0)
	{

// #ifdef ACC_DOUBLE_PRECISION
// 			if (translation_num > blocks3D*4)
// 				CRITICAL(ERR_TRANSLIM);
// #else
// 			if (translation_num > blocks3D*8)
// 				CRITICAL(ERR_TRANSLIM);
// #endif

		long unsigned rest = orientation_num % blocks3D;
		long unsigned even_orientation_num = orientation_num - rest;
// TODO - find a more compact way to represent these combinations resulting in
// a single call to diff2_course?
		if (translation_num <= blocks3D)
		{
			if (even_orientation_num != 0)
			{
				if(data_is_3D)
					AccUtilities::diff2_coarse_double<true,true, D2C_BLOCK_SIZE_DATA3D_D, D2C_EULERS_PER_BLOCK_DATA3D_D, 4>(
						even_orientation_num/(unsigned long)D2C_EULERS_PER_BLOCK_DATA3D_D,
						D2C_BLOCK_SIZE_DATA3D_D,
						d_eulers,
						trans_x,
						trans_y,
						trans_z,
						Fimg_real,
						Fimg_imag,
						projector,
						corr_img,
						diff2s,
						translation_num,
						image_size,
						stream,
						filter_img);
				else
					AccUtilities::diff2_coarse_double<true,false, D2C_BLOCK_SIZE_REF3D_D, D2C_EULERS_PER_BLOCK_REF3D_D, 4>(
						even_orientation_num/(unsigned long)D2C_EULERS_PER_BLOCK_REF3D_D,
						D2C_BLOCK_SIZE_REF3D_D,
						d_eulers,
						trans_x,
						trans_y,
						trans_z,
						Fimg_real,
						Fimg_imag,
						projector,
						corr_img,
						diff2s,
						translation_num,
						image_size,
						stream,
						filter_img);
			}

			if (rest != 0)
			{
				if(data_is_3D)
					AccUtilities::diff2_coarse_double<true,true, D2C_BLOCK_SIZE_DATA3D_D, 1, 4>(
						rest,
						D2C_BLOCK_SIZE_DATA3D_D,
						&d_eulers[9*even_orientation_num],
						trans_x,
						trans_y,
						trans_z,
						Fimg_real,
						Fimg_imag,
						projector,
						corr_img,
						&diff2s[translation_num*even_orientation_num],
						translation_num,
						image_size,
						stream,
						filter_img);
				else
					AccUtilities::diff2_coarse_double<true,false, D2C_BLOCK_SIZE_REF3D_D, 1, 4>(
						rest,
						D2C_BLOCK_SIZE_REF3D_D,
						&d_eulers[9*even_orientation_num],
						trans_x,
						trans_y,
						trans_z,
						Fimg_real,
						Fimg_imag,
						projector,
						corr_img,
						&diff2s[translation_num*even_orientation_num],
						translation_num,
						image_size,
						stream,
						filter_img);
			}
		}
		else if (translation_num <= blocks3D*2)
		{
			if (even_orientation_num != 0)
			{
				if(data_is_3D)
					AccUtilities::diff2_coarse_double<true,true, D2C_BLOCK_SIZE_DATA3D_D * 2, D2C_EULERS_PER_BLOCK_DATA3D_D, 4>(
						even_orientation_num/(unsigned long)D2C_EULERS_PER_BLOCK_DATA3D_D,
						D2C_BLOCK_SIZE_DATA3D_D * 2,
						d_eulers,
						trans_x,
						trans_y,
						trans_z,
						Fimg_real,
						Fimg_imag,
						projector,
						corr_img,
						diff2s,
						translation_num,
						image_size,
						stream,
						filter_img);
				else
					AccUtilities::diff2_coarse_double<true,false, D2C_BLOCK_SIZE_REF3D_D * 2, D2C_EULERS_PER_BLOCK_REF3D_D, 4>(
						even_orientation_num/(unsigned long)D2C_EULERS_PER_BLOCK_REF3D_D,
						D2C_BLOCK_SIZE_REF3D_D * 2,
						d_eulers,
						trans_x,
						trans_y,
						trans_z,
						Fimg_real,
						Fimg_imag,
						projector,
						corr_img,
						diff2s,
						translation_num,
						image_size,
						stream,
						filter_img);

			}

			if (rest != 0)
			{
				if(data_is_3D)
					AccUtilities::diff2_coarse_double<true, true, D2C_BLOCK_SIZE_DATA3D_D * 2, 1, 4>(
						rest,
						D2C_BLOCK_SIZE_DATA3D_D * 2,
						&d_eulers[9*even_orientation_num],
						trans_x,
						trans_y,
						trans_z,
						Fimg_real,
						Fimg_imag,
						projector,
						corr_img,
						&diff2s[translation_num*even_orientation_num],
						translation_num,
						image_size,
						stream,
						filter_img);
				else
					AccUtilities::diff2_coarse_double<true,false, D2C_BLOCK_SIZE_REF3D_D * 2, 1, 4>(
						rest,
						D2C_BLOCK_SIZE_REF3D_D * 2,
						&d_eulers[9*even_orientation_num],
						trans_x,
						trans_y,
						trans_z,
						Fimg_real,
						Fimg_imag,
						projector,
						corr_img,
						&diff2s[translation_num*even_orientation_num],
						translation_num,
						image_size,
						stream,
						filter_img);
			}
		}
		else if (translation_num <= blocks3D*4)
		{
			if (even_orientation_num != 0)
			{
				if(data_is_3D)
					AccUtilities::diff2_coarse_double<true,true, D2C_BLOCK_SIZE_DATA3D_D*4, D2C_EULERS_PER_BLOCK_DATA3D_D, 4>(
						even_orientation_num/(unsigned long)D2C_EULERS_PER_BLOCK_DATA3D_D,
						D2C_BLOCK_SIZE_DATA3D_D*4,
						d_eulers,
						trans_x,
						trans_y,
						trans_z,
						Fimg_real,
						Fimg_imag,
						projector,
						corr_img,
						diff2s,
						translation_num,
						image_size,
						stream,
						filter_img);
				else
					AccUtilities::diff2_coarse_double<true,false, D2C_BLOCK_SIZE_REF3D_D*4, D2C_EULERS_PER_BLOCK_REF3D_D, 4>(
						even_orientation_num/(unsigned long)D2C_EULERS_PER_BLOCK_REF3D_D,
						D2C_BLOCK_SIZE_REF3D_D*4,
						d_eulers,
						trans_x,
						trans_y,
						trans_z,
						Fimg_real,
						Fimg_imag,
						projector,
						corr_img,
						diff2s,
						translation_num,
						image_size,
						stream,
						filter_img);
			}

			if (rest != 0)
			{
				if(data_is_3D)
					AccUtilities::diff2_coarse_double<true,true, D2C_BLOCK_SIZE_DATA3D_D*4, 1, 4>(
						rest,
						D2C_BLOCK_SIZE_DATA3D_D*4,
						&d_eulers[9*even_orientation_num],
						trans_x,
						trans_y,
						trans_z,
						Fimg_real,
						Fimg_imag,
						projector,
						corr_img,
						&diff2s[translation_num*even_orientation_num],
						translation_num,
						image_size,
						stream,
						filter_img);
				else
					AccUtilities::diff2_coarse_double<true,false, D2C_BLOCK_SIZE_REF3D_D*4, 1, 4>(
						rest,
						D2C_BLOCK_SIZE_REF3D_D*4,
						&d_eulers[9*even_orientation_num],
						trans_x,
						trans_y,
						trans_z,
						Fimg_real,
						Fimg_imag,
						projector,
						corr_img,
						&diff2s[translation_num*even_orientation_num],
						translation_num,
						image_size,
						stream,
						filter_img);
			}
		}
#ifndef ACC_DOUBLE_PRECISION
		else
		{
			if (even_orientation_num != 0)
			{
				if(data_is_3D)
					AccUtilities::diff2_coarse_double<true,true, D2C_BLOCK_SIZE_DATA3D_D*8, D2C_EULERS_PER_BLOCK_DATA3D_D, 4>(
						even_orientation_num/(unsigned long)D2C_EULERS_PER_BLOCK_DATA3D_D,
						D2C_BLOCK_SIZE_DATA3D_D*8,
						d_eulers,
						trans_x,
						trans_y,
						trans_z,
						Fimg_real,
						Fimg_imag,
						projector,
						corr_img,
						diff2s,
						translation_num,
						image_size,
						stream,
						filter_img);
				else
					AccUtilities::diff2_coarse_double<true,false, D2C_BLOCK_SIZE_REF3D_D*8, D2C_EULERS_PER_BLOCK_REF3D_D, 4>(
						even_orientation_num/(unsigned long)D2C_EULERS_PER_BLOCK_REF3D_D,
						D2C_BLOCK_SIZE_REF3D_D*8,
						d_eulers,
						trans_x,
						trans_y,
						trans_z,
						Fimg_real,
						Fimg_imag,
						projector,
						corr_img,
						diff2s,
						translation_num,
						image_size,
						stream,
						filter_img);
			}

			if (rest != 0)
			{
				if(data_is_3D)
					AccUtilities::diff2_coarse_double<true,true, D2C_BLOCK_SIZE_DATA3D_D*8, 1, 4>(
						rest,
						D2C_BLOCK_SIZE_DATA3D_D*8,
						&d_eulers[9*even_orientation_num],
						trans_x,
						trans_y,
						trans_z,
						Fimg_real,
						Fimg_imag,
						projector,
						corr_img,
						&diff2s[translation_num*even_orientation_num],
						translation_num,
						image_size,
						stream,
						filter_img);
				else
					AccUtilities::diff2_coarse_double<true,false, D2C_BLOCK_SIZE_REF3D_D*8, 1, 4>(
						rest,
						D2C_BLOCK_SIZE_REF3D_D*8,
						&d_eulers[9*even_orientation_num],
						trans_x,
						trans_y,
						trans_z,
						Fimg_real,
						Fimg_imag,
						projector,
						corr_img,
						&diff2s[translation_num*even_orientation_num],
						translation_num,
						image_size,
						stream,
						filter_img);
			}
		}
#endif
	}  // projector.mdlZ!=0
}

#endif