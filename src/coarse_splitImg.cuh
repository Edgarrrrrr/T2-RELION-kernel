#ifndef COARSE_MATRIX_KERNEL_IM2COL_SPLITIMG_CUH
#define COARSE_MATRIX_KERNEL_IM2COL_SPLITIMG_CUH

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
#include "./kernel_block_params.cuh"

template<typename TParams>
__launch_bounds__(128, BLOCKS_PER_SM)
__global__ void CoarseMatrixKernelIm2colSplitImgKernel(
	XFLOAT *g_eulers,
	XFLOAT *trans_x,
	XFLOAT *trans_y,
	XFLOAT *g_real,
	XFLOAT *g_imag,
	AccProjectorKernel projector,
	XFLOAT *g_corr,
	XFLOAT *g_diff2s,
	XFLOAT *g_diff2s_dest,
	const int translation_num,
	const int orientation_num,
	const int image_size);

template <typename TParams = void>
struct CoarseMatrixKernelIm2colSplitImg {
  const int translation_num_;
  const int orientation_num_;
  const int image_size_;
  const int sm_num_;

  CoarseMatrixKernelIm2colSplitImg(
      int translation_num, int orientation_num, int image_size, int sm_num = 108)
      : translation_num_(translation_num),
        orientation_num_(orientation_num),
        image_size_(image_size),
        sm_num_(sm_num) {}

  // Assistant function to dispatch TParams based on translation_num_
  template <typename Func>
  auto dispatch_TParams(Func&& func) const {
    return func(TypeHolder<CoarseTParam64x128_32x64>{});
  }

  // Calculate workspace size in bytes
  size_t get_workspace_size_bytes() const {
    return 0;
  }

  void run(XFLOAT *g_eulers, 
           XFLOAT *trans_x, 
           XFLOAT *trans_y, 
           XFLOAT *g_real,
           XFLOAT *g_imag, 
           AccProjectorKernel projector, 
           XFLOAT *g_corr,
           XFLOAT *g_diff2s, 
           XFLOAT *g_diff2s_dest,
           cudaStream_t stream,
		   XFLOAT* g_coor_xy = nullptr,
           uint32_t *work_space = nullptr) {
    
    (void)work_space;

    if constexpr (!std::is_same_v<TParams, void>) {
      dim3 grid(sm_num_ * BLOCKS_PER_SM, 1, 1);
      dim3 block(TParams::kBlockSize, 1, 1);
      CoarseMatrixKernelIm2colSplitImgKernel<TParams>
	  <<<grid, block, 0, stream>>>(
          g_eulers, trans_x, trans_y, g_real, g_imag, projector, g_corr,
          g_diff2s, g_diff2s_dest, translation_num_, orientation_num_, image_size_);
    } else {
      dispatch_TParams([=](auto dummy) {
        using SelectedTParams = typename decltype(dummy)::type;
        
		dim3 grid(sm_num_ * BLOCKS_PER_SM, 1, 1);
        dim3 block(SelectedTParams::kBlockSize, 1, 1);
        
		CoarseMatrixKernelIm2colSplitImgKernel<SelectedTParams><<<grid, block, 0, stream>>>(
            g_eulers, trans_x, trans_y, g_real, g_imag, projector, g_corr,
            g_diff2s, g_diff2s_dest, translation_num_, orientation_num_, image_size_);
      });
    }
  }

  void print_workspace_buffer(uint32_t* workspace) {
	printf("workspace buffer is empty\n");
  }
};

template<typename TParams>
__launch_bounds__(128, BLOCKS_PER_SM)
__global__ void CoarseMatrixKernelIm2colSplitImgKernel(
	XFLOAT *g_eulers,
	XFLOAT *trans_x,
	XFLOAT *trans_y,
	XFLOAT *g_real,
	XFLOAT *g_imag,
	AccProjectorKernel projector,
	XFLOAT *g_corr,
	XFLOAT *g_diff2s,
	XFLOAT *g_diff2s_dest,
	const int translation_num,
	const int orientation_num,
	const int image_size) {
    static_assert(TParams::kBlockSize % 32 == 0, "kBlockSize must be a multiple of 32");
	static_assert(TParams::kImgBlockSize == TParams::kWarpImgTileSize, "kImgBlockSize must be equal to kWarpImgTileSize");
	static_assert(TParams::kBlockSize >= TParams::kTransBlockSize, "kBlockSize must be greater than or equal to kTransBlockSize");
	static_assert(TParams::kBlockSize >= TParams::kOrientBlockSize, "kBlockSize must be greater than or equal to kOrientBlockSize");

	static_assert(TParams::kTransBlockSize % TParams::kWarpTransTileSize == 0, "kTransBlockSize must be a multiple of kWarpTransTileSize");
	static_assert(TParams::kOrientBlockSize % TParams::kWarpOrientTileSize == 0, "kOrientBlockSize must be a multiple of kWarpOrientTileSize");
	static_assert(TParams::kTransBlockSize % TParams::kMmaTransTileSize == 0, "kTransBlockSize must be a multiple of kMmaTransTileSize");
	static_assert(TParams::kOrientBlockSize % TParams::kMmaOrientTileSize == 0, "kOrientBlockSize must be a multiple of kMmaOrientTileSize");
	static_assert(TParams::kBlockSize / 32 == (TParams::kTransBlockSize / TParams::kWarpTransTileSize) * (TParams::kOrientBlockSize / TParams::kWarpOrientTileSize), "kBlockSize must be equal to the product of the number of warps in translation, orientation and image dimension");

	static_assert(TParams::kImgBlockSize == 16, "kImgBlockSize must be 16");

	const int tid = threadIdx.x;          // thread id in a block
	const int bid = blockIdx.x;           // block id in a grid
	const int warp_id  = tid / 32;        // warp id in a block
	const int warp_num = TParams::kBlockSize / 32; // number of warps in a block
	const int lane_id  = tid % 32;        // thread id in a warp

	const int trans_block_num = (translation_num + TParams::kTransBlockSize - 1) / TParams::kTransBlockSize;
	const int orient_block_num = (orientation_num + TParams::kOrientBlockSize - 1) / TParams::kOrientBlockSize;

	int trans_block_idx = (bid % trans_block_num) * TParams::kTransBlockSize;
	int orient_block_idx = (bid / trans_block_num) * TParams::kOrientBlockSize;

	assert(blockIdx.y == 0 && blockIdx.z == 0 && threadIdx.y == 0 && threadIdx.z == 0);
	
	CoarseScheduler<TParams::kTransBlockSize, 
					TParams::kOrientBlockSize, 
					TParams::kImgBlockSize, 
					CoarseSchedulerStrategy::SplitK>
		scheduler(translation_num, orientation_num, image_size);
	

	// 'img' data is stored contiguously.
	__align__(16) __shared__ XFLOAT s_trans_mat_block[2 * TParams::kTransBlockSize * TParams::kImgBlockSize];
	__align__(16) __shared__ XFLOAT s_orient_mat_block[2 * TParams::kOrientBlockSize * TParams::kImgBlockSize];
	
	SharedMemorySwizzle<float, TParams::kTransBlockSize, 2 * TParams::kImgBlockSize, 0> s_trans_mat_block_swizzle(s_trans_mat_block);
	SharedMemorySwizzle<float, TParams::kTransBlockSize, TParams::kImgBlockSize, 0> s_trans_real_mat_block_swizzle(s_trans_mat_block);
	SharedMemorySwizzle<float, TParams::kTransBlockSize, TParams::kImgBlockSize, TParams::kImgBlockSize> s_trans_imag_mat_block_swizzle(s_trans_mat_block);

	SharedMemorySwizzle<float, TParams::kOrientBlockSize, 2 * TParams::kImgBlockSize, 0> s_orient_mat_block_swizzle(s_orient_mat_block);
	SharedMemorySwizzle<float, TParams::kOrientBlockSize, TParams::kImgBlockSize, 0> s_orient_real_mat_block_swizzle(s_orient_mat_block);
	SharedMemorySwizzle<float, TParams::kOrientBlockSize, TParams::kImgBlockSize, TParams::kImgBlockSize> s_orient_imag_mat_block_swizzle(s_orient_mat_block);

	// double buffer for s_corr_div_2, s_coor_x, s_coor_y
	__align__(16) __shared__ XFLOAT s_corr_div_2[2][TParams::kImgBlockSize];
	__align__(16) __shared__ XFLOAT s_coor_x[2][TParams::kImgBlockSize];
	__align__(16) __shared__ XFLOAT s_coor_y[2][TParams::kImgBlockSize];

	// reduce buffer
	__align__(16) __shared__ XFLOAT s_trans_pow2_accumulator[(TParams::kBlockSize / TParams::kTransBlockSize) * TParams::kTransBlockSize];
	__align__(16) __shared__ XFLOAT s_orient_pow2_accumulator[(TParams::kBlockSize / TParams::kOrientBlockSize) * TParams::kOrientBlockSize];

	// register
	constexpr int kNumMmaTransInWarpTile = TParams::kWarpTransTileSize / TParams::kMmaTransTileSize;
	constexpr int kNumMmaOrientInWarpTile = TParams::kWarpOrientTileSize / TParams::kMmaOrientTileSize;
	constexpr int kNumMmaImgInWarpTile = TParams::kWarpImgTileSize / TParams::kMmaImgTileSize;

	constexpr int kFragmentASize = TParams::kMmaTransTileSize * TParams::kMmaImgTileSize / kWarpSize;
	constexpr int kFragmentBSize = TParams::kMmaOrientTileSize * TParams::kMmaImgTileSize / kWarpSize;
	constexpr int kFragmentCSize = TParams::kMmaTransTileSize * TParams::kMmaOrientTileSize / kWarpSize;

	XFLOAT fragment_c[kNumMmaTransInWarpTile][kNumMmaOrientInWarpTile][kFragmentCSize];

	// ============================= lambda function =============================
	//given current img_block_idx, load global array into corr_div_2, coord_x, coord_y
	auto load_coord_xy = [&](int img_block_idx, XFLOAT* corr_div_2, XFLOAT* coord_x, XFLOAT* coord_y) {
		#pragma unroll 
		for (int i = tid; i < TParams::kImgBlockSize; i += TParams::kBlockSize) {
			if (img_block_idx + i < image_size) {
				corr_div_2[i] = g_corr[img_block_idx + i] / 2;
			} else {
				corr_div_2[i] = 0.;
			}

			int x, y;
			pixel_index2coor(img_block_idx + i, projector.imgX, projector.imgY, projector.maxR, x, y);
			coord_x[i] = x;
			coord_y[i] = y;
		}
	};

	//given current img_block_idx, corr_div_2, coord_x, coord_y, load trans mat into s_trans_real_mat_block and s_trans_imag_mat_block
	auto load_trans_mat = [&](int img_block_idx, XFLOAT* corr_div_2, XFLOAT* coord_x, XFLOAT* coord_y) {
		assert(TParams::kBlockSize % TParams::kTransBlockSize == 0);
		#pragma unroll
		for (int i = tid / TParams::kTransBlockSize; i < TParams::kImgBlockSize; i += TParams::kBlockSize / TParams::kTransBlockSize) {
			int g_img_idx = img_block_idx + i;
			int trans_idx = tid % TParams::kTransBlockSize;
			if (g_img_idx >= image_size) {
				assert(trans_idx < TParams::kTransBlockSize);
				assert(i < TParams::kImgBlockSize);
				s_trans_real_mat_block_swizzle(trans_idx, i) = 0.;
				s_trans_imag_mat_block_swizzle(trans_idx, i) = 0.;
				continue;
			}
			int g_trans_idx = trans_block_idx + trans_idx;
			if (g_trans_idx >= translation_num) {
				continue;
			}
			XFLOAT tx = trans_x[g_trans_idx];
			XFLOAT ty = trans_y[g_trans_idx];
			XFLOAT real = g_real[g_img_idx];
			XFLOAT imag = g_imag[g_img_idx];

			int x = coord_x[i];
			int y = coord_y[i];
			XFLOAT trans_real, trans_imag;
			translatePixel(x, y, tx, ty, real, imag, trans_real, trans_imag);

			s_trans_real_mat_block_swizzle(trans_idx, i) = -2 * trans_real * corr_div_2[i];
			s_trans_imag_mat_block_swizzle(trans_idx, i) = -2 * trans_imag * corr_div_2[i];

			XFLOAT magnitude_squared_sum = trans_real * trans_real * corr_div_2[i] + trans_imag * trans_imag * corr_div_2[i];
			s_trans_pow2_accumulator[tid] += magnitude_squared_sum;
		}
	};

	auto project3Dmodel_sp = [&](
			XFLOAT x,
			XFLOAT y,
			XFLOAT e0,
			XFLOAT e1,
			XFLOAT e3,
			XFLOAT e4,
			XFLOAT e6,
			XFLOAT e7,
			XFLOAT &real,
			XFLOAT &imag,
			uint32_t& flag_minus,
			uint32_t mask) {
		XFLOAT xp = (e0 * x + e1 * y ) * projector.padding_factor;
		XFLOAT yp = (e3 * x + e4 * y ) * projector.padding_factor;
		XFLOAT zp = (e6 * x + e7 * y ) * projector.padding_factor;
		int r2 = xp*xp + yp*yp + zp*zp;
		if (r2 <= projector.maxR2_padded)
		{
			bool xp_neg = xp < 0;
			flag_minus += xp_neg ? mask : 0;
			// NOTICE: if xp_neg, imag = -imag
			if (xp_neg) {
				// Get complex conjugated hermitian symmetry pair
				xp = -xp;
				yp = -yp;
				zp = -zp;
				yp -= projector.mdlInitY;
				zp -= projector.mdlInitZ;
			}
			else {
				yp -= projector.mdlInitY;
				zp -= projector.mdlInitZ;
			}
			real =    tex3D<XFLOAT>(projector.mdlReal, xp + (XFLOAT)0.5, yp + (XFLOAT)0.5, zp + (XFLOAT)0.5);
			imag =    tex3D<XFLOAT>(projector.mdlImag, xp + (XFLOAT)0.5, yp + (XFLOAT)0.5, zp + (XFLOAT)0.5);
		}
		else {
			real = (XFLOAT)0;
			imag = (XFLOAT)0;
		}
	};

	constexpr int kDimOrientSlice = (TParams::kBlockSize / TParams::kOrientBlockSize);
	constexpr int kNumOrientSlice = (TParams::kImgBlockSize + kDimOrientSlice - 1) / kDimOrientSlice;
	assert(kNumOrientSlice <=32);


	XFLOAT orient_real_buf[kNumOrientSlice], orient_imag_buf[kNumOrientSlice];
	uint32_t flag_minus_buf[2] = {0, 0};

	//given current img_block_idx, corr_div_2, coord_x, coord_y, load orient mat into orient_real_buf and orient_imag_buf
	auto load_orient_mat_buf = [&](int img_block_idx, XFLOAT* corr_div_2, XFLOAT* coord_x, XFLOAT* coord_y, uint32_t& flag_minus)  {
		uint32_t flag_minus_loc = 0;

		#pragma unroll
		for (int cur_slice = 0; cur_slice < kNumOrientSlice; cur_slice++) {
			int i = tid / TParams::kOrientBlockSize + cur_slice * kDimOrientSlice;
			XFLOAT& orient_real = orient_real_buf[cur_slice];
			XFLOAT& orient_imag = orient_imag_buf[cur_slice];
			int g_img_idx = img_block_idx + i;
			int orient_idx = tid % TParams::kOrientBlockSize;
			int g_orient_idx = orient_block_idx + orient_idx;
			if (g_img_idx >= image_size || g_orient_idx >= orientation_num) {
				assert(orient_idx < TParams::kOrientBlockSize);
				assert(i < TParams::kImgBlockSize);
				orient_real = 0.0;
				orient_imag = 0.0;
			} else {
				XFLOAT e0 = g_eulers[g_orient_idx * 9];
				XFLOAT e1 = g_eulers[g_orient_idx * 9 + 1];
				XFLOAT e3 = g_eulers[g_orient_idx * 9 + 3];
				XFLOAT e4 = g_eulers[g_orient_idx * 9 + 4];
				XFLOAT e6 = g_eulers[g_orient_idx * 9 + 6];
				XFLOAT e7 = g_eulers[g_orient_idx * 9 + 7];


				project3Dmodel_sp(coord_x[i], coord_y[i], e0, e1, e3, e4, e6, e7, orient_real, orient_imag, flag_minus_loc, 1U << (cur_slice % 32));

			}
		}
		flag_minus += flag_minus_loc;
	};

	//given current img_block_idx, corr_div_2, coord_x, coord_y, dump orient_real_buf and orient_imag_buf into s_orient_real_mat_block and s_orient_imag_mat_block
	auto dump_orient_mat_shm = [&](int img_block_idx, XFLOAT* corr_div_2, XFLOAT* coord_x, XFLOAT* coord_y, uint32_t& flag_minus)  {
		#pragma unroll
		for (int cur_slice = 0; cur_slice < kNumOrientSlice; cur_slice++) {
			int i = tid / TParams::kOrientBlockSize + cur_slice * kDimOrientSlice;
			XFLOAT& orient_real = orient_real_buf[cur_slice];
			XFLOAT& orient_imag = orient_imag_buf[cur_slice];
			
			bool flag_cur_minus = (flag_minus & (1U << (cur_slice % 32))) >> (cur_slice % 32);
			orient_imag = flag_cur_minus ? -orient_imag : orient_imag;

			int orient_idx = tid % TParams::kOrientBlockSize;

			s_orient_real_mat_block_swizzle(orient_idx, i) = orient_real;
			s_orient_imag_mat_block_swizzle(orient_idx, i) = orient_imag;

			XFLOAT magnitude_squared_sum = orient_real * orient_real * corr_div_2[i] + orient_imag * orient_imag * corr_div_2[i];
			s_orient_pow2_accumulator[tid] += magnitude_squared_sum;
		}
		flag_minus = 0;
	};

    auto init_fragment_c = [&] () {
		// Default: need read from g_diff2s
		if (scheduler.get_strategy() == CoarseSchedulerStrategy::Default || 
		   (scheduler.get_strategy() == CoarseSchedulerStrategy::SplitK && g_diff2s != g_diff2s_dest)) {		
			#pragma unroll
			for (int i = 0; i < kNumMmaTransInWarpTile; ++i) {
				#pragma unroll
				for (int j = 0; j < kNumMmaOrientInWarpTile; ++j) {
					#pragma unroll
					for (int k = 0; k < kFragmentCSize; ++k) {
						int m = fragment_c_m_idx_in_global<TParams>(trans_block_idx, warp_id, lane_id, i, k);
						int n = fragment_c_n_idx_in_global<TParams>(orient_block_idx, warp_id, lane_id, j, k);
						if (m < translation_num && n < orientation_num) {
							fragment_c[i][j][k] = g_diff2s[n * translation_num + m];
						} else {
							fragment_c[i][j][k] = 0.0;
						}
					}
				}
			}
		}
		// SplitK: use atomicAdd to accumulate, if diff2s source == diff2s dest, no need to read from g_diff2s
		// else, read from g_diff2s
		else if (scheduler.get_strategy() == CoarseSchedulerStrategy::SplitK && g_diff2s == g_diff2s_dest) {
			#pragma unroll
			for (int i = 0; i < kNumMmaTransInWarpTile; ++i) {
				#pragma unroll
				for (int j = 0; j < kNumMmaOrientInWarpTile; ++j) {
					#pragma unroll
					for (int k = 0; k < kFragmentCSize; ++k) {
						fragment_c[i][j][k] = 0.0;
					}
				}
			}
			
		}
		else {
			assert(false);
		}
    };

	auto epilogue = [&] () {
		// write fragment_c back to g_diff2s_dest
        if (scheduler.get_strategy() == CoarseSchedulerStrategy::Default) {
			#pragma unroll
			for (int i = 0; i < kNumMmaTransInWarpTile; ++i) {
				#pragma unroll
				for (int j = 0; j < kNumMmaOrientInWarpTile; ++j) {
					#pragma unroll
					for (int k = 0; k < kFragmentCSize; ++k) {
						int m = fragment_c_m_idx_in_global<TParams>(trans_block_idx, warp_id, lane_id, i, k);
						int n = fragment_c_n_idx_in_global<TParams>(orient_block_idx, warp_id, lane_id, j, k);

						if (m < translation_num && n < orientation_num) {
							g_diff2s_dest[n * translation_num + m] = fragment_c[i][j][k];
						}
					}
				}
			}
		} else if (scheduler.get_strategy() == CoarseSchedulerStrategy::SplitK) {
			// use atomic add
			#pragma unroll
			for (int i = 0; i < kNumMmaTransInWarpTile; ++i) {
				#pragma unroll
				for (int j = 0; j < kNumMmaOrientInWarpTile; ++j) {
					#pragma unroll
					for (int k = 0; k < kFragmentCSize; ++k) {
						int m = fragment_c_m_idx_in_global<TParams>(trans_block_idx, warp_id, lane_id, i, k);
						int n = fragment_c_n_idx_in_global<TParams>(orient_block_idx, warp_id, lane_id, j, k);

						if (m < translation_num && n < orientation_num) {
							atomicAdd(&g_diff2s_dest[n * translation_num + m], fragment_c[i][j][k]);
						}
					}
				}
			}
		} else {
			assert(false);
		}
	};

    // =====================================================================
	// ============================= main loop =============================
    // =====================================================================
    while (scheduler.has_work()) {
		__syncthreads();

        trans_block_idx = scheduler.get_current_work_m_block_offset();
        orient_block_idx = scheduler.get_current_work_n_block_offset();

        // initialize shared memory to zero
        for (int i = tid; i < 2 * TParams::kTransBlockSize * TParams::kImgBlockSize; i += TParams::kBlockSize) {
            s_trans_mat_block[i] = 0.0;
        }
        for (int i = tid; i < 2 * TParams::kOrientBlockSize * TParams::kImgBlockSize; i += TParams::kBlockSize) {
            s_orient_mat_block[i] = 0.0;
        }

        s_trans_pow2_accumulator[tid] = 0.0;
        s_orient_pow2_accumulator[tid] = 0.0;

        for (int i = tid; i < TParams::kImgBlockSize; i += TParams::kBlockSize) {
            s_corr_div_2[0][i] = 0.0;
            s_corr_div_2[1][i] = 0.0;
            s_coor_x[0][i] = 0.0;
            s_coor_y[0][i] = 0.0;
            s_coor_x[1][i] = 0.0;
            s_coor_y[1][i] = 0.0;
        }

		flag_minus_buf[0] = 0;
		flag_minus_buf[1] = 0;

        // read fragment_c from g_diff2s
        init_fragment_c();

        __syncthreads();

/*=============================== FOR IMAGE BLOCK ==============================*/
		int img_block_idx = -1;
        while (scheduler.get_current_work_next_k_block_offset(img_block_idx)) {
			assert(img_block_idx >= 0);
			// int img_iter = img_block_idx / TParams::kImgBlockSize;
			int k_cycle = scheduler.get_current_work_k_cycle();
			int k_cycle_mod2 = k_cycle % 2;
			int k_cycle_next_mod2 = (k_cycle + 1) % 2;
            load_coord_xy(img_block_idx, s_corr_div_2[k_cycle_mod2], s_coor_x[k_cycle_mod2], s_coor_y[k_cycle_mod2]);
            __syncthreads();
            load_trans_mat(img_block_idx, s_corr_div_2[k_cycle_mod2], s_coor_x[k_cycle_mod2], s_coor_y[k_cycle_mod2]);
            __syncthreads();
            load_orient_mat_buf(img_block_idx, s_corr_div_2[k_cycle_mod2], s_coor_x[k_cycle_mod2], s_coor_y[k_cycle_mod2], flag_minus_buf[k_cycle_mod2]);
            __syncthreads();
            dump_orient_mat_shm(img_block_idx, s_corr_div_2[k_cycle_mod2], s_coor_x[k_cycle_mod2], s_coor_y[k_cycle_mod2], flag_minus_buf[k_cycle_mod2]);
            __syncthreads();
    /*=============================== COMPUTE CROSS TERM ==============================*/

            block_mma_tf32_sim_fp32<decltype(s_trans_mat_block_swizzle), decltype(s_orient_mat_block_swizzle), 
            TParams::kTransBlockSize, TParams::kOrientBlockSize, 2 * TParams::kImgBlockSize,
            TParams::kWarpTransTileSize, TParams::kWarpOrientTileSize, 2 * TParams::kWarpImgTileSize,
            TParams::kMmaTransTileSize, TParams::kMmaOrientTileSize, TParams::kMmaImgTileSize>(
                fragment_c, s_trans_mat_block_swizzle, s_orient_mat_block_swizzle, warp_id, lane_id);

        } // end of image block

        // reduce s_trans_pow2_accumulator
        for (int i = 1; i < TParams::kBlockSize / TParams::kTransBlockSize; ++i) {
            if (tid < TParams::kTransBlockSize) {
                s_trans_pow2_accumulator[tid] += s_trans_pow2_accumulator[i * TParams::kTransBlockSize + tid];
            }
        }
        // reduce s_orient_pow2_accumulator
        for (int i = 1; i < TParams::kBlockSize / TParams::kOrientBlockSize; ++i) {
            if (tid < TParams::kOrientBlockSize) {
                s_orient_pow2_accumulator[tid] += s_orient_pow2_accumulator[i * TParams::kOrientBlockSize + tid];
            }
        }

    /*=============================== REDUCE IN FRAGMENT_C ==============================*/
        __syncthreads();
        
        #pragma unroll
        for (int i = 0; i < kNumMmaTransInWarpTile; ++i) {
            #pragma unroll
            for (int j = 0; j < kNumMmaOrientInWarpTile; ++j) {
                #pragma unroll
                for (int k = 0; k < kFragmentCSize; ++k) {
                    int m = fragment_c_m_idx_in_block<TParams>(warp_id, lane_id, i, k);
                    int n = fragment_c_n_idx_in_block<TParams>(warp_id, lane_id, j, k);
                    fragment_c[i][j][k] += s_trans_pow2_accumulator[m] + s_orient_pow2_accumulator[n];
                }
            }
        }
        __syncthreads();

		epilogue();

        scheduler.advance_to_next_work();
    } // end of while has_work
}

# endif // COARSE_MATRIX_KERNEL_IM2COL_SPLITIMG_CUH