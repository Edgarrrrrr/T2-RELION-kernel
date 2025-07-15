#ifndef COARSE_MATRIX_KERNEL_IM2COL_SPLITIMG_BCF_PMO_NFUSE_CUH
#define COARSE_MATRIX_KERNEL_IM2COL_SPLITIMG_BCF_PMO_NFUSE_CUH

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
__global__ void CoarseMatrixKernelIm2colSplitImgBCFProjOverlapNfuseKernel(
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
	const int image_size,
	const int full_image_size = 0,
	XFLOAT *g_coor_xy = nullptr);

template <typename TParams = void>
struct CoarseMatrixKernelIm2colSplitImgBCFProjOverlapNfuse {
  const int translation_num_;
  const int orientation_num_;
  const int image_size_;
  const int full_image_size_;
  const int sm_num_;

  CoarseMatrixKernelIm2colSplitImgBCFProjOverlapNfuse(
      int translation_num, int orientation_num, int image_size, int full_image_size = 0, int sm_num = 108)
      : translation_num_(translation_num),
        orientation_num_(orientation_num),
        image_size_(image_size),
        full_image_size_(full_image_size),
        sm_num_(sm_num) {}

  // Assistant function to dispatch TParams based on translation_num_
  template <typename Func>
  auto dispatch_TParams(Func&& func) const {
    if (translation_num_ <= 32)
      return func(TypeHolder<CoarseTParam32x128_32x32>{});
    else if (translation_num_ <= 64)
      return func(TypeHolder<CoarseTParam64x128_32x64>{});
    else if (translation_num_ <= 128)
      return func(TypeHolder<CoarseTParam128x64_64x32>{});
    else
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
      CoarseScheduler<TParams::kTransBlockSize,
                      TParams::kOrientBlockSize,
                      TParams::kImgBlockSize,
                      CoarseSchedulerStrategy::InterleavedSplitK,
                      2> scheduler(translation_num_, orientation_num_, image_size_, 1, 0);
      dim3 grid(sm_num_ * BLOCKS_PER_SM, 1, 1);
      dim3 block(TParams::kBlockSize, 1, 1);
      CoarseMatrixKernelIm2colSplitImgBCFProjOverlapNfuseKernel<TParams>
	  <<<grid, block, 0, stream>>>(
          g_eulers, trans_x, trans_y, g_real, g_imag, projector, g_corr,
          g_diff2s, g_diff2s_dest, translation_num_, orientation_num_, image_size_, full_image_size_, g_coor_xy);
    } else {
      dispatch_TParams([=](auto dummy) {
        using SelectedTParams = typename decltype(dummy)::type;
        CoarseScheduler<SelectedTParams::kTransBlockSize,
                        SelectedTParams::kOrientBlockSize,
                        SelectedTParams::kImgBlockSize,
                        CoarseSchedulerStrategy::InterleavedSplitK,
                        2> scheduler(translation_num_, orientation_num_, image_size_, 1, 0);
        
		dim3 grid(sm_num_ * BLOCKS_PER_SM, 1, 1);
        dim3 block(SelectedTParams::kBlockSize, 1, 1);
        
		CoarseMatrixKernelIm2colSplitImgBCFProjOverlapNfuseKernel<SelectedTParams><<<grid, block, 0, stream>>>(
            g_eulers, trans_x, trans_y, g_real, g_imag, projector, g_corr,
            g_diff2s, g_diff2s_dest, translation_num_, orientation_num_, image_size_, full_image_size_, g_coor_xy);
      });
    }
  }

  void print_workspace_buffer(uint32_t* workspace) {
	printf("workspace buffer is empty\n");
  }
};


// construct translation matrix
// add translation_matrix_handler
template<typename TParams>
__launch_bounds__(128, BLOCKS_PER_SM)
// __launch_bounds__(128, 4)
__global__ void CoarseMatrixKernelIm2colSplitImgBCFProjOverlapNfuseKernel(
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
	const int image_size,
	const int full_image_size,
	XFLOAT *g_coor_xy) {
	static_assert(TParams::kBlockSize % 32 == 0, "kBlockSize must be a multiple of 32");
	static_assert(TParams::kBlockSize >= TParams::kTransBlockSize, "kBlockSize must be greater than or equal to kTransBlockSize");
	// static_assert(TParams::kBlockSize >= TParams::kOrientBlockSize, "kBlockSize must be greater than or equal to kOrientBlockSize");

	static_assert(TParams::kTransBlockSize % TParams::kWarpTransTileSize == 0, "kTransBlockSize must be a multiple of kWarpTransTileSize");
	static_assert(TParams::kOrientBlockSize % TParams::kWarpOrientTileSize == 0, "kOrientBlockSize must be a multiple of kWarpOrientTileSize");
	static_assert(TParams::kTransBlockSize % TParams::kMmaTransTileSize == 0, "kTransBlockSize must be a multiple of kMmaTransTileSize");
	static_assert(TParams::kOrientBlockSize % TParams::kMmaOrientTileSize == 0, "kOrientBlockSize must be a multiple of kMmaOrientTileSize");
	static_assert(TParams::kBlockSize / 32 == (TParams::kTransBlockSize / TParams::kWarpTransTileSize) * (TParams::kOrientBlockSize / TParams::kWarpOrientTileSize), "kBlockSize must be equal to the product of the number of warps in translation, orientation and image dimension");

	static_assert(TParams::kImgBlockSize == 16, "kImgBlockSize must be 16");
	static_assert(TParams::kImgBlockSize == TParams::kWarpImgTileSize, "kImgBlockSize must be equal to kWarpImgTileSize");

	const int tid = threadIdx.x;          // thread id in a block
	const int bid = blockIdx.x;           // block id in a grid
	const int warp_id  = tid / 32;        // warp id in a block
	constexpr int kWarpNum = TParams::kBlockSize / 32; // number of warps in a block
	const int lane_id  = tid % 32;        // thread id in a warp

	const int trans_block_num = (translation_num + TParams::kTransBlockSize - 1) / TParams::kTransBlockSize;
	const int orient_block_num = (orientation_num + TParams::kOrientBlockSize - 1) / TParams::kOrientBlockSize;

	int trans_block_idx = (bid % trans_block_num) * TParams::kTransBlockSize;
	int orient_block_idx = (bid / trans_block_num) * TParams::kOrientBlockSize;

	assert(blockIdx.y == 0 && blockIdx.z == 0 && threadIdx.y == 0 && threadIdx.z == 0);
	
	CoarseScheduler<TParams::kTransBlockSize, 
					TParams::kOrientBlockSize, 
					TParams::kImgBlockSize, 
					CoarseSchedulerStrategy::SplitK,
					2>
		scheduler(translation_num, orientation_num, image_size);

	__align__(16) __shared__ XFLOAT s_trans_mat_block[2 * TParams::kTransBlockSize * TParams::kImgBlockSize];
	__align__(16) __shared__ XFLOAT s_trans_mat_block_bak[2 * TParams::kTransBlockSize * TParams::kImgBlockSize];
	
	// __shared__ XFLOAT s_orient_mat_block[2 * 2 * TParams::kOrientBlockSize * TParams::kImgBlockSize];
	__align__(16) __shared__ XFLOAT s_orient_mat_block[2 * TParams::kOrientBlockSize * TParams::kImgBlockSize];

	using TransMatLayout = SharedMemorySwizzle<float, TParams::kTransBlockSize, 2 * TParams::kImgBlockSize, 0>;
	using TransRealMatLayout = SharedMemorySwizzle<float, TParams::kTransBlockSize, TParams::kImgBlockSize, 0>;
	using TransImagMatLayout = SharedMemorySwizzle<float, TParams::kTransBlockSize, TParams::kImgBlockSize, TParams::kImgBlockSize>;
	
	using OrientMatLayout = SharedMemorySwizzle<float, TParams::kOrientBlockSize, 2 * TParams::kImgBlockSize, 0>;
	using OrientRealMatLayout = SharedMemorySwizzle<float, TParams::kOrientBlockSize, TParams::kImgBlockSize, 0>;
	using OrientImagMatLayout = SharedMemorySwizzle<float, TParams::kOrientBlockSize, TParams::kImgBlockSize, TParams::kImgBlockSize>;

	TransMatLayout s_trans_mat_block_swizzle(s_trans_mat_block);
	TransRealMatLayout s_trans_real_mat_block_swizzle(s_trans_mat_block);
	TransImagMatLayout s_trans_imag_mat_block_swizzle(s_trans_mat_block);

	TransMatLayout s_trans_mat_block_swizzle_bak(s_trans_mat_block_bak);
	TransRealMatLayout s_trans_real_mat_block_swizzle_bak(s_trans_mat_block_bak);
	TransImagMatLayout s_trans_imag_mat_block_swizzle_bak(s_trans_mat_block_bak);

	OrientMatLayout s_orient_mat_block_swizzle[2] = {
		OrientMatLayout(s_orient_mat_block),
		OrientMatLayout(s_orient_mat_block)
	};

	OrientRealMatLayout s_orient_real_mat_block_swizzle[2] = {
		OrientRealMatLayout(s_orient_mat_block),
		OrientRealMatLayout(s_orient_mat_block)
	};
	OrientImagMatLayout s_orient_imag_mat_block_swizzle[2] = {
		OrientImagMatLayout(s_orient_mat_block),
		OrientImagMatLayout(s_orient_mat_block)
	};


	OrientationMatrixHandler<TParams::kOrientBlockSize,
							 TParams::kImgBlockSize,
							 kWarpNum,
							 OrientRealMatLayout,
							 OrientImagMatLayout,
							 TransMatLayout, OrientMatLayout, 
							 TParams::kTransBlockSize, TParams::kOrientBlockSize, 2 * TParams::kImgBlockSize,
							 TParams::kWarpTransTileSize, TParams::kWarpOrientTileSize, 2 * TParams::kWarpImgTileSize,
							 TParams::kMmaTransTileSize, TParams::kMmaOrientTileSize, TParams::kMmaImgTileSize>
		orientation_matrix_handler(image_size, orientation_num);

	TranslationMatrixHandler<TParams::kTransBlockSize,
							 TParams::kImgBlockSize,
							 kWarpNum,
							 TParams::kBlockSize,
							 TransMatLayout,
							 TransRealMatLayout,
							 TransImagMatLayout>
		translation_matrix_handler(image_size, translation_num);

	// double buffer for s_corr_div_2, s_coor_x, s_coor_y
	__align__(16) __shared__ XFLOAT s_corr_div_2[2][TParams::kImgBlockSize];
	__align__(16) __shared__ XFLOAT s_coor_x[2][TParams::kImgBlockSize];
	__align__(16) __shared__ XFLOAT s_coor_y[2][TParams::kImgBlockSize];

	// ============================  new  ============================
	// double buffer
	__align__(16) __shared__ XFLOAT s_fcoor_xy[2][TParams::kImgBlockSize * 2]; // img -> x,y
	__align__(16) __shared__ XFLOAT s_img_real_imag[2][TParams::kImgBlockSize * 2]; // img -> real,imag

	// For a 2D scenario, e8 is not used, so it’s not stored in shared memory.
	// e2 and e5 are also unused, but they remain in shared memory for alignment.
	__align__(16) __shared__ XFLOAT s_eulers_scaled_head[TParams::kOrientBlockSize * 4]; // (e0 e1 e2 e3)  * projector.padding_factor
	__align__(16) __shared__ XFLOAT s_eulers_scaled_tail[TParams::kOrientBlockSize * 4]; // (e4 e5 e6 e7)  * projector.padding_factor

	__align__(16) __shared__ XFLOAT s_trans_xy[TParams::kTransBlockSize * 2]; // trans_num -> x,y 

	// reduce buffer
	__align__(16) __shared__ XFLOAT s_trans_pow2_accumulator[(TParams::kBlockSize / TParams::kTransBlockSize) * TParams::kTransBlockSize];
	// __shared__ XFLOAT s_orient_pow2_accumulator[(TParams::kBlockSize / TParams::kOrientBlockSize) * TParams::kOrientBlockSize];

	__align__(16) __shared__ XFLOAT s_trans_pow2_accumulator_bak[TParams::kTransBlockSize];
	__align__(16) __shared__ XFLOAT s_orient_pow2_accumulator[TParams::kOrientBlockSize];

	// used for dummy store shared
	// __align__(16) __shared__ XFLOAT s_test_buffer[4 *  4 * 4];
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
	auto load_coord_xy = [&](int img_block_idx, XFLOAT* corr_div_2, XFLOAT* coord_x, XFLOAT* coord_y, XFLOAT* fcoor_xy, XFLOAT* img_real_imag) {
		#pragma unroll 
		for (int i = tid; i < TParams::kImgBlockSize; i += TParams::kBlockSize) {
			if (img_block_idx + i < image_size) {
				corr_div_2[i] = g_corr[img_block_idx + i] / 2;
				img_real_imag[i * 2 + 0] = g_real[img_block_idx + i];
				img_real_imag[i * 2 + 1] = g_imag[img_block_idx + i];
				int x, y;
				pixel_index2coor(img_block_idx + i, projector.imgX, projector.imgY, projector.maxR, x, y);
				coord_x[i] = x;
				coord_y[i] = y;
				if (g_coor_xy != nullptr) {
					fcoor_xy[i * 2 + 0] = g_coor_xy[(img_block_idx + i) * 2 + 0];
					fcoor_xy[i * 2 + 1] = g_coor_xy[(img_block_idx + i) * 2 + 1];
				} else {
					int x, y;
					pixel_index2coor(img_block_idx + i, projector.imgX, projector.imgY, projector.maxR, x, y);
					fcoor_xy[2 * i + 0] = x;
					fcoor_xy[2 * i + 1] = y;
				}	
			} else {
				corr_div_2[i] = 0.;
				img_real_imag[i * 2 + 0] = 0.;
				img_real_imag[i * 2 + 1] = 0.;
				fcoor_xy[2 * i + 0] = 0.;
				fcoor_xy[2 * i + 1] = 0.;
				coord_x[i] = 0;
				coord_y[i] = 0;
			}
		}
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

		// read fragment_c from g_diff2s
        init_fragment_c();

		// load eulers to smem
		#pragma unroll
		for (int i = tid; i < TParams::kOrientBlockSize; i += TParams::kBlockSize) {
			if (orient_block_idx + i < orientation_num) {
				#pragma unroll
				for (int j = 0; j < 4; j ++) {
					s_eulers_scaled_head[i * 4 + j] = g_eulers[(orient_block_idx + i) * 9 + j] * projector.padding_factor;
					s_eulers_scaled_tail[i * 4 + j] = g_eulers[(orient_block_idx + i) * 9 + 4 + j] * projector.padding_factor;
				}
			} else {
				#pragma unroll
				for (int j = 0; j < 4; j ++) {
					s_eulers_scaled_head[i * 4 + j] = 0;
					s_eulers_scaled_tail[i * 4 + j] = 0;
				}
			}
		}

		// load trans to smem
		#pragma unroll
		for (int i = tid; i < TParams::kTransBlockSize; i += TParams::kBlockSize) {
			if (trans_block_idx + i < translation_num) {
				s_trans_xy[i * 2 + 0] = trans_x[trans_block_idx + i];
				s_trans_xy[i * 2 + 1] = trans_y[trans_block_idx + i];
			} else {
				s_trans_xy[i * 2 + 0] = 0.;
				s_trans_xy[i * 2 + 1] = 0.;
			}
		}

        // initialize shared memory to zero
        for (int i = tid; i < 2 * TParams::kTransBlockSize * TParams::kImgBlockSize; i += TParams::kBlockSize) {
            s_trans_mat_block[i] = 0.0;
        }
        // for (int i = tid; i < 2 * 2 * TParams::kOrientBlockSize * TParams::kImgBlockSize; i += TParams::kBlockSize) {
        for (int i = tid; i < 2 * TParams::kOrientBlockSize * TParams::kImgBlockSize; i += TParams::kBlockSize) {
            s_orient_mat_block[i] = 0.0;
        }

        s_trans_pow2_accumulator[tid] = 0.0;
		if (g_coor_xy != nullptr && scheduler.get_current_work_k_split_block() == 0) {
			for (int i = image_size + tid; i < full_image_size; i += TParams::kBlockSize) {
				s_trans_pow2_accumulator[tid] += (g_real[i] * g_real[i] + g_imag[i] * g_imag[i]) * g_corr[i] / 2;
			}
			__syncthreads();
			// reduce
			for (int i = TParams::kBlockSize / 2; i > 0; i /= 2) {
				if (tid < i) {
					s_trans_pow2_accumulator[tid] += s_trans_pow2_accumulator[tid + i];
				}
				__syncthreads();
			}

			s_trans_pow2_accumulator[tid] = s_trans_pow2_accumulator[0];
		}
        // s_orient_pow2_accumulator[tid] = 0.0;

		if (tid < TParams::kTransBlockSize) {
			s_trans_pow2_accumulator_bak[tid] = 0.;
		}
		if (tid < TParams::kOrientBlockSize) {
			s_orient_pow2_accumulator[tid] = 0.;
		}

        for (int i = tid; i < TParams::kImgBlockSize; i += TParams::kBlockSize) {
            s_corr_div_2[0][i] = 0.0;
            s_corr_div_2[1][i] = 0.0;
        }

        __syncthreads();

/*=============================== FOR IMAGE BLOCK ==============================*/
		int k_cycle;
        while (scheduler.get_current_work_next_k_cycle(k_cycle)) {
			__syncthreads();
			if (k_cycle > scheduler.get_current_work_k_cycle_start()) {
				int img_block_idx = scheduler.get_k_block_offset_from_k_cycle(k_cycle);
				int k_cycle_mod2 = scheduler.k_cycle_mod<2>(k_cycle);
				translation_matrix_handler.construct_translation_matrix_bank_conflict_free(
					s_trans_mat_block_swizzle,
					s_trans_real_mat_block_swizzle,
					s_trans_imag_mat_block_swizzle,
					s_trans_pow2_accumulator,
					s_trans_xy,
					s_img_real_imag[k_cycle_mod2],
					s_fcoor_xy[k_cycle_mod2],
					s_corr_div_2[k_cycle_mod2],
					img_block_idx,
					trans_block_idx,
					warp_id,
					lane_id
				);
			}

			if (k_cycle < scheduler.get_current_work_k_cycle_end() - 1) {
				int img_block_idx = scheduler.get_k_block_offset_from_k_cycle(k_cycle + 1);
				int k_cycle_next_mod2 = scheduler.k_cycle_mod<2>(k_cycle + 1);
				load_coord_xy(img_block_idx, s_corr_div_2[k_cycle_next_mod2], 
							  s_coor_x[k_cycle_next_mod2], s_coor_y[k_cycle_next_mod2], 
							  s_fcoor_xy[k_cycle_next_mod2], s_img_real_imag[k_cycle_next_mod2]);
			}

			if (k_cycle > scheduler.get_current_work_k_cycle_start()) {
				int img_block_idx = scheduler.get_k_block_offset_from_k_cycle(k_cycle);
				int k_cycle_mod2 = scheduler.k_cycle_mod<2>(k_cycle);
				orientation_matrix_handler.sync_and_store_orientation_matrix_with_reduce(
					s_orient_mat_block_swizzle[k_cycle_mod2],
					s_orient_real_mat_block_swizzle[k_cycle_mod2],
					s_orient_imag_mat_block_swizzle[k_cycle_mod2],
					s_orient_pow2_accumulator,
					s_corr_div_2[k_cycle_mod2],
					warp_id,
					lane_id
				);
			}

			__syncthreads();
			// __threadfence_block();
			if (k_cycle > scheduler.get_current_work_k_cycle_start() && 
				k_cycle < scheduler.get_current_work_k_cycle_end() - 1) {

				int img_block_idx = scheduler.get_k_block_offset_from_k_cycle(k_cycle);
				int k_cycle_mod2 = scheduler.k_cycle_mod<2>(k_cycle);
				int k_cycle_next_mod2 = scheduler.k_cycle_mod<2>(k_cycle + 1);

				orientation_matrix_handler.process_and_prefetch_orientation_matrix(
					projector,
					s_eulers_scaled_head,
					s_eulers_scaled_tail,
					s_fcoor_xy[k_cycle_next_mod2],
					img_block_idx,
					orient_block_idx,
					warp_id,
					lane_id
				);
				/*=============================== COMPUTE CROSS TERM ==============================*/
				block_mma_tf32_sim_fp32<TransMatLayout, OrientMatLayout, 
				TParams::kTransBlockSize, TParams::kOrientBlockSize, 2 * TParams::kImgBlockSize,
				TParams::kWarpTransTileSize, TParams::kWarpOrientTileSize, 2 * TParams::kWarpImgTileSize,
				TParams::kMmaTransTileSize, TParams::kMmaOrientTileSize, TParams::kMmaImgTileSize>(
					fragment_c, s_trans_mat_block_swizzle, s_orient_mat_block_swizzle[k_cycle_mod2], warp_id, lane_id);
			}

			if (k_cycle == scheduler.get_current_work_k_cycle_start()) {
				int img_block_idx = scheduler.get_k_block_offset_from_k_cycle(k_cycle + 1);
				int k_cycle_next_mod2 = scheduler.k_cycle_mod<2>(k_cycle + 1);
				orientation_matrix_handler.process_and_prefetch_orientation_matrix(
					projector,
					s_eulers_scaled_head,
					s_eulers_scaled_tail,
					s_fcoor_xy[k_cycle_next_mod2],
					img_block_idx,
					orient_block_idx,
					warp_id,
					lane_id
				);
			}

			if (k_cycle == scheduler.get_current_work_k_cycle_end() - 1) {
				int img_block_idx = scheduler.get_k_block_offset_from_k_cycle(k_cycle);
				int k_cycle_mod2 = scheduler.k_cycle_mod<2>(k_cycle);
				/*=============================== COMPUTE CROSS TERM ==============================*/
				block_mma_tf32_sim_fp32<TransMatLayout, OrientMatLayout, 
				TParams::kTransBlockSize, TParams::kOrientBlockSize, 2 * TParams::kImgBlockSize,
				TParams::kWarpTransTileSize, TParams::kWarpOrientTileSize, 2 * TParams::kWarpImgTileSize,
				TParams::kMmaTransTileSize, TParams::kMmaOrientTileSize, TParams::kMmaImgTileSize>(
					fragment_c, s_trans_mat_block_swizzle, s_orient_mat_block_swizzle[k_cycle_mod2], warp_id, lane_id);
			}
        } // end of image block

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

    /*=============================== WRITE BACK ==============================*/
		epilogue();

        scheduler.advance_to_next_work();
    } // end of while has_work
}

#endif // COARSE_MATRIX_KERNEL_IM2COL_SPLITIMG_BCF_PMO_NFUSE_CUH