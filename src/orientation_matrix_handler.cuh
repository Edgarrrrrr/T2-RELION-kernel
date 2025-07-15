#ifndef ORIENT_MATRIX_HANDLER_CUH
#define ORIENT_MATRIX_HANDLER_CUH

#include "./acc_projectorkernel_impl.h"

#include "./warp_layout.cuh"
#include "./reg_bitmap.cuh"
#include "./mma_utils.cuh"

#include <assert.h>
#include <cuda_runtime.h>
#include <stdint.h>
#include <stdio.h>
#include <type_traits>

template <int kOrientBlockSize, int kImgBlockSize, int kWarpNum,
			typename OrientMatLayoutReal, typename OrientMatLayoutImag,
			typename SmemLayoutA, typename SmemLayoutB, int kMblock, int kNblock,
			int kKblock, int kMwarp, int kNwarp, int kKwarp, int kMmma, int kNmma,
			int kKmma>
struct OrientationMatrixHandler {

	// z-order
	// row: 8 orientation block
	// col: 4 img block
	/**                                  <-4->
	 *    +----------------+   -----   ^ +---+---+---+---+
	 *    |       w0       |           1 |  0|  2|  4|  6|
	 *    +----------------+   \       v +---+---+---+---+
	 *    |       w1       |    \        |  1|  3|  5|  7|
	 *    +----------------+     \       +---+---+---+---+
	 *    |       w2       |      \      |  8| 10| 12| 14|
	 *    +----------------+       \     +---+---+---+---+
	 *    |       w3       |             |  9| 11| 13| 15|
	 *    +----------------+             +---+---+---+---+
	 *    |       w0       |             | 16| 18| 20| 22|
	 *    +----------------+             +---+---+---+---+
	 *    |       w1       |             | 17| 19| 21| 23|
	 *           ...                     +---+---+---+---+
     *                                   | 24| 26| 28| 30|
     *                                   +---+---+---+---+
	 *                                   | 25| 27| 29| 31|
     *                                   +---+---+---+---+
     */
    // using WarpLayout<4, 2, LayoutMajorType::ColumnMajor> OrientWarpLayout;
    using OrientWarpLayout = WarpLayout<4, 2, LayoutMajorType::ColumnMajor>;
    // WarpLayout orient_warp_layout;

    static const int kOMImgPerThread = 4;
    static const int kOMOrientPerThread = 1;

    static const int kOMOrientPerWarpTile = 8;
    static const int kOMImgPerWarp = 4 * 4;

    static_assert(kOMOrientPerWarpTile == OrientWarpLayout::rows * kOMOrientPerThread, 
        "kOMOrientPerWarpTile must be equal to OrientWarpLayout::rows");
    static_assert(kOMImgPerWarp == OrientWarpLayout::cols * kOMImgPerThread, 
        "kOMImgPerWarp must be equal to OrientWarpLayout::cols");

    static const int kOMNumOrientWarpTile = kOrientBlockSize / kOMOrientPerWarpTile;
    static const int kOMNumOrientWarpTilePerWarp = kOMNumOrientWarpTile / kWarpNum;

    // [real/imag] [kOMNumOrientWarpTilePerWarp] [kOMImgPerThread]
	//  2           x                             4          
    float reg_tex_buf_[2][kOMNumOrientWarpTilePerWarp][kOMImgPerThread];
    RegBitmap<kOMNumOrientWarpTilePerWarp * kOMImgPerThread> flag_minus_;
    const int& image_size_;
    const int& orientation_num_;

    // constructor
    __device__ __forceinline__
    OrientationMatrixHandler(const int& img_size, const int& orient_num) : 
      image_size_(img_size), orientation_num_(orient_num) {
        flag_minus_.clear_all();
    }

    __device__ __forceinline__
    void process_and_prefetch_orientation_matrix(
        const AccProjectorKernel& projector,
        const float* s_eulers_scaled_head, // size: kOrientBlockSize * 4
        const float* s_eulers_scaled_tail, // size: kOrientBlockSize * 4
        const float* s_fcoor_xy,    // size: kImgBlockSize * 2
        const int img_block_idx,
        const int orient_block_idx,
        const int warp_id,
        const int lane_id
    ) {
        __align__(16) XFLOAT reg_fcoor_xy[kOMImgPerThread][2]; // 4 x 2 (x, y)
		__align__(16) XFLOAT reg_eulers[kOMNumOrientWarpTilePerWarp][8]; // x x 8
		
		const XFLOAT reg_minus_mdl_init_y_plus_dot_5 = - projector.mdlInitY + 0.5f;
		const XFLOAT reg_minus_mdl_init_z_plus_dot_5 = - projector.mdlInitZ + 0.5f;
	
		// load coor
		// step is 2, for 2(step) x 2(x,y) = 4 (float4)
		#pragma unroll
		for (int i = 0; i < kOMImgPerThread; i += 2) {
			// float1 idx : (get_col_idx * kOMImgPerThread + i) * 2 (x,y)
			int s_img_float4_idx = (OrientWarpLayout::get_col_idx(lane_id) * kOMImgPerThread + i) / 2;
			
			assert(i < kOMImgPerThread);
			assert(s_img_float4_idx * 4 < kImgBlockSize * 2);
			*reinterpret_cast<float4*>(&reg_fcoor_xy[i][0])
				= reinterpret_cast<const float4*>(s_fcoor_xy)[s_img_float4_idx];
		}

		
		// each ld128 will only need 2 transaction without bankconflict
		// load eulers
		#pragma unroll
		for (int i = 0; i < kOMNumOrientWarpTilePerWarp; i ++) {
			int s_orient_idx = (i * kWarpNum + warp_id) * OrientWarpLayout::rows + OrientWarpLayout::get_row_idx(lane_id);
			// load float4
			*reinterpret_cast<float4*>(&reg_eulers[i][0])
				= (reinterpret_cast<const float4*>(s_eulers_scaled_head)) [s_orient_idx];
			*reinterpret_cast<float4*>(&reg_eulers[i][4])
				= (reinterpret_cast<const float4*>(s_eulers_scaled_tail)) [s_orient_idx];

			#pragma unroll
            for (int j = 0; j < kOMImgPerThread; j ++) {
				int g_img_idx = img_block_idx + OrientWarpLayout::get_col_idx(lane_id) * kOMImgPerThread + j;

				XFLOAT& x = reg_fcoor_xy[j][0];
				XFLOAT& y = reg_fcoor_xy[j][1];
				XFLOAT& e0 = reg_eulers[i][0];
				XFLOAT& e1 = reg_eulers[i][1];
				XFLOAT& e3 = reg_eulers[i][3];
				XFLOAT& e4 = reg_eulers[i][4];
				XFLOAT& e6 = reg_eulers[i][6];
				XFLOAT& e7 = reg_eulers[i][7];

				XFLOAT yp = (e3 * x + e4 * y );
				XFLOAT xp = (e0 * x + e1 * y );
				XFLOAT zp = (e6 * x + e7 * y );
				int r2 = xp*xp + yp*yp + zp*zp;
				bool r2_within_bounds = r2 <= projector.maxR2_padded;
				bool xp_neg = xp < 0;
				flag_minus_.set(i * kOMImgPerThread + j, xp_neg);
				
				XFLOAT reg_sign_digit = (xp_neg) ? -1. : 1.;
				xp = xp * reg_sign_digit + 0.5;
				yp = yp * reg_sign_digit + reg_minus_mdl_init_y_plus_dot_5;
				zp = zp * reg_sign_digit + reg_minus_mdl_init_z_plus_dot_5;
				reg_tex_buf_[0][i][j] = r2_within_bounds ? tex3D<XFLOAT>(projector.mdlReal, xp, yp, zp) : 0;
				reg_tex_buf_[1][i][j] = r2_within_bounds ? tex3D<XFLOAT>(projector.mdlImag, xp, yp, zp) : 0;
			}
		}
    }

    // Sync and store orientation matrix with reduction from registers
    // template<typename OrientMatLayoutReal, typename OrientMatLayoutImag>
    __device__ __forceinline__
    void sync_and_store_orientation_matrix_with_reduce(
        // out
		SmemLayoutB s_orient_mat_block_swizzle,
        OrientMatLayoutReal s_orient_real_mat_block_swizzle,
        OrientMatLayoutImag s_orient_imag_mat_block_swizzle,
        float* s_orient_pow2_accumulator, // size: kOrientBlockSize
        // in
        float* s_corr_div_2, // size: kImgBlockSize
        int warp_id,
        int lane_id
    ) {

        __align__(16) float reg_corr_div_2[kOMImgPerThread];

        // load corr_div_2
        assert(OrientWarpLayout::get_col_idx(lane_id) * kOMImgPerThread < kImgBlockSize);
		
        *reinterpret_cast<float4*>(&reg_corr_div_2[0])
            = *reinterpret_cast<const float4*>(&s_corr_div_2[OrientWarpLayout::get_col_idx(lane_id) * kOMImgPerThread]);
        
        // Negate the imaginary component if xp_neg was set earlier, 
        // ensuring correct Hermitian symmetry.
        #pragma unroll
        for (int i = 0; i < kOMNumOrientWarpTilePerWarp; i ++) {
            #pragma unroll
			for (int j = 0; j < kOMImgPerThread; j ++) {
                if (flag_minus_.get(i * kOMImgPerThread + j)) {
                    reg_tex_buf_[1][i][j] = -reg_tex_buf_[1][i][j];
                }
			}
		}
        // Clear the flag
        flag_minus_.clear_all();

        // Store orientation matrix to smem
        #pragma unroll
		for (int i = 0; i < kOMNumOrientWarpTilePerWarp; i ++) {
			
			if ((lane_id & 0x2) != 0) {
				// swap
				float tmp;
				for (int j = 0; j < 4; j ++) {
					tmp = reg_tex_buf_[0][i][j];
					reg_tex_buf_[0][i][j] = reg_tex_buf_[1][i][j];
					reg_tex_buf_[1][i][j] = tmp; 
				}
			}

			int s_orient_idx = (i * kWarpNum + warp_id) * OrientWarpLayout::rows + OrientWarpLayout::get_row_idx(lane_id);
			int s_img_idx1, s_img_idx2;
			
			if ((lane_id & 0x2) == 0) {
				s_img_idx1 = OrientWarpLayout::get_col_idx(lane_id) * kOMImgPerThread;
				s_img_idx2 = s_img_idx1 + 16;
			}
			else {
				s_img_idx2 = OrientWarpLayout::get_col_idx(lane_id) * kOMImgPerThread;
				s_img_idx1 = s_img_idx2 + 16;
			}

			*reinterpret_cast<float4*>(
					&s_orient_mat_block_swizzle(s_orient_idx, s_img_idx1)) = *reinterpret_cast<float4*>(&reg_tex_buf_[0][i][0]);
			*reinterpret_cast<float4*>(
					&s_orient_mat_block_swizzle(s_orient_idx, s_img_idx2)) = *reinterpret_cast<float4*>(&reg_tex_buf_[1][i][0]);
		}

        // image reduction
        #pragma unroll
        for (int i = 0; i < kOMNumOrientWarpTilePerWarp; i ++) {
            float magnitude_squared_sum = 0.;
            #pragma unroll
            for (int j = 0; j < kOMImgPerThread; j ++) {
                magnitude_squared_sum += (reg_tex_buf_[0][i][j] * reg_tex_buf_[0][i][j] 
                                      +   reg_tex_buf_[1][i][j] * reg_tex_buf_[1][i][j]) 
                                      * reg_corr_div_2[j];
            }
            magnitude_squared_sum = OrientWarpLayout::reduce_by_rows(magnitude_squared_sum);

            int s_orient_idx = (i * kWarpNum + warp_id) * OrientWarpLayout::rows + OrientWarpLayout::get_row_idx(lane_id);
            // after reduction, only the lane in the first column will store the result
            if (OrientWarpLayout::get_col_idx(lane_id) == 0) {
                s_orient_pow2_accumulator[s_orient_idx] += magnitude_squared_sum;
            }
        }
    }

	__device__ __forceinline__
	void construct_orientation_matrix(
		// out
		SmemLayoutB s_orient_mat_block_swizzle,
		OrientMatLayoutReal s_orient_real_mat_block_swizzle,
		OrientMatLayoutImag s_orient_imag_mat_block_swizzle,
		float* s_orient_pow2_accumulator, // size: kOrientBlockSize
		// in
		const AccProjectorKernel& projector,
        const float* s_eulers_scaled_head, // size: kOrientBlockSize * 4
        const float* s_eulers_scaled_tail, // size: kOrientBlockSize * 4
        const float* s_fcoor_xy,    // size: kImgBlockSize * 2
		const float* s_corr_div_2, // size: kImgBlockSize
        const int img_block_idx,
        const int orient_block_idx,
        const int warp_id,
        const int lane_id
    ) {
        XFLOAT reg_fcoor_xy[kOMImgPerThread][2]; // 4 x 2 (x, y)
		XFLOAT reg_eulers[kOMNumOrientWarpTilePerWarp][8]; // x x 8
		
		const XFLOAT reg_minus_mdl_init_y_plus_dot_5 = - projector.mdlInitY + 0.5f;
		const XFLOAT reg_minus_mdl_init_z_plus_dot_5 = - projector.mdlInitZ + 0.5f;
	
		// load coor
		// step is 2, for 2(step) x 2(x,y) = 4 (float4)
		#pragma unroll
		for (int i = 0; i < kOMImgPerThread; i += 2) {
			// float1 idx : (get_col_idx * kOMImgPerThread + i) * 2 (x,y)
			int s_img_float4_idx = (OrientWarpLayout::get_col_idx(lane_id) * kOMImgPerThread + i) / 2;
			
			assert(i < kOMImgPerThread);
			assert(s_img_float4_idx * 4 < kImgBlockSize * 2);
			*reinterpret_cast<float4*>(&reg_fcoor_xy[i][0])
				= reinterpret_cast<const float4*>(s_fcoor_xy)[s_img_float4_idx];
		}

		
		// each ld128 will only need 2 transaction without bankconflict
		// load eulers
		#pragma unroll
		for (int i = 0; i < kOMNumOrientWarpTilePerWarp; i ++) {
			int s_orient_idx = (i * kWarpNum + warp_id) * OrientWarpLayout::rows + OrientWarpLayout::get_row_idx(lane_id);
			// int g_orient_idx = orient_block_idx + s_orient_idx;
			// load float4
			*reinterpret_cast<float4*>(&reg_eulers[i][0])
				= reinterpret_cast<const float4*>(s_eulers_scaled_head)[s_orient_idx];
			*reinterpret_cast<float4*>(&reg_eulers[i][4])
				= reinterpret_cast<const float4*>(s_eulers_scaled_tail)[s_orient_idx];

			// #pragma unroll
			#pragma unroll
            for (int j = 0; j < kOMImgPerThread; j ++) {
				// int g_img_idx = img_block_idx + OrientWarpLayout::get_col_idx(lane_id) * kOMImgPerThread + j;
				// bool within_bounds = g_img_idx < image_size_ && g_orient_idx < orientation_num_;

				XFLOAT& x = reg_fcoor_xy[j][0];
				XFLOAT& y = reg_fcoor_xy[j][1];
				XFLOAT& e0 = reg_eulers[i][0];
				XFLOAT& e1 = reg_eulers[i][1];
				XFLOAT& e3 = reg_eulers[i][3];
				XFLOAT& e4 = reg_eulers[i][4];
				XFLOAT& e6 = reg_eulers[i][6];
				XFLOAT& e7 = reg_eulers[i][7];

				XFLOAT yp = (e3 * x + e4 * y );
				XFLOAT xp = (e0 * x + e1 * y );
				XFLOAT zp = (e6 * x + e7 * y );
				int r2 = xp*xp + yp*yp + zp*zp;
				bool r2_within_bounds = r2 <= projector.maxR2_padded;
				bool xp_neg = xp < 0;
				
				XFLOAT reg_sign_digit = (xp_neg) ? -1. : 1.;
				xp = xp * reg_sign_digit + 0.5;
				yp = yp * reg_sign_digit + reg_minus_mdl_init_y_plus_dot_5;
				zp = zp * reg_sign_digit + reg_minus_mdl_init_z_plus_dot_5;
				reg_tex_buf_[0][i][j] = r2_within_bounds ? tex3D<XFLOAT>(projector.mdlReal, xp, yp, zp) : 0;
				reg_tex_buf_[1][i][j] = r2_within_bounds ? tex3D<XFLOAT>(projector.mdlImag, xp, yp, zp) : 0;
				
				reg_tex_buf_[1][i][j] = xp_neg ? -reg_tex_buf_[1][i][j] : reg_tex_buf_[1][i][j];
			}
		}

        float reg_corr_div_2[kOMImgPerThread];

        // load corr_div_2
        assert(OrientWarpLayout::get_col_idx(lane_id) * kOMImgPerThread < kImgBlockSize);
		
        *reinterpret_cast<float4*>(&reg_corr_div_2[0])
            = *reinterpret_cast<const float4*>(&s_corr_div_2[OrientWarpLayout::get_col_idx(lane_id) * kOMImgPerThread]);
        
        // Store orientation matrix to smem
        #pragma unroll
		for (int i = 0; i < kOMNumOrientWarpTilePerWarp; i ++) {
			#pragma unroll
            for (int j = 0; j < kOMImgPerThread; j ++) {
				int s_orient_idx = (i * kWarpNum + warp_id) * OrientWarpLayout::rows + OrientWarpLayout::get_row_idx(lane_id);
				int s_img_idx = OrientWarpLayout::get_col_idx(lane_id) * kOMImgPerThread + j;
                
                assert(s_orient_idx < kOrientBlockSize);
                assert(s_img_idx < kImgBlockSize);

				s_orient_real_mat_block_swizzle(s_orient_idx, s_img_idx) = reg_tex_buf_[0][i][j];
				s_orient_imag_mat_block_swizzle(s_orient_idx, s_img_idx) = reg_tex_buf_[1][i][j];
			}
		}

        // image reduction
        #pragma unroll
        for (int i = 0; i < kOMNumOrientWarpTilePerWarp; i ++) {
            float magnitude_squared_sum = 0.;
            #pragma unroll
            for (int j = 0; j < kOMImgPerThread; j ++) {
                magnitude_squared_sum += (reg_tex_buf_[0][i][j] * reg_tex_buf_[0][i][j] 
                                      +   reg_tex_buf_[1][i][j] * reg_tex_buf_[1][i][j]) 
                                      * reg_corr_div_2[j];
				// magnitude_squared_sum += 1;
            }
            magnitude_squared_sum = OrientWarpLayout::reduce_by_rows(magnitude_squared_sum);

            int s_orient_idx = (i * kWarpNum + warp_id) * OrientWarpLayout::rows + OrientWarpLayout::get_row_idx(lane_id);
            // after reduction, only the lane in the first column will store the result
            if (OrientWarpLayout::get_col_idx(lane_id) == 0) {
                s_orient_pow2_accumulator[s_orient_idx] += magnitude_squared_sum;
            }
        }

    }

	
	__device__ __forceinline__
	void construct_orientation_matrix_bank_conflict_free(
		// out
		SmemLayoutB s_orient_mat_block_swizzle,
		OrientMatLayoutReal s_orient_real_mat_block_swizzle,
		OrientMatLayoutImag s_orient_imag_mat_block_swizzle,
		float* s_orient_pow2_accumulator, // size: kOrientBlockSize
		// in
		const AccProjectorKernel& projector,
        const float* s_eulers_scaled_head, // size: kOrientBlockSize * 4
        const float* s_eulers_scaled_tail, // size: kOrientBlockSize * 4
        const float* s_fcoor_xy,    // size: kImgBlockSize * 2
		const float* s_corr_div_2, // size: kImgBlockSize
        const int img_block_idx,
        const int orient_block_idx,
        const int warp_id,
        const int lane_id
    ) {
        XFLOAT reg_fcoor_xy[kOMImgPerThread][2]; // 4 x 2 (x, y)
		XFLOAT reg_eulers[kOMNumOrientWarpTilePerWarp][8]; // x x 8
		
		const XFLOAT reg_minus_mdl_init_y_plus_dot_5 = - projector.mdlInitY + 0.5f;
		const XFLOAT reg_minus_mdl_init_z_plus_dot_5 = - projector.mdlInitZ + 0.5f;
	
		// load coor
		// step is 2, for 2(step) x 2(x,y) = 4 (float4)
		#pragma unroll
		for (int i = 0; i < kOMImgPerThread; i += 2) {
			// float1 idx : (get_col_idx * kOMImgPerThread + i) * 2 (x,y)
			int s_img_float4_idx = (OrientWarpLayout::get_col_idx(lane_id) * kOMImgPerThread + i) / 2;
			
			assert(i < kOMImgPerThread);
			assert(s_img_float4_idx * 4 < kImgBlockSize * 2);
			*reinterpret_cast<float4*>(&reg_fcoor_xy[i][0])
				= reinterpret_cast<const float4*>(s_fcoor_xy)[s_img_float4_idx];
		}

		
		// each ld128 will only need 2 transaction without bankconflict
		// load eulers
		#pragma unroll
		for (int i = 0; i < kOMNumOrientWarpTilePerWarp; i ++) {
			int s_orient_idx = (i * kWarpNum + warp_id) * OrientWarpLayout::rows + OrientWarpLayout::get_row_idx(lane_id);
			// int g_orient_idx = orient_block_idx + s_orient_idx;
			// load float4
			*reinterpret_cast<float4*>(&reg_eulers[i][0])
				= reinterpret_cast<const float4*>(s_eulers_scaled_head)[s_orient_idx];
			*reinterpret_cast<float4*>(&reg_eulers[i][4])
				= reinterpret_cast<const float4*>(s_eulers_scaled_tail)[s_orient_idx];

			// #pragma unroll
			#pragma unroll
            for (int j = 0; j < kOMImgPerThread; j ++) {
				// int g_img_idx = img_block_idx + OrientWarpLayout::get_col_idx(lane_id) * kOMImgPerThread + j;
				// bool within_bounds = g_img_idx < image_size_ && g_orient_idx < orientation_num_;

				XFLOAT& x = reg_fcoor_xy[j][0];
				XFLOAT& y = reg_fcoor_xy[j][1];
				XFLOAT& e0 = reg_eulers[i][0];
				XFLOAT& e1 = reg_eulers[i][1];
				XFLOAT& e3 = reg_eulers[i][3];
				XFLOAT& e4 = reg_eulers[i][4];
				XFLOAT& e6 = reg_eulers[i][6];
				XFLOAT& e7 = reg_eulers[i][7];

				XFLOAT yp = (e3 * x + e4 * y );
				XFLOAT xp = (e0 * x + e1 * y );
				XFLOAT zp = (e6 * x + e7 * y );
				int r2 = xp*xp + yp*yp + zp*zp;
				bool r2_within_bounds = r2 <= projector.maxR2_padded;
				bool xp_neg = xp < 0;
				
				XFLOAT reg_sign_digit = (xp_neg) ? -1. : 1.;
				xp = xp * reg_sign_digit + 0.5;
				yp = yp * reg_sign_digit + reg_minus_mdl_init_y_plus_dot_5;
				zp = zp * reg_sign_digit + reg_minus_mdl_init_z_plus_dot_5;
				reg_tex_buf_[0][i][j] = r2_within_bounds ? tex3D<XFLOAT>(projector.mdlReal, xp, yp, zp) : 0;
				reg_tex_buf_[1][i][j] = r2_within_bounds ? tex3D<XFLOAT>(projector.mdlImag, xp, yp, zp) : 0;
				
				reg_tex_buf_[1][i][j] = xp_neg ? -reg_tex_buf_[1][i][j] : reg_tex_buf_[1][i][j];
			}
		}

        float reg_corr_div_2[kOMImgPerThread];

        // load corr_div_2
        assert(OrientWarpLayout::get_col_idx(lane_id) * kOMImgPerThread < kImgBlockSize);
		
        *reinterpret_cast<float4*>(&reg_corr_div_2[0])
            = *reinterpret_cast<const float4*>(&s_corr_div_2[OrientWarpLayout::get_col_idx(lane_id) * kOMImgPerThread]);

        // Store orientation matrix to smem
        #pragma unroll
		for (int i = 0; i < kOMNumOrientWarpTilePerWarp; i ++) {
			if ((lane_id & 0x2) != 0) {
				// swap
				float tmp;
				for (int j = 0; j < 4; j ++) {
					tmp = reg_tex_buf_[0][i][j];
					reg_tex_buf_[0][i][j] = reg_tex_buf_[1][i][j];
					reg_tex_buf_[1][i][j] = tmp; 
				}
			}

			int s_orient_idx = (i * kWarpNum + warp_id) * OrientWarpLayout::rows + OrientWarpLayout::get_row_idx(lane_id);
			int s_img_idx1, s_img_idx2;
			
			if ((lane_id & 0x2) == 0) {
				s_img_idx1 = OrientWarpLayout::get_col_idx(lane_id) * kOMImgPerThread;
				s_img_idx2 = s_img_idx1 + 16;
			}
			else {
				s_img_idx2 = OrientWarpLayout::get_col_idx(lane_id) * kOMImgPerThread;
				s_img_idx1 = s_img_idx2 + 16;
			}

			*reinterpret_cast<float4*>(
					&s_orient_mat_block_swizzle(s_orient_idx, s_img_idx1)) = *reinterpret_cast<float4*>(&reg_tex_buf_[0][i][0]);
			*reinterpret_cast<float4*>(
					&s_orient_mat_block_swizzle(s_orient_idx, s_img_idx2)) = *reinterpret_cast<float4*>(&reg_tex_buf_[1][i][0]);
		}

        // image reduction
        #pragma unroll
        for (int i = 0; i < kOMNumOrientWarpTilePerWarp; i ++) {
            float magnitude_squared_sum = 0.;
            #pragma unroll
            for (int j = 0; j < kOMImgPerThread; j ++) {
                magnitude_squared_sum += (reg_tex_buf_[0][i][j] * reg_tex_buf_[0][i][j] 
                                      +   reg_tex_buf_[1][i][j] * reg_tex_buf_[1][i][j]) 
                                      * reg_corr_div_2[j];
            }
            magnitude_squared_sum = OrientWarpLayout::reduce_by_rows(magnitude_squared_sum);

            int s_orient_idx = (i * kWarpNum + warp_id) * OrientWarpLayout::rows + OrientWarpLayout::get_row_idx(lane_id);
            // after reduction, only the lane in the first column will store the result
            if (OrientWarpLayout::get_col_idx(lane_id) == 0) {
                s_orient_pow2_accumulator[s_orient_idx] += magnitude_squared_sum;
            }
        }

    }
};



#endif // ORIENT_MATRIX_HANDLER_CUH