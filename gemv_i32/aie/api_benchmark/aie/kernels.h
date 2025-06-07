
#ifndef FUNCTION_KERNELS_H
#define FUNCTION_KERNELS_H

	void gemm(input_window_int32 * __restrict matA, input_window_int32 * __restrict matB,
						output_window_int32 * __restrict matC);


#endif