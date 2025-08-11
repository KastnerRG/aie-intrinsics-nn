#include "adf/window/types.h"

#ifndef FUNCTION_KERNELS_H
#define FUNCTION_KERNELS_H

void GemV8(
	input_window_int32 * __restrict in, 
    output_window_int32 * __restrict out);

void GemV4(
	input_window_int32 * __restrict in, 
    output_window_int32 * __restrict out);

#endif
