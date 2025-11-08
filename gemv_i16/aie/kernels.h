
#ifndef FUNCTION_KERNELS_H
#define FUNCTION_KERNELS_H

void GemV_i16_mac8 (input_window_int16 * __restrict in, output_window_int16 * __restrict out);
void GemV_i16_mac16 (input_window_int16 * __restrict in, output_window_int16 * __restrict out);

#endif
