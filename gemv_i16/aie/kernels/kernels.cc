#include <adf.h>
#include "aie_api/aie.hpp"
#include "aie_api/aie_adf.hpp"
#include "matrix.h"

// #include <stdio.h>
// #include <stdlib.h>
// #include <type_traits>

// GemV: 42 clocks
// Main: 51
// II  : 86 - 3rd
// Lat : 24

void GemV(
	input_window_int16 * __restrict in, 
  output_window_int16 * __restrict out)
{
    aie::accum<acc48, DY> acc (aie::zeros<acc48,DY>());
    aie::vector<DTYPE,DY> m [Q];
    aie::vector<DTYPE,DX> vx = aie::load_v<DX>((DTYPE*)in);

    for (int i=0, id=0; i<DX; i+=Q, id+=DY) {
        for (int q=0; q<Q; q++)
            m[q] = aie::load_v<DY>((DTYPE*)matrix[q] + id);
        
        // https://www.xilinx.com/htmldocs/xilinx2022_2/aiengine_intrinsics/intrinsics/group__vect__mult__16x16.html#ga1e00ad6eedd92916e22e27f83abe5f01
        acc = mac16(
            acc,           // v16acc48 acc

            // data buffer - 16bx16b scheme
            concat(MQS),   // v32i16  xbuff       - Input buffer of 32 elements of type i16
            0,             // int     xstart      - Starting position offset applied to all lanes of input from X buffer. xstart is restricted to multiples of 2 as granularity for xbuff is 32-bit.
            0x73727170,    // uint    xoffsets    - 4b offset for each lane, corresponds to 2x the lane number and each second lane is an offset to the lane before + 1. LSB apply to first lane
            0x77767574,    // uint    xoffsets_hi - 4b offset for each lane, corresponds to 2x the lane number and each second lane is an offset to the lane before + 1. LSB apply to 8th lane
            0x3120,        // uint    xsquare     - Select order of the mini-permute square (default=0x3210). LSB apply to first element
        
            // coef buffer - general scheme
            vx,            // v16i16  zbuff       - Input buffer of 16 elements of type i16
            i,             // int     zstart      - Starting position offset applied to all lanes for input from Z buffer. This must be a compile time constant. Only the 4 LSB of the argument are used.
            0x0,           // uint    zoffsets    - 4b offset for each lane, applied to input from Z buffer. LSB apply to first lane
            0x0,           // uint    zoffsets_hi - 4b offset for each lane, applied to input from Z buffer. LSB apply to 8th lane
            1              // int     zstep       - Step between each column for selection in the zbuffer.
        );
    }

    aie::vector<DTYPE, DY> vy = acc.template to_vector<DTYPE>(0);
    aie::store_v((DTYPE*)out, vy);
}