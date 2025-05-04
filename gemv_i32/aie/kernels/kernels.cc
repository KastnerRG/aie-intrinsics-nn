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
	input_window_int32 * __restrict in, 
    output_window_int32 * __restrict out)
{
    aie::accum<acc80, DV> acc1 (aie::zeros<acc80,DV>());
    aie::accum<acc80, DV> acc2 (aie::zeros<acc80,DV>());
    aie::vector<DTYPE,DY> m;
    aie::vector<DTYPE,DV> vx;

    for (int i=0; i < DY/DV; ++i) {
        vx = window_readincr_v8(in);
        
        for (int j=0; j < DV; ++j) {
            m = aie::load_v<DY>((DTYPE*)(matrix[(DV*i + j)%2][(DV*i + j)/2]));
            
            // https://www.xilinx.com/htmldocs/xilinx2022_2/aiengine_intrinsics/intrinsics/group__vect__mult__32x32.html#ga357eb0c3874047277ccd3ae7c55f4b2b
            acc1 = lmac8(
                acc1,           // v8acc80 acc

                // data buffer - general scheme
                m,             // v16i32  xbuff       - Input buffer of 16 elements of type int32
                0,             // int     xstart      - Starting position offset applied to all lanes of input from X buffer
                0x76543210,    // uint    xoffsets    - 4b offset for each lane, applied to the x buffer. LSB apply to first lane
        
                // coef buffer - general scheme
                vx,            // v8i32  zbuff       - Input buffer of 8 elements of type int32
                j,             // int     zstart      - Starting position offset applied to all lanes for input from Z buffer. This must be a compile time constant. Only the 4 LSB of the argument are used
                0x0           // uint    zoffsets    - 4b offset for each lane, applied to input from Z buffer. LSB apply to first lane
            );

            // process second half of columns in m
            acc2 = lmac8(acc2, m, DV, 0x76543210, vx, j, 0x0);
        }
    }

    aie::vector<DTYPE, DV> vy1 = acc1.to_vector<DTYPE>();
    aie::vector<DTYPE, DV> vy2 = acc2.to_vector<DTYPE>();
    window_writeincr(out, vy1);
    window_writeincr(out, vy2);
}