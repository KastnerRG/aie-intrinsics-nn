#include <adf.h>
#include "aie_api/aie.hpp"
#include "aie_api/aie_adf.hpp"
#include "matrix.h"

#define V8 8
#define V4 4

// #include <stdio.h>
// #include <stdlib.h>
// #include <type_traits>

// GemV8: 
// Main: 
// II  : 
// Lat : 

// GemV4: 
// Main: 
// II  : 
// Lat : 

void GemV8(
	input_window_int32 * __restrict in, 
    output_window_int32 * __restrict out)
{
    aie::accum<acc80, V8> acc1 (aie::zeros<acc80,V8>());
    aie::accum<acc80, V8> acc2 (aie::zeros<acc80,V8>());
    aie::vector<DTYPE,DY> m;
    aie::vector<DTYPE,V8> vx;

    for (int i=0; i < DY/V8; ++i) {
        vx = window_readincr_v8(in);
        
        for (int j=0; j < V8; ++j) {
            m = aie::load_v<DY>((DTYPE*)(matrix[(V8*i + j)%2][(V8*i + j)/2]));
            
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
            acc2 = lmac8(acc2, m, V8, 0x76543210, vx, j, 0x0);
        }
    }

    aie::vector<DTYPE, V8> vy1 = acc1.to_vector<DTYPE>();
    aie::vector<DTYPE, V8> vy2 = acc2.to_vector<DTYPE>();
    window_writeincr(out, vy1);
    window_writeincr(out, vy2);
}


void GemV4(
	input_window_int32 * __restrict in, 
    output_window_int32 * __restrict out)
{
    aie::accum<acc80, V4> acc1 (aie::zeros<acc80,V4>());
    aie::accum<acc80, V4> acc2 (aie::zeros<acc80,V4>());
    aie::accum<acc80, V4> acc3 (aie::zeros<acc80,V4>());
    aie::accum<acc80, V4> acc4 (aie::zeros<acc80,V4>());
    aie::vector<DTYPE,DY> m[Q];
    aie::vector<DTYPE,V8> vx;
    aie::vector<DTYPE, DY*2> rows;

    for (int i=0; i < DY/V8; ++i) {
        vx = window_readincr_v8(in);
        
        for (int j=0; j < V4; ++j) {
            for (int q=0; q<Q; q++)
                m[q] = aie::load_v<DY>((DTYPE*)(matrix[q][V4*i + j]));
            rows = concat(MQS);
            
            // https://www.xilinx.com/htmldocs/xilinx2022_2/aiengine_intrinsics/intrinsics/group__vect__mult__32x32.html#ga357eb0c3874047277ccd3ae7c55f4b2b
            acc1 = lmac4(
                acc1,           // v4acc80 acc

                // data buffer - general scheme
                rows,             // v32i32  xbuff       - Input buffer of 16 elements of type int32
                0,             // int     xstart      - Starting position offset applied to all lanes of input from X buffer
                0x00003210,    // uint    xoffsets    - 4b offset for each lane, applied to the x buffer. LSB apply to first lane
                DY,             // uint     xstep       - Step between each column for selection in buffer
        
                // coef buffer - general scheme
                vx,            // v8i32  zbuff       - Input buffer of 8 elements of type int32
                j*2,             // int     zstart      - Starting position offset applied to all lanes for input from Z buffer. This must be a compile time constant. Only the 4 LSB of the argument are used
                0x0,           // uint    zoffsets    - 4b offset for each lane, applied to input from Z buffer. LSB apply to first lane
                1             // int     zstep       - Step between each column for selection in zbuffer
            );

            // process second quarter of columns in m
            acc2 = lmac4(acc2, rows, V4, 0x00003210, DY, vx, j*2, 0x0, 1);

            // process third quarter of columns in m
            acc3 = lmac4(acc3, rows, V4*2, 0x00003210, DY, vx, j*2, 0x0, 1);

            // process fourth quarter of columns in m
            acc4 = lmac4(acc4, rows, V4*3, 0x00003210, DY, vx, j*2, 0x0, 1);
        }
    }

    aie::vector<DTYPE, V4> vy1 = acc1.to_vector<DTYPE>();
    aie::vector<DTYPE, V4> vy2 = acc2.to_vector<DTYPE>();
    aie::vector<DTYPE, V4> vy3 = acc3.to_vector<DTYPE>();
    aie::vector<DTYPE, V4> vy4 = acc4.to_vector<DTYPE>();
    window_writeincr(out, vy1);
    window_writeincr(out, vy2);
    window_writeincr(out, vy3);
    window_writeincr(out, vy4);
}