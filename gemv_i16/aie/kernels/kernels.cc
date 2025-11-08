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

#define V8 8
#define V16 16


void GemV_i16_mac16 (input_window_int16 * __restrict in, output_window_int16 * __restrict out) {

    aie::vector<DTYPE,V16> m [Q];

    // calculate array size for input vector when it is larger than 16
    int num_vx = (DX + V16 - 1) / V16; 
    aie::vector<DTYPE,V16> vx_arr[num_vx];

    for (int i = 0; i < num_vx; i++)
        vx_arr[i] = window_readincr_v16(in);


    // works on a matrix every 16 output lane
    for (int k = 0; k < DY/V16; k++) {
	aie::accum<acc48,V16> acc (aie::zeros<acc48,V16>());
        int j = 0;
	aie::vector<DTYPE, V16> vx = vx_arr[j];

	for (int i=0; i < DX/2; i++) {
	    for (int q=0; q<Q; q++) { 
                m[q] = aie::load_v<V16>((DTYPE*)&matrix[q][i][16*k]);
            }
	    acc = mac16(
	        acc,
	        concat(MQS),
	        0,
		0x73727170,
		0x77767574,
		0x3120,

	    	vx,
	        0,
		0x0,
		0x0,
		1
            );
	    if ((i+1) % 8 == 0) { 
	        vx = vx_arr[++j];
	    }
	    else {
	        vx = aie::shuffle_down_rotate(vx, 2);
	    }
	}
        aie::vector<DTYPE, V16> vy = acc.to_vector<DTYPE>();
        window_writeincr(out, vy);
    }   
} 


void GemV_i16_mac8 (input_window_int16 * __restrict in, output_window_int16 * __restrict out) {
/*    aie::accum<acc48, V8> acc (aie::zeros<acc48,V8>());
    aie::vector<DTYPE,V8> m [Q];
    aie::vector<DTYPE,16> vx = window_readincr_v16(in);

    for (int i=0, id=0; i<DX; i+=Q, id+=DY) {
        for (int q=0; q<Q; q++)
            m[q] = aie::load_v<V8>((DTYPE*)matrix[q] + id);

        // https://www.xilinx.com/htmldocs/xilinx2022_2/aiengine_intrinsics/intrinsics/group__vect__mult__16x16.html#gac7a2f861000ea79918c5dc662f9be71d
        acc = mac8(
            acc,           // v8acc48 acc

            // data buffer - 16bx16b scheme
            concat(MQS),   // v32i16  xbuff       - Input buffer of 32 elements of type i16
            0,             // int     xstart      - Starting position offset applied to all lanes of input from X buffer. xstart is restricted to multiples of 2 as granularity for xbuff is 32-bit.
            0x33323130,    // uint    xoffsets    - 4b offset for each lane, corresponds to 2x the lane number and each second lane is an offset to the lane before + 1. LSB apply to first lane
            16,             // int     xstep
            0x3120,        // uint    xsquare     - Select order of the mini-permute square (default=0x3210). LSB apply to first element

            // coef buffer - general scheme
            vx,            // v16i16  zbuff       - Input buffer of 16 elements of type i16
            i,             // int     zstart      - Starting position offset applied to all lanes for input from Z buffer. This must be a compile time constant. Only the 4 LSB of the argument are used.
            0x0,           // uint    zoffsets    - 4b offset for each lane, applied to input from Z buffer. LSB apply to first lane
	    1              // int     zstep       - Step between each column for selection in the zbuffer.
        );
    }

    aie::vector<DTYPE, DY> vy = acc.to_vector<DTYPE>();
    window_writeincr(out, vy);*/
}
