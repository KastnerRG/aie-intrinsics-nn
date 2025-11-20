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
    int num_vx = (DX + 16 - 1) / 16; 
    aie::vector<DTYPE,16> vx_arr[num_vx];

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
    
    aie::vector<DTYPE,V8> m[4];

    // calculate array size for input vector when it is larger than 16
    int num_vx = (DX + 16 - 1) / 16;
    aie::vector<DTYPE,16> vx_arr[num_vx];

    for (int i = 0; i < num_vx; i++)
        vx_arr[i] = window_readincr_v16(in);

    // works on a matrix every 8 output lane
    for (int k = 0; k < DY/V8; k++) {
        aie::accum<acc48,V8> acc (aie::zeros<acc48,V8>());
        int j = 0;
        aie::vector<DTYPE, 16> vx = vx_arr[j];

        for (int i=0; i < DX/4; i++) {
            m[0] = aie::load_v<V8>((DTYPE*)&matrix[0][i*2][8*k]);
	    m[1] = aie::load_v<V8>((DTYPE*)&matrix[1][i*2][8*k]);
	    m[2] = aie::load_v<V8>((DTYPE*)&matrix[0][i*2+1][8*k]);
	    m[3] = aie::load_v<V8>((DTYPE*)&matrix[1][i*2+1][8*k]);

	    acc = mac8(
                acc,
                concat(m[0], m[1], m[2], m[3]),
                0,
                0x33323130,
                16,
		0x3120,
		
                vx,
                0,
                0x0,
                1
            );
            if ((i+1) % 4 == 0) {
                vx = vx_arr[++j];
            } 
            else {
                vx = aie::shuffle_down_rotate(vx, 4);
            }
        }
        aie::vector<DTYPE, V8> vy = acc.to_vector<DTYPE>();
	window_writeincr(out, vy);
    }
}
