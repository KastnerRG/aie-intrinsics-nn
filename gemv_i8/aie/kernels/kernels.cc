#include <adf.h>
#include <cstdio>
#include "adf/window/window.h"
#include "aie_api/aie.hpp"
#include "aie_api/aie_adf.hpp"
#include "matrix.h"


void GemV8(
	input_window_int8 * __restrict in, 
  output_window_int16 * __restrict out)
{
    aie::accum<acc48, 8> acc1 (aie::zeros<acc48,8>()); //v8acc48
    aie::accum<acc48, 8> acc2 (aie::zeros<acc48,8>());
    aie::accum<acc48, 8> acc3 (aie::zeros<acc48,8>()); //v8acc48
    aie::accum<acc48, 8> acc4 (aie::zeros<acc48,8>());
    aie::vector<DTYPE,DX> first_m [8]; //v16int8
    aie::vector<DTYPE,DX> second_m[8]; //v16int8
    aie::vector<DTYPE,DX> vx = window_readincr_v16(in); //v32int8
    aie::vector<DTYPE,DX> v  = aie::zeros<DTYPE,16>();
    aie::vector<DTYPE,DX*2> vv = concat(vx,v);
    aie::vector<DTYPE,DX*2> vx_1 = aie::zeros<DTYPE,32>();
    aie::vector<DTYPE,DX*2> vx_2 = aie::zeros<DTYPE,32>();
    constexpr aie::mask<32> mask1 = aie::mask<32>(0xFFFFFF00); // it is litle endinan
    vx_1 = aie::select(vv, vx_1, mask1);
    vx_2 = aie::select(vx_2, vv, mask1);
    
    for (int q=0; q<8; q++){
        first_m[q] = aie::load_v<DX>((DTYPE*)matrix[q%2] + (q/2 * DX));
        second_m[q] = aie::load_v<DX>((DTYPE*)matrix[q%2] + DX*4 + (q/2 * DX));
    }
        
        // https://www.xilinx.com/htmldocs/xilinx2022_2/aiengine_intrinsics/intrinsics/group__vect__mult__16x16.html#ga1e00ad6eedd92916e22e27f83abe5f01
    acc1 = mac8(
            acc1,                            //v8acc48 	acc,
            concat(first_m[0],first_m[1],first_m[2],first_m[3],first_m[4],first_m[5],first_m[6],first_m[7]),  //v128int8 	xbuff,
            0,                              //int 	xstart,
            0x3130,                         //unsigned int 	xoffsets,
            32,                             //int 	xstep,
            0x3120,                         //unsigned int 	xsquare,
            vx_1,                           //v32int8 	zbuff,
            0,                              //int 	zstart,s
            0x0000,                         //unsigned int 	zoffsets,
            2,                              //int 	zstep,
            0x3210                          //unsigned int 	zsquare 
       );

    acc2 = mac8(
            acc2,                            //v8acc48 	acc,
            concat(second_m[0],second_m[1],second_m[2],second_m[3],second_m[4],second_m[5],second_m[6],second_m[7]),  //v128int8 	xbuff,
            0,                              //int 	xstart,
            0x3130,                         //unsigned int 	xoffsets,
            32,                             //int 	xstep,
            0x3120,                         //unsigned int 	xsquare,
            vx_2,                           //v32int8 	zbuff,
            0,                              //int 	zstart,
            0x0000,                         //unsigned int 	zoffsets,
            2,                              //int 	zstep,
            0x3210                          //unsigned int 	zsquare 
       );
    acc3 = mac8(
            acc3,                            //v8acc48 	acc,
            concat(first_m[0],first_m[1],first_m[2],first_m[3],first_m[4],first_m[5],first_m[6],first_m[7]),  //v128int8 	xbuff,
            8,                              //int 	xstart,
            0x3130,                         //unsigned int 	xoffsets,
            32,                             //int 	xstep,
            0x3120,                         //unsigned int 	xsquare,
            vx_1,                           //v32int8 	zbuff,
            0,                              //int 	zstart,s
            0x0000,                         //unsigned int 	zoffsets,
            2,                              //int 	zstep,
            0x3210                          //unsigned int 	zsquare 
       );

    acc4 = mac8(
            acc4,                            //v8acc48 	acc,
            concat(second_m[0],second_m[1],second_m[2],second_m[3],second_m[4],second_m[5],second_m[6],second_m[7]),  //v128int8 	xbuff,
            8,                              //int 	xstart,
            0x3130,                         //unsigned int 	xoffsets,
            32,                             //int 	xstep,
            0x3120,                         //unsigned int 	xsquare,
            vx_2,                           //v32int8 	zbuff,
            0,                              //int 	zstart,
            0x0000,                         //unsigned int 	zoffsets,
            2,                              //int 	zstep,
            0x3210                          //unsigned int 	zsquare 
       );

    // aie::vector<DTYPE, 16> vy = acc.to_vector<DTYPE>();
    aie::vector<int16, 8> vy = acc1.to_vector<int16>();
    aie::vector<int16, 8> vvy = acc2.to_vector<int16>();
    vvy = add(vy,vvy);
    aie::vector<int16, 8> vy2 = acc3.to_vector<int16>();
    aie::vector<int16, 8> vvy2 = acc4.to_vector<int16>();
    vvy2 = add(vy2,vvy2);

    window_writeincr(out, vvy);
    window_writeincr(out,vvy2);

}

void GemV16(
	input_window_int8 * __restrict in, 
    output_window_int8 * __restrict out)
{
    aie::accum<acc48, DY> acc (aie::zeros<acc48,DY>());
    // load in 32 values at a time
    aie::vector<DTYPE,32> vx = window_readincr_v32(in);

    for (int q=0; q<Q; q++) {
	    aie::vector<int8,128> MQS_concat = aie::load_v<128>((DTYPE*)&matrix[q][0][0]);
        int xstart = q * 128;
        // https://www.xilinx.com/htmldocs/xilinx2022_2/aiengine_intrinsics/intrinsics/group__vect__mult__16x16.html#ga1e00ad6eedd92916e22e27f83abe5f01
        acc = mac16(
            acc,           // v16acc48 acc

            // data buffer - 16bx16b scheme
            MQS_concat,    // v128i8   xbuff      - Input buffer of 128 elements of type i8
            xstart,             // int     xstart      - Starting position offset applied to all lanes of input from X buffer. xstart is restricted to multiples of 2 as granularity for xbuff is 32-bit.
            0x33323130,    // uint    xoffsets    - 4b offset for each lane, corresponds to 2x the lane number and each second lane is an offset to the lane before + 1. LSB apply to first lane
            32,    // uint    xoffsets_hi - 4b offset for each lane, corresponds to 2x the lane number and each second lane is an offset to the lane before + 1. LSB apply to 8th lane
            0x3120,        // uint    xsquare     - Select order of the mini-permute square (default=0x3210). LSB apply to first element
        
            // coef buffer - general scheme
            vx,            // v32i8   zbuff       - Input buffer of 32 elements of type i8
            0,             // int     zstart      - Starting position offset applied to all lanes for input from Z buffer. This must be a compile time constant. Only the 4 LSB of the argument are used.
            0x0,           // uint    zoffsets    - 4b offset for each lane, applied to input from Z buffer. LSB apply to first lane
            2              // int     zstep       - Step between each column for selection in the zbuffer.
        );
    }

    aie::vector<DTYPE, DY> vy = acc.to_vector<DTYPE>();
    window_writeincr(out, vy);
}
