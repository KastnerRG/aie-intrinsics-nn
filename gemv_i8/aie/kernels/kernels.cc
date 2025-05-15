#include <adf.h>
#include "adf/window/window.h"
#include "aie_api/aie.hpp"
#include "aie_api/aie_adf.hpp"
#include "matrix.h"


void GemV(
	input_window_int8 * __restrict in, 
  output_window_int8 * __restrict out)
{
    aie::accum<acc48, 8> acc (aie::zeros<acc48,8>()); //v8acc48
    aie::vector<DTYPE,64> first_m [Q]; //v16int8
    aie::vector<DTYPE,64> second_m[Q]; //v16int8
    aie::vector<DTYPE,16> vx = window_readincr_v16(in); //v16int8
    aie::vector<DTYPE,16> vx_1 = aie::zeros<DTYPE,16>();
    aie::vector<DTYPE,16> vx_2 = aie::zeros<DTYPE,16>();
    constexpr aie::mask<16> mask1 = aie::mask<16>(0xFFFF);
    vx_1 = aie::select(vx, vx_1, mask1);
    vx_2 = aie::select(vx_2, vx, mask1);
    
    for (int q=0; q<2; q++){
        first_m[q] = aie::load_v<64>((DTYPE*)matrix[q%2]);
        second_m[q] = aie::load_v<64>((DTYPE*)matrix[q%2] + 16*4);
    }
        
        // https://www.xilinx.com/htmldocs/xilinx2022_2/aiengine_intrinsics/intrinsics/group__vect__mult__16x16.html#ga1e00ad6eedd92916e22e27f83abe5f01
    mac8(
            acc,    //v8acc48 	acc,
            concat(first_m[0],first_m[0]),//v128int8 	xbuff,
            0,      //int 	xstart,
            //unsigned int 	xoffsets,
            //int 	xstep,
            //unsigned int 	xsquare,
            //v32int8 	zbuff,
            //int 	zstart,
            //unsigned int 	zoffsets,
            //int 	zstep,
            //unsigned int 	zsquare 
       );

    // aie::vector<DTYPE, 16> vy = acc.to_vector<DTYPE>();
    // aie::vector<int16, DY> vy = srs(acc,0);
    // window_writeincr(out, window_read_v16(in));
}