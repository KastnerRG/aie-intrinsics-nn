#include <adf.h>
#include "aie_api/aie.hpp"
#include "aie_api/aie_adf.hpp"
#include "matrix.h"

#define V8 8
#define V4 4

// #include <stdio.h>
// #include <stdlib.h>
// #include <type_traits>

// GemV8: 71 clocks
// Main: 
// II  : 
// Lat : 

// GemV4: 77 clocks
// Main: 
// II  : 
// Lat : 

void GemV8(
	input_window_int32 * __restrict in, 
    output_window_int32 * __restrict out)
{
    aie::accum<acc80, V8> acc1 (aie::zeros<acc80,V8>());
    aie::accum<acc80, V8> acc2 (aie::zeros<acc80,V8>());
    
    aie::vector<DTYPE,V8> vx;

    unsigned long long cycle_num[2];
    aie::tile tile = aie::tile::current();
    cycle_num[0] = tile.cycles();
    
   
    for (int i = 0; i < DY / V8; ++i) chess_prepare_for_pipelining chess_flatten_loop {
        vx = window_readincr_v8(in);

        // unrolled inner loop manually 
        // prefetch data

        aie::vector<DTYPE,DY> m0 = aie::load_v<DY>((DTYPE*)(matrix[(V8*i + 0)%2][(V8*i + 0)/2]));
        aie::vector<DTYPE,DY> m1 = aie::load_v<DY>((DTYPE*)(matrix[(V8*i + 1)%2][(V8*i + 1)/2]));

        acc1 = lmac8(acc1, m0, 0, 0x76543210, vx, 0, 0x0);
        acc2 = lmac8(acc2, m0, V8, 0x76543210, vx, 0, 0x0);

        aie::vector<DTYPE,DY> m2 = aie::load_v<DY>((DTYPE*)(matrix[(V8*i + 2)%2][(V8*i + 2)/2]));
        acc1 = lmac8(acc1, m1, 0, 0x76543210, vx, 1, 0x0);
        acc2 = lmac8(acc2, m1, V8, 0x76543210, vx, 1, 0x0);

        aie::vector<DTYPE,DY> m3 = aie::load_v<DY>((DTYPE*)(matrix[(V8*i + 3)%2][(V8*i + 3)/2]));
        acc1 = lmac8(acc1, m2, 0, 0x76543210, vx, 2, 0x0);
        acc2 = lmac8(acc2, m2, V8, 0x76543210, vx, 2, 0x0);

        aie::vector<DTYPE,DY> m4 = aie::load_v<DY>((DTYPE*)(matrix[(V8*i + 4)%2][(V8*i + 4)/2]));
        acc1 = lmac8(acc1, m3, 0, 0x76543210, vx, 3, 0x0);
        acc2 = lmac8(acc2, m3, V8, 0x76543210, vx, 3, 0x0);

        aie::vector<DTYPE,DY> m5 = aie::load_v<DY>((DTYPE*)(matrix[(V8*i + 5)%2][(V8*i + 5)/2]));
        acc1 = lmac8(acc1, m4, 0, 0x76543210, vx, 4, 0x0);
        acc2 = lmac8(acc2, m4, V8, 0x76543210, vx, 4, 0x0);

        aie::vector<DTYPE,DY> m6 = aie::load_v<DY>((DTYPE*)(matrix[(V8*i + 6)%2][(V8*i + 6)/2]));
        acc1 = lmac8(acc1, m5, 0, 0x76543210, vx, 5, 0x0);
        acc2 = lmac8(acc2, m5, V8, 0x76543210, vx, 5, 0x0);

        aie::vector<DTYPE,DY> m7 = aie::load_v<DY>((DTYPE*)(matrix[(V8*i + 7)%2][(V8*i + 7)/2]));
        acc1 = lmac8(acc1, m6, 0, 0x76543210, vx, 6, 0x0);
        acc2 = lmac8(acc2, m6, V8, 0x76543210, vx, 6, 0x0);


        acc1 = lmac8(acc1, m7, 0, 0x76543210, vx, 7, 0x0);
        acc2 = lmac8(acc2, m7, V8, 0x76543210, vx, 7, 0x0);
    
    }

    cycle_num[1] = tile.cycles();
    printf("start = %lld, end = %lld, total = %lld\n", cycle_num[0], cycle_num[1], cycle_num[1]-cycle_num[0]);
    
    window_writeincr(out, acc1.to_vector<DTYPE>());
    window_writeincr(out, acc2.to_vector<DTYPE>());
}


void GemV4(
	input_window_int32 *__restrict in,
	output_window_int32 *__restrict out)
{
	aie::accum<acc80, V4> acc1(aie::zeros<acc80, V4>());
	aie::accum<acc80, V4> acc2(aie::zeros<acc80, V4>());
	aie::accum<acc80, V4> acc3(aie::zeros<acc80, V4>());
	aie::accum<acc80, V4> acc4(aie::zeros<acc80, V4>());

	aie::vector<DTYPE, DY> m[Q];
	aie::vector<DTYPE, V8> vx;

	unsigned long long cycle_num[2];
	aie::tile tile = aie::tile::current();
	cycle_num[0] = tile.cycles();

	
	for (int i = 0; i < DY / V8; ++i) chess_prepare_for_pipelining chess_flatten_loop {
		vx = window_readincr_v8(in);

		
		for (int q = 0; q < Q; ++q)
			m[q] = aie::load_v<DY>((DTYPE *)(matrix[q][V4 * i + 0]));
		aie::vector<DTYPE, DY * 2> rows0 = concat(MQS);

		// prefetch
		for (int q = 0; q < Q; ++q)
			m[q] = aie::load_v<DY>((DTYPE *)(matrix[q][V4 * i + 1]));
		aie::vector<DTYPE, DY * 2> rows1 = concat(MQS);

		
		acc1 = lmac4(acc1, rows0, 0, 0x00003210, DY, vx, 0, 0x0, 1);
		acc2 = lmac4(acc2, rows0, V4, 0x00003210, DY, vx, 0, 0x0, 1);
		acc3 = lmac4(acc3, rows0, V4 * 2, 0x00003210, DY, vx, 0, 0x0, 1);
		acc4 = lmac4(acc4, rows0, V4 * 3, 0x00003210, DY, vx, 0, 0x0, 1);

		
		for (int q = 0; q < Q; ++q)
			m[q] = aie::load_v<DY>((DTYPE *)(matrix[q][V4 * i + 2]));
		aie::vector<DTYPE, DY * 2> rows2 = concat(MQS);

		
		acc1 = lmac4(acc1, rows1, 0, 0x00003210, DY, vx, 2, 0x0, 1);
		acc2 = lmac4(acc2, rows1, V4, 0x00003210, DY, vx, 2, 0x0, 1);
		acc3 = lmac4(acc3, rows1, V4 * 2, 0x00003210, DY, vx, 2, 0x0, 1);
		acc4 = lmac4(acc4, rows1, V4 * 3, 0x00003210, DY, vx, 2, 0x0, 1);

		
		for (int q = 0; q < Q; ++q)
			m[q] = aie::load_v<DY>((DTYPE *)(matrix[q][V4 * i + 3]));
		aie::vector<DTYPE, DY * 2> rows3 = concat(MQS);

		
		acc1 = lmac4(acc1, rows2, 0, 0x00003210, DY, vx, 4, 0x0, 1);
		acc2 = lmac4(acc2, rows2, V4, 0x00003210, DY, vx, 4, 0x0, 1);
		acc3 = lmac4(acc3, rows2, V4 * 2, 0x00003210, DY, vx, 4, 0x0, 1);
		acc4 = lmac4(acc4, rows2, V4 * 3, 0x00003210, DY, vx, 4, 0x0, 1);

		
		acc1 = lmac4(acc1, rows3, 0, 0x00003210, DY, vx, 6, 0x0, 1);
		acc2 = lmac4(acc2, rows3, V4, 0x00003210, DY, vx, 6, 0x0, 1);
		acc3 = lmac4(acc3, rows3, V4 * 2, 0x00003210, DY, vx, 6, 0x0, 1);
		acc4 = lmac4(acc4, rows3, V4 * 3, 0x00003210, DY, vx, 6, 0x0, 1);
	}

	cycle_num[1] = tile.cycles();
	printf("start = %lld, end = %lld, total = %lld\n", cycle_num[0], cycle_num[1], cycle_num[1] - cycle_num[0]);

	window_writeincr(out, acc1.to_vector<DTYPE>());
	window_writeincr(out, acc2.to_vector<DTYPE>());
	window_writeincr(out, acc3.to_vector<DTYPE>());
	window_writeincr(out, acc4.to_vector<DTYPE>());
}




