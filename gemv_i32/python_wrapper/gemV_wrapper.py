import argparse
from textwrap import dedent

TEMPLATE_HEADER = dedent(r"""
#include <adf.h>
#include "aie_api/aie.hpp"
#include "aie_api/aie_adf.hpp"
#include "matrix.h"

#ifndef M
#define M {M}
#endif

#ifndef K
#define K {K}
#endif

#ifndef N
#define N {N}
#endif

#define V8 8
#define V4 4
""")

TEMPLATE_GEMV8 = dedent(r"""
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
            
            acc1 = lmac8(
                acc1,
                m,
                0,
                0x76543210,
                vx,
                j,
                0x0
            );

            acc2 = lmac8(acc2, m, V8, 0x76543210, vx, j, 0x0);
        }
    }

    aie::vector<DTYPE, V8> vy1 = acc1.to_vector<DTYPE>();
    aie::vector<DTYPE, V8> vy2 = acc2.to_vector<DTYPE>();
    window_writeincr(out, vy1);
    window_writeincr(out, vy2);
}
""")

TEMPLATE_GEMV4 = dedent(r"""
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
            
            acc1 = lmac4(
                acc1,
                rows,
                0,
                0x00003210,
                DY,
                vx,
                j*2,
                0x0,
                1
            );

            acc2 = lmac4(acc2, rows, V4, 0x00003210, DY, vx, j*2, 0x0, 1);
            acc3 = lmac4(acc3, rows, V4*2, 0x00003210, DY, vx, j*2, 0x0, 1);
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
""")

def generate_gemV_int32(lmac_mode: str, m: int, k: int, n: int) -> str:
    header = TEMPLATE_HEADER.format(M=m, K=k, N=n)
    if lmac_mode == "lmac8":
        return header + TEMPLATE_GEMV8
    elif lmac_mode == "lmac4":
        return header + TEMPLATE_GEMV4
    else:
        raise ValueError(f"Unknown lmac mode: {lmac_mode}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Generate GemV4 or GemV8 kernel C++ source.")
    parser.add_argument("--lmac", choices=["lmac4", "lmac8"], required=True,
                        help="MAC mode to generate.")
    parser.add_argument("--m", type=int, required=True, help="Rows of A / output matrix.")
    parser.add_argument("--k", type=int, required=True, help="Inner dimension.")
    parser.add_argument("--n", type=int, required=True, help="Cols of B / output matrix.")
    parser.add_argument("--out", type=str, default="gemV_int32.cpp",
                        help="Output file name.")
    args = parser.parse_args()

    cpp_code = generate_gemV_int32(args.lmac, args.m, args.k, args.n)
    with open(args.out, "w") as f:
        f.write(cpp_code)

    print(f"Generated kernel written to {args.out}")
