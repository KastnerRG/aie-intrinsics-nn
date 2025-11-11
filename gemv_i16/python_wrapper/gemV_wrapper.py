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

#ifndef DX
#define DX {K}
#endif

#ifndef DY
#define DY {N}
#endif

#define V16 16
""")

TEMPLATE_GEMV_I16_MAC16 = dedent(r"""
void GemV_i16_mac16(
    input_window_int16 * __restrict in,
    output_window_int16 * __restrict out)
{
    v16int16 vx_arr[DX / V16];
    for (int i = 0; i < DX / V16; ++i)
        vx_arr[i] = window_readincr_v16(in);

    for (int k = 0; k < DY / V16; ++k) {
        aie::accum<acc48, V16> acc(aie::zeros<acc48, V16>());

        for (int i = 0; i < DX / V16; ++i) {
            v16int16 vx = vx_arr[i];

            for (int j = 0; j < V16 / 2; ++j) {
                v16int16 m_lo = aie::load_v<V16>((DTYPE*)
                    (&matrix[0][(V16 / 2) * i + j][k * V16]));

                v16int16 m_hi = aie::load_v<V16>((DTYPE*)
                    (&matrix[1][(V16 / 2) * i + j][k * V16]));

                v32int16 m32 = concat(m_lo, m_hi);

                acc = mac16(
                    acc,
                    m32,
                    0,
                    0x73727170,
                    0x77767574,
                    0x3120,
                    vx,
                    j * 2,
                    0x0,
                    0x0,
                    1
                );
            }
        }

        aie::vector<DTYPE, V16> vy = acc.to_vector<DTYPE>();
        window_writeincr(out, vy);
    }
}
""")

def generate_gemV_i16_mac16(m: int, k: int, n: int) -> str:
    header = TEMPLATE_HEADER.format(M=m, K=k, N=n)
    return header + TEMPLATE_GEMV_I16_MAC16

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Generate GemV_i16_mac16 kernel C++ source.")
    parser.add_argument("--m", type=int, required=True, help="Rows of A / output matrix.")
    parser.add_argument("--k", type=int, required=True, help="Inner dimension.")
    parser.add_argument("--n", type=int, required=True, help="Cols of B / output matrix.")
    parser.add_argument("--out", type=str, default="gemV_i16_mac16.cpp",
                        help="Output file name.")
    args = parser.parse_args()

    cpp_code = generate_gemV_i16_mac16(args.m, args.k, args.n)
    with open(args.out, "w") as f:
        f.write(cpp_code)

    print(f"Generated kernel written to {args.out}")