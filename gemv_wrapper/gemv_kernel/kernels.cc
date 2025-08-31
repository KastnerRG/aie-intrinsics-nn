#include <adf.h>
#include "adf/window/window.h"
#include "aie_api/aie.hpp"
#include "aie_api/aie_adf.hpp"
#include "matrix.h"


void GemV_int16_int16_Mac16(input_window_int16 * __restrict in,
                        output_window_int16 * __restrict out)
{
  aie::accum<acc48, DY> acc (aie::zeros<acc48,DY>());
  aie::vector<DTYPE, DY> m[Q];
  aie::vector<DTYPE, DX> vx = window_readincr_v16(in);

  for (int i=0, id=0; i<DX; i+=Q, id+=DY) {
    for (int q=0; q<Q; ++q)
      m[q] = aie::load_v<DY>((DTYPE*)matrix[q] + id);

    acc = mac16(acc, concat(MQS), 0x0, 0x73727170, 0x77767574, 0x3120, vx, i, 0x0, 0x0, 1 );
  }

  aie::vector<DTYPE, DY> vy = acc.to_vector<DTYPE>();
  window_writeincr(out, vy);
}