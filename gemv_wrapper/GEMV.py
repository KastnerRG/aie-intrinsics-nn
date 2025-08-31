"""
Generates:
  - kernels.cc : a single kernel with a strategy-derived body
  - graph.cpp  : ADF graph with window sizes

"""

from __future__ import annotations
from dataclasses import dataclass, field
from pathlib import Path
from typing import Callable, Dict, Tuple, Optional, Any, Union

DTYPES = {"int8", "int16", "int32"}

# ---------------------------
# Utility helpers
# ---------------------------
def _require(cond: bool, msg: str):
    if not cond:
        raise ValueError(msg)

def _hex(v: Any) -> str:
    if isinstance(v, int):
        return hex(v)
    return str(v)

# ---------------------------
# Strategies
# ---------------------------

def strat_int16_scheme(cfg, p) -> str:
    """Parametric 16b×16b schemes for mac16 / mac8 with concat(MQS)."""
    _require(cfg.K == 16,  "int16 schemes assume K==16 (v16 read).")
    _require(cfg.N == 16,  "int16 schemes assume N==16 (v16 lanes in acc).")

    xstart      = _hex(p["xstart"])
    xoffs_lo    = _hex(p["xoffsets_lo"])
    xsquare     = _hex(p["xsquare"])
    zoffs_lo    = _hex(p.get("zoffsets_lo", 0))
    zstep       = p.get("zstep", 1)
    intrinsic   = cfg.mac_intrinsic

    if intrinsic == "mac16":
        xoffs_hi = _hex(p["xoffsets_hi"])
        zoffs_hi = _hex(p.get("zoffsets_hi", 0))
        mac_call = f"""acc = mac16(acc, concat(MQS), {xstart}, {xoffs_lo}, {xoffs_hi}, {xsquare}, vx, i, {zoffs_lo}, {zoffs_hi}, {zstep} );"""
    elif intrinsic == "mac8":
        xstep = p["xstep"]
        mac_call = f"""acc = mac8(acc, concat(MQS), {xstart}, {xoffs_lo}, {xstep}, {xsquare}, vx, i, {zoffs_lo}, {zstep} );"""
    else:
        raise ValueError("int16 scheme only supports mac16/mac8")

    return f"""
void {cfg.kernel_name}(input_window_int16 * __restrict in,
                        output_window_int16 * __restrict out)
{{
  aie::accum<acc48, DY> acc (aie::zeros<acc48,DY>());
  aie::vector<DTYPE, DY> m[Q];
  aie::vector<DTYPE, DX> vx = window_readincr_v16(in);

  for (int i=0, id=0; i<DX; i+=Q, id+=DY) {{
    for (int q=0; q<Q; ++q)
      m[q] = aie::load_v<DY>((DTYPE*)matrix[q] + id);

    {mac_call}
  }}

  aie::vector<DTYPE, DY> vy = acc.to_vector<DTYPE>();
  window_writeincr(out, vy);
}}"""


def strat_int32_lmac(cfg, p) -> str:
    """Parametric 32b lmacX scheme (lmac8 halves or lmac4 quarters)."""
    V        = p["V"]
    parts    = p["parts"]
    xoffs    = _hex(p["xoffsets"])
    acc_tag  = p.get("acc_tag", "acc80")
    _require(cfg.N % V == 0,   f"N must be multiple of {V} for lmac{V}.")
    _require(cfg.N == 16,      "This generator expects N==16 for lmac patterns.")
    _require(cfg.K == cfg.N,   "For int32 lmac, set K==N so input window matches reads.")

    if cfg.mac_intrinsic == "lmac8":
        xbuff_decl  = "aie::vector<DTYPE, DY> m;"
        xbuff_fill  = "m = aie::load_v<DY>((DTYPE*)(matrix[(V*i + j)%2][(V*i + j)/2]));"
        call_lines  = "\n      ".join(
            [f"acc[{k}] = lmac{V}(acc[{k}], m, {k}*V, {xoffs}, vx, j, 0x0);" for k in range(parts)]
        )
        acc_decl    = f"aie::accum<{acc_tag}, V> acc[{parts}] = {{"
        acc_decl   += ", ".join([f"aie::zeros<{acc_tag},V>()" for _ in range(parts)]) + "};"
        write_out   = "\n  ".join(
            [f"aie::vector<DTYPE, V> vy{k} = acc[{k}].to_vector<DTYPE>();" for k in range(parts)]
        ) + f"""
  window_writeincr(out, vy0);
  window_writeincr(out, vy1);"""
    else:
        xbuff_decl  = """aie::vector<DTYPE, DY> m[Q];
  aie::vector<DTYPE, DY*2> rows;"""
        xbuff_fill  = """for (int q=0; q<Q; ++q)
        m[q] = aie::load_v<DY>((DTYPE*)(matrix[q][V*i + j]));
      rows = concat(MQS);"""
        call_lines  = "\n      ".join(
            [f"acc[{k}] = lmac{V}(acc[{k}], rows, {k}*V, 0x00003210, DY, vx, j*2, 0x0, 1);" for k in range(parts)]
        )
        acc_decl    = f"aie::accum<{acc_tag}, V> acc[{parts}] = {{"
        acc_decl   += ", ".join([f"aie::zeros<{acc_tag},V>()" for _ in range(parts)]) + "};"
        write_out   = "\n  ".join(
            [f"aie::vector<DTYPE, V> vy{k} = acc[{k}].to_vector<DTYPE>();" for k in range(parts)]
        ) + "\n  " + "\n  ".join([f"window_writeincr(out, vy{k});" for k in range(parts)])

    return f"""
void {cfg.kernel_name}(input_window_int32 * __restrict in,
                        output_window_int32 * __restrict out)
{{
  constexpr int V = {V};
  {acc_decl}
  aie::vector<DTYPE, V>  vx;
  {xbuff_decl}

  for (int i=0; i < DY/V; ++i) {{
    vx = window_readincr_v{V}(in);
    for (int j=0; j < V; ++j) {{
      {xbuff_fill}
      {call_lines}
    }}
  }}

  {write_out}
}}"""


def strat_int8_mac16(cfg, p) -> str:
    """Parametric 8b mac16 tiled scheme (your original requires K=32)."""
    _require(cfg.K == 32, "int8 mac16 expects K==32 (v32 read).")
    _require(cfg.N == 16, "int8 mac16 expects N==16 lanes.")
    xoffs_lo = _hex(p["xoffsets_lo"])
    xoffs_hi_or_xstep = _hex(p["xoffs_hi_or_xstep"])
    xsquare  = _hex(p["xsquare"])
    zstep    = p.get("zstep", 2)
    Qsym     = p.get("Q_symbol", "Q")
    xstride  = p.get("xstart_stride", 128)
    mload    = p.get("mload_expr", "(DTYPE*)&matrix[q][0][0]")

    return f"""
void {cfg.kernel_name}(input_window_int8 * __restrict in,
                        output_window_int8 * __restrict out)
{{
  aie::accum<acc48, DY> acc (aie::zeros<acc48,DY>());
  aie::vector<DTYPE, 32> vx = window_readincr_v32(in);

  for (int q=0; q<{Qsym}; ++q) {{
    aie::vector<int8,128> MQS_concat = aie::load_v<128>({mload});
    int xstart = q * {xstride};
    acc = mac16(
      acc, MQS_concat,
      xstart,
      {xoffs_lo},
      {xoffs_hi_or_xstep},
      {xsquare},
      vx,
      0,
      0x0,
      {zstep}
    );
  }}

  aie::vector<DTYPE, DY> vy = acc.to_vector<DTYPE>();
  window_writeincr(out, vy);
}}"""


def strat_int8_mac8(cfg, p) -> str:
    """Parametric 8b mac8 two-half pattern (K=N=16)."""
    _require(cfg.K == 16, "int8 mac8 expects K==16 (v16 read).")
    _require(cfg.N == 16, "int8 mac8 expects N==16 (two v8 writes).")

    xoffs    = _hex(p["xoffsets"])
    xstep    = p["xstep"]
    xsquare  = _hex(p["xsquare"])
    zoffs    = _hex(p.get("zoffsets", 0))
    zstep    = p.get("zstep", 2)
    zsquare  = _hex(p.get("zsquare", 0x3210))
    first_off= p.get("first_block_off", "0")
    second_off= p.get("second_block_off", "DX*4")
    mask32   = _hex(p.get("mask32", 0xFFFFFF00))

    return f"""
void {cfg.kernel_name}(input_window_int8 * __restrict in,
                        output_window_int16 * __restrict out)
{{
  aie::accum<acc48, 8> acc1 (aie::zeros<acc48,8>());
  aie::accum<acc48, 8> acc2 (aie::zeros<acc48,8>());
  aie::accum<acc48, 8> acc3 (aie::zeros<acc48,8>());
  aie::accum<acc48, 8> acc4 (aie::zeros<acc48,8>());

  aie::vector<DTYPE,DX> first_m[8];
  aie::vector<DTYPE,DX> second_m[8];

  aie::vector<DTYPE,DX> vx = window_readincr_v16(in);
  aie::vector<DTYPE,DX> v  = aie::zeros<DTYPE,16>();
  aie::vector<DTYPE,DX*2> vv   = concat(vx, v);
  aie::vector<DTYPE,DX*2> vx_1 = aie::zeros<DTYPE,32>();
  aie::vector<DTYPE,DX*2> vx_2 = aie::zeros<DTYPE,32>();

  constexpr aie::mask<32> mask1 = aie::mask<32>({mask32});
  vx_1 = aie::select(vv,   vx_1, mask1);
  vx_2 = aie::select(vx_2, vv,   mask1);

  for (int q=0; q<8; ++q) {{
    first_m[q]  = aie::load_v<DX>((DTYPE*)matrix[q%2] + ({first_off})  + (q/2 * DX));
    second_m[q] = aie::load_v<DX>((DTYPE*)matrix[q%2] + ({second_off}) + (q/2 * DX));
  }}

  acc1 = mac8(acc1, concat(first_m[0],first_m[1],first_m[2],first_m[3],first_m[4],first_m[5],first_m[6],first_m[7]),
              0, {xoffs}, {xstep}, {xsquare}, vx_1, 0, {zoffs}, {zstep}, {zsquare});
  acc2 = mac8(acc2, concat(second_m[0],second_m[1],second_m[2],second_m[3],second_m[4],second_m[5],second_m[6],second_m[7]),
              0, {xoffs}, {xstep}, {xsquare}, vx_2, 0, {zoffs}, {zstep}, {zsquare});
  acc3 = mac8(acc3, concat(first_m[0],first_m[1],first_m[2],first_m[3],first_m[4],first_m[5],first_m[6],first_m[7]),
              8, {xoffs}, {xstep}, {xsquare}, vx_1, 0, {zoffs}, {zstep}, {zsquare});
  acc4 = mac8(acc4, concat(second_m[0],second_m[1],second_m[2],second_m[3],second_m[4],second_m[5],second_m[6],second_m[7]),
              8, {xoffs}, {xstep}, {xsquare}, vx_2, 0, {zoffs}, {zstep}, {zsquare});

  aie::vector<int16, 8> v0 = acc1.to_vector<int16>();
  aie::vector<int16, 8> v1 = acc2.to_vector<int16>();
  aie::vector<int16, 8> v2 = acc3.to_vector<int16>();
  aie::vector<int16, 8> v3 = acc4.to_vector<int16>();

  v1 = add(v0, v1);
  v3 = add(v2, v3);

  window_writeincr(out, v1);
  window_writeincr(out, v3);
}}"""

# ---------------------------
# Variant registry
# ---------------------------

VariantKey = Tuple[str, str, str]
Strategy   = Callable[[Any, Dict[str, Any]], str]
HeaderVal  = Union[str, int]  # "K"/"N" or fixed int

VARIANTS: Dict[VariantKey, Dict[str, Any]] = {
    # ---- int16 × int16 ----
    ("int16","int16","mac16"): dict(
        y_dtype="int16",
        header=dict(DX="K", DY="N"),
        validate=lambda K,N: (_require(K==16, "mac16 int16 needs K=16"), _require(N==16, "N=16")),
        strategy=strat_int16_scheme,
        params=dict(
            xstart=0,
            xoffsets_lo=0x73727170,
            xoffsets_hi=0x77767574,
            xsquare=0x3120,
            zoffsets_lo=0x0,
            zoffsets_hi=0x0,
            zstep=1,
        ),
    ),
    ("int16","int16","mac8"): dict(
        y_dtype="int16",
        header=dict(DX="K", DY="N"),
        validate=lambda K,N: (_require(K==16, "mac8 int16 needs K=16"), _require(N==16, "N=16")),
        strategy=strat_int16_scheme,
        params=dict(
            xstart=0,
            xoffsets_lo=0x33323130,
            xstep=16,
            xsquare=0x3120,
            zoffsets_lo=0x0,
            zstep=1,
        ),
    ),

    # ---- int32 × int32 ----
    ("int32","int32","lmac8"): dict(
        y_dtype="int32",
        header=dict(DX="K", DY="N"),
        validate=lambda K,N: (_require(N==16, "N=16 for lmac8"), _require(K==N, "Use K==N")),
        strategy=strat_int32_lmac,
        params=dict(V=8, parts=2, xoffsets=0x76543210, acc_tag="acc80"),
    ),
    ("int32","int32","lmac4"): dict(
        y_dtype="int32",
        header=dict(DX="K", DY="N"),
        validate=lambda K,N: (_require(N==16, "N=16 for lmac4"), _require(K==N, "Use K==N")),
        strategy=strat_int32_lmac,
        params=dict(V=4, parts=4, xoffsets=0x00003210, acc_tag="acc80"),
    ),

    # ---- int8 × int8 ----
    ("int8","int8","mac16"): dict(
        y_dtype="int8",
        header=dict(DX="K", DY="N"),
        validate=lambda K,N: (_require(K==32, "mac16 int8 needs K=32 because z buffer requires 32 elements "),
                              _require(N==16, "N=16")),
        strategy=strat_int8_mac16,
        params=dict(
            xoffsets_lo=0x33323130,
            xoffs_hi_or_xstep=32,
            xsquare=0x3120,
            zstep=2,
            Q_symbol="Q",
            xstart_stride=128,
            mload_expr="(DTYPE*)&matrix[q][0][0]",
        ),
    ),
    ("int8","int8","mac8"): dict(
        y_dtype="int16",
        header=dict(DX="K", DY="N"),
        validate=lambda K,N: (_require(K==16, "mac8 int8 needs K=16"), _require(N==16, "N=16")),
        strategy=strat_int8_mac8,
        params=dict(
            xoffsets=0x3130,
            xstep=32,
            xsquare=0x3120,
            zoffsets=0x0000,
            zstep=2,
            zsquare=0x3210,
            first_block_off="0",
            second_block_off="DX*4",
            mask32=0xFFFFFF00,
        ),
    ),
}

# ---------------------------
# Config + generator
# ---------------------------

@dataclass
class GemVConfig:
    x_dtype: str
    z_dtype: str
    mac_intrinsic: str
    K: int = 16
    N: int = 16
    y_dtype: Optional[str] = None
    kernel_name: Optional[str] = None
    graph_name: str = "simpleGraph"
    overrides: Dict[str, Any] = field(default_factory=dict)

    def finalize(self):
        _require(self.x_dtype in DTYPES and self.z_dtype in DTYPES, f"Supported dtypes: {sorted(DTYPES)}")
        key = (self.x_dtype, self.z_dtype, self.mac_intrinsic)
        _require(key in VARIANTS, f"No variant registered for {key}")
        var = VARIANTS[key]

        # Validate K/N for the selected strategy
        val = var["validate"]
        if isinstance(val, tuple):
            for v in val: v(self.K, self.N)
        else:
            val(self.K, self.N)

        if self.y_dtype is None:
            self.y_dtype = var["y_dtype"]

        if self.kernel_name is None:
            base = {"mac16":"Mac16","mac8":"Mac8","lmac4":"Lmac4","lmac8":"Lmac8"}.get(self.mac_intrinsic, self.mac_intrinsic)
            self.kernel_name = f"GemV_{self.x_dtype}_{self.z_dtype}_{base}"

class GemVWrapper:
    def __init__(self, cfg: GemVConfig):
        cfg.finalize()
        self.cfg = cfg

    def _header(self) -> str:
        c = self.cfg
        var = VARIANTS[(c.x_dtype, c.z_dtype, c.mac_intrinsic)]
        return (
            '#include <adf.h>\n'
            '#include "adf/window/window.h"\n'
            '#include "aie_api/aie.hpp"\n'
            '#include "aie_api/aie_adf.hpp"\n'
            '#include "matrix.h"\n\n'
        )

    def render_kernels_cc(self) -> str:
        c   = self.cfg
        var = VARIANTS[(c.x_dtype, c.z_dtype, c.mac_intrinsic)]
        params = {**var["params"], **(c.overrides or {})}
        body   = var["strategy"](c, params)
        return self._header() + body

    def render_graph_cpp(self) -> str:
        c   = self.cfg
        var = VARIANTS[(c.x_dtype, c.z_dtype, c.mac_intrinsic)]

        return f"""
#include <adf.h>
#include "kernels.h"
#include <vector>

#define DX 16
#define DY 16

using namespace adf;

class {c.graph_name} : public adf::graph {{
private:
  kernel gemv_kernel;
public:
  input_plio X;
  output_plio Y;

  {c.graph_name}() {{
    X = input_plio::create(plio_128_bits, "data/x.txt");
    Y = output_plio::create(plio_128_bits, "data/y_sim.txt");
    gemv_kernel = kernel::create({c.kernel_name});

    connect< window<DX*sizeof({c.x_dtype}_t)> >(X.out[0], gemv_kernel.in[0]);
    connect< window<DY*sizeof({c.y_dtype}_t)> >(gemv_kernel.out[0], Y.in[0]);

    source(gemv_kernel) = "kernels.cc";
    runtime<ratio>(gemv_kernel) = 1.0;
  }}
}};

{c.graph_name} mygraph;

int main() {{
  mygraph.init();
  mygraph.run(20);
  mygraph.end();
  return 0;
}}
"""

    def write_all(self, out_dir: Path | str):
        out = Path(out_dir)
        out.mkdir(parents=True, exist_ok=True)
        (out / "kernels.cc").write_text(self.render_kernels_cc(), encoding="utf-8")
        (out / "graph.cpp").write_text(self.render_graph_cpp(), encoding="utf-8")


if __name__ == "__main__":
    import sys
    dtype = input("dtype (int8/int16/int32): ").strip()

    if dtype not in DTYPES:
        print(f"Unsupported dtype. Choose from {sorted(DTYPES)}.")
        sys.exit(1)

    # Prompt intrinsic based on dtype
    if dtype in ("int8", "int16"):
        allowed = ("mac8", "mac16")
    else:  # int32
        allowed = ("lmac4", "lmac8")

    print(f"Choose intrinsic {allowed}:")
    intrinsic = input("> ").strip()
    if intrinsic not in allowed:
        print("Invalid intrinsic choice.")
        sys.exit(1)

    # K and N fixed to 16
    K = 16
    N = 16

    try:
        cfg = GemVConfig(x_dtype=dtype, z_dtype=dtype, mac_intrinsic=intrinsic, K=K, N=N)
        gen = GemVWrapper(cfg)
    except ValueError as e:
        print(f"Configuration error: {e}")
        if dtype == "int8" and intrinsic == "mac16":
            print("Tip: For dtype=int8 with K=16, choose intrinsic 'mac8'.")
        sys.exit(1)

    out_dir = Path("./gemv_kernel")
    gen.write_all(out_dir)
    print(f"Wrote kernels.cc and graph.cpp to {out_dir.resolve()}")
