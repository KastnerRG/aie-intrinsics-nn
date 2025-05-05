# Optimized Intrinsics based AIE Programming

## Roadmap:

- [x] Minimal GemV intrinsic for int16 with python golden model
- [ ] int8 GemV
- [ ] int32 GemV
- [ ] `GeMV(K,N, x_dtype, w_dtype, y_dtype)` python class to generate kernel code
- [ ] GeMM by intrinsics
- [ ] `GeMM()` class to generate kernel code
- [ ] Support any sizes for inputs & outputs
- [ ] `Conv()` class
- [ ] `Dense()` class creates and connects multiple `GeMM(...)` objects with equal II
- [ ] `NN()` class creates and connects multiple `Dense()` and `Conv()` classes
- [ ] Integrate with `hls4ml` as limited backend

## How to Run:

In waiter, do:
```bash
source /tools/Xilinx/Vivado/2024.1/settings64.sh && source /opt/xilinx/xrt/setup.sh
export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:/opt/xilinx/xrt/lib:/tools/Xilinx/Vitis/2024.1/aietools/lib/lnx64.o
```

To run simulation, enter a directory and 
```bash
make clean
make analyze
```

## Important Links:

* [Versal ACAP Architecture Manual](https://docs.amd.com/r/en-US/am020-versal-aie-ml/Overview)
* [Kernel Coding with Intrinsics User Guide](https://www.xilinx.com/htmldocs/xilinx2022_2/aiengine_intrinsics/intrinsics/group__vect__mult__intrinsics__explained.html)
* [Xilinx Intrinsics User Guide](https://docs.amd.com/r/en-US/ug1079-ai-engine-kernel-coding/Design-Analysis-and-Programming-using-Intrinsics)
* [Kernel (API) + Graph Coding for basic dense layer with weights on chip](https://github.com/zhenghuama/Versal-Linux-cmd-clean/tree/aba/weights_on_chip_i16)
* [Working intrinsics example](https://github.com/abarajithan11/amd_aie_matlab/blob/main/gemv_opt/GemV.cpp)

