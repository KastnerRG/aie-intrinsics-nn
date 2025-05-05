
#include <adf.h>
#include "kernels.h"
#include <vector>

using namespace adf;

#define DX 16
#define DY 16

class simpleGraph : public adf::graph {
private:
  kernel gemv_kernel;

public:

  input_plio  X;
  output_plio Y;

  simpleGraph(){

		X = input_plio::create(plio_128_bits, "data/x.txt");
		Y = output_plio::create(plio_128_bits, "data/y_sim.txt");
		gemv_kernel = kernel::create(GemV8); // Modify to use GemV8 or GemV4

	  connect< window<DX*sizeof(int32_t)> >  (X.out[0], gemv_kernel.in[0]);
	  connect< window<DY*sizeof(int32_t)> >  (gemv_kernel.out[0], Y.in[0]);
	  source(gemv_kernel) = "kernels/kernels.cc";

	  runtime<ratio>(gemv_kernel) = 1.0;
  }
};

simpleGraph mygraph;

int main(void) {
  mygraph.init();
  mygraph.run(20);
  mygraph.end();
  return 0;
}
