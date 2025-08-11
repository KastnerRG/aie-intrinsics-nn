#include <vector>

#include <adf.h>
#include "kernels.h"
#include "kernels/include.h"

using namespace adf;

class simpleGraph : public adf::graph {
private:


  kernel mat_mul_k[mult_Y * mult_X * mult_Z];

public:

  input_plio  A[mult_X * mult_Y];
  input_plio  B[mult_Y * mult_Z];
  output_plio C[mult_X * mult_Z];


  simpleGraph(){

	  // input and output PLIOs creation below
	  for (int i = 0; i < mult_X * mult_Y; i++){
		  A[i] = input_plio::create(plio_128_bits, "data/matA" + std::to_string(i) + ".txt");
	  }

	  for (int i = 0; i < mult_Y * mult_Z; i++){
		  B[i] = input_plio::create(plio_128_bits, "data/matB" + std::to_string(i) + ".txt");
	  }

	  for (int i = 0; i < mult_X * mult_Z; i++){
		  C[i] = output_plio::create(plio_128_bits, "data/matC" + std::to_string(i) + ".txt");
	  }

	  // kernels creation
	  for (int i = 0; i < mult_Y * mult_X * mult_Z; i++){
		  mat_mul_k[i] = kernel::create(gemm);
	  }

	  // Single kernel connections
	  connect< window<single_M*single_K*1> >  (A[0].out[0], mat_mul_k[0].in[0]);
	  connect< window<single_K*single_N*1> >  (B[0].out[0], mat_mul_k[0].in[1]);

	  // Place buffers in different banks to prevent memory stalls (see UG1076 for more details)
	  not_equal(location<buffer>(mat_mul_k[0].in[0]), location<buffer>(mat_mul_k[0].in[1]));

	  connect< window<single_M*single_N*4> >  (mat_mul_k[0].out[0], C[0].in[0]);

	  // direct the source file of kernels
	  for (int i = 0; i < mult_Y * mult_X * mult_Z; i++){
		  source(mat_mul_k[i]) = "aie/kernels/kernels.cc";
	  }

	  runtime<ratio>(mat_mul_k[0]) = 1.0;
  }
};