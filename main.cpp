// OpenMP header
#include "utility.h"

int main(int argc, char* argv[])
{
	int nthreads, tid;

	// Begin of parallel region
	#pragma omp parallel private(nthreads, tid)
	{
		// Getting thread number
		tid = omp_get_thread_num();
		printf("Welcome to GFG from thread = %d\n",
			tid);

		if (tid == 0) {

			// Only master thread does this
			nthreads = omp_get_num_threads();
			test(nthreads);
		}
	}
}
