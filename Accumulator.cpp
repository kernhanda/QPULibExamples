#include <sys/time.h>
#include <math.h>
#include <QPULib.h>

#include <algorithm>
#include <numeric>
#include <random>

#include <openblas/cblas.h>

using namespace QPULib;

void sum(Int N, Ptr<Float> A, Ptr<Float> result)
{
	Int inc = 16;//numQPUs() << 4;
	Int qpuID = me();
	Ptr<Float> a = A + index() + (qpuID * N);
	Ptr<Float> c = result + /*index() +*/ (qpuID << 4);
	gather(c);
	gather(a);

	Float aOld, sum;
	receive(sum);
  sum = 0;
	For (Int i = 0, i < N, i = i + inc)
		gather(a + i + inc);
		receive(aOld);
		sum = sum + aOld;
	End

	store(sum, c);
  receive(aOld);
}

// ============================================================================
// Main
// ============================================================================

int main()
{
  // Timestamps
  timeval tvStart, tvEnd, tvDiff;

  const int N = 16 * 12 * 256; // 192000

  const int NumQPUs = 12;

auto k = compile(sum);

  // Use 12 QPUs
  k.setNumQPUs(NumQPUs);

  SharedArray<float> result(16 * k.numQPUs);
  std::fill_n(&result[0], result.size, 0.0f);

  // Allocate and initialise arrays shared between ARM and GPU
  SharedArray<float> x(N);//, result(16 * k.numQPUs);
  std::random_device rd;
  std::default_random_engine engine(rd());
  std::uniform_real_distribution<float> dist;
  std::generate_n(&x[0], N, [&] { return dist(engine); });

{
  gettimeofday(&tvStart, NULL);
  auto expected = std::accumulate(&x[0], &x[0] + x.size, 0.f);
  gettimeofday(&tvEnd, NULL);
  timersub(&tvEnd, &tvStart, &tvDiff);
  printf("CPU: %ld.%06lds\n", tvDiff.tv_sec, tvDiff.tv_usec);
  printf("expected = %f\n", expected);
}

  gettimeofday(&tvStart, NULL);
  float gpuOut = 0.f;
  for (int i = 0; i < 50; ++i) {
  // Construct kernel

    k(N / k.numQPUs, &x, &result);
    gpuOut = std::accumulate(&result[0], &result[0] + result.size, 0.f);
  printf("gpuOutput = %f\n", gpuOut);
  // QPULib::astHeap.clear();
  }
  gettimeofday(&tvEnd, NULL);
  timersub(&tvEnd, &tvStart, &tvDiff);
  printf("GPU: %ld.%06lds\n", tvDiff.tv_sec, tvDiff.tv_usec);

  gettimeofday(&tvStart, NULL);
  volatile float expected = 0.f;
  for (int i = 0; i < 50; ++i) {
  expected = std::accumulate(&x[0], &x[0] + x.size, 0.f);
  }
  gettimeofday(&tvEnd, NULL);
  timersub(&tvEnd, &tvStart, &tvDiff);
  printf("CPU: %ld.%06lds\n", tvDiff.tv_sec, tvDiff.tv_usec);
  // printf("expected = %f\n", expected);

  // printf("\nIndex,A,B,GPU,CPU,Delta\n");
  // for (int i = 0; i < N; ++i) {
  //   float a = x[i], b = y[i];
  //   float gpu = result[i];
  //   float cpu = a * b;
  //   float delta = cpu - gpu;
  //   printf("%d,%f,%f,%f,%f,%f\n", i, a, b, gpu, cpu, delta);
  // }

  return 0;
}
