#include <math.h>
#include <sys/time.h>

#include <QPULib.h>

#include <algorithm>
#include <array>
#include <chrono>
#include <numeric>
#include <random>

#include <openblas/cblas.h>

using namespace QPULib;

void dot1(Int N, Ptr<Float> A, Ptr<Float> B, Ptr<Float> result) {
  Int inc = 16;
  Int qpuID = me();
  Ptr<Float> a = A + index() + (qpuID * N);
  Ptr<Float> b = B + index() + (qpuID * N);
  Ptr<Float> c = result + index() + (qpuID << 4);
  gather(c);
  gather(a);
  gather(b);

  Float aOld, bOld, sum;
  receive(sum);
  sum = 0;
  For(Int i = 0, i < N, i = i + inc)
    gather(a + i + inc);
    gather(b + i + inc);
    receive(aOld);
    receive(bOld);
    sum = sum + (aOld * bOld);
  End

  store(sum, c);

  receive(aOld);
  receive(bOld);
}

void dot2(Int N, Ptr<Float> A, Ptr<Float> B, Ptr<Float> result) {
  Int inc = numQPUs() << 4;
  Int qpuID = me();
  Ptr<Float> a = A + index() + (qpuID << 4);
  Ptr<Float> b = B + index() + (qpuID << 4);
  Ptr<Float> c = result + index() + (qpuID << 4);
  gather(c);
  gather(a); gather(b);

  Float aOld, bOld, sum;
  receive(sum);
  sum = 0;
  For(Int i = 0, i < N, i = i + inc)
    gather(a + inc); gather(b + inc);
    receive(aOld); receive(bOld);
    sum = sum + (aOld * bOld);
    a = a + inc; b = b + inc;
  End

  store(sum, c);

  receive(aOld); receive(bOld);
}

// ============================================================================
// Main
// ============================================================================

int main() {
  std::random_device rd;
  std::default_random_engine engine(rd());
  std::uniform_real_distribution<float> dist;

  // Timestamps
  timeval tvStart, tvEnd, tvDiff;
  const int NumQPUs = 12;

  // Construct kernel
  auto kDot1 = compile(dot1);
  auto kDot2 = compile(dot2);

  // Use 12 QPUs
  kDot1.setNumQPUs(NumQPUs);
  kDot2.setNumQPUs(NumQPUs);

  printf("N,GPU1_Time,GPU2_Time,BLAS_Time\n");
  const int ITERATIONS = 100;
  for (unsigned i = 2; i <= (2 << 12); i *= 2) {
    const unsigned N = 16 * NumQPUs * i;

    // Allocate and initialise arrays shared between ARM and GPU
    std::array<SharedArray<float>, 5> x;
    std::array<SharedArray<float>, 5> y;
    SharedArray<float> result(NumQPUs * 16);
    for (unsigned i = 0; i < x.size(); ++i) {
      x[i].alloc(N);
      std::generate_n(&x[i][0], x[i].size, [&] { return dist(engine); });
      y[i].alloc(N);
      std::generate_n(&y[i][0], y[i].size, [&] { return dist(engine); });
    }

    ///////////////
    ///////////////
    ///////////////

    // warm-up gpu
    for (unsigned i = 0; i < x.size(); ++i) {
      kDot1((int)(N / NumQPUs), &x[i], &y[i], &result);
      (void)std::accumulate(&result[0], &result[0] + result.size, 0.f);
    }

    printf("%u,", N);
    volatile float unusedOut;
    gettimeofday(&tvStart, NULL);
    for (unsigned j = 0; j < ITERATIONS; ++j) {
      for (unsigned i = 0; i < x.size(); ++i) {
        kDot1((int)(N / NumQPUs), &x[i], &y[i], &result);
        unusedOut = std::accumulate(&result[0], &result[0] + result.size, 0.f);
      }
    }
    gettimeofday(&tvEnd, NULL);
    timersub(&tvEnd, &tvStart, &tvDiff);
    printf("%ld.%06lds,", tvDiff.tv_sec, tvDiff.tv_usec);

    ///////////////
    ///////////////
    ///////////////

    // warm-up gpu
    for (unsigned i = 0; i < x.size(); ++i) {
      kDot2((int)(N), &x[i], &y[i], &result);
      (void)std::accumulate(&result[0], &result[0] + result.size, 0.f);
    }

    gettimeofday(&tvStart, NULL);
    for (unsigned j = 0; j < ITERATIONS; ++j) {
      for (unsigned i = 0; i < x.size(); ++i) {
        kDot2((int)(N), &x[i], &y[i], &result);
        unusedOut = std::accumulate(&result[0], &result[0] + result.size, 0.f);
      }
    }
    gettimeofday(&tvEnd, NULL);
    timersub(&tvEnd, &tvStart, &tvDiff);
    printf("%ld.%06lds,", tvDiff.tv_sec, tvDiff.tv_usec);

    ///////////////
    ///////////////
    ///////////////

    gettimeofday(&tvStart, NULL);
    for (unsigned j = 0; j < ITERATIONS; ++j) {
      for (unsigned i = 0; i < x.size(); ++i) {
        unusedOut = cblas_sdot(N, &x[i][0], 1, &y[i][0], 1);
      }
    }
    gettimeofday(&tvEnd, NULL);
    timersub(&tvEnd, &tvStart, &tvDiff);
    printf("%ld.%06lds\n", tvDiff.tv_sec, tvDiff.tv_usec);
  }

  return 0;
}
