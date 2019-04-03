#include <sys/time.h>
#include <math.h>

#include <QPULib.h>

#include <algorithm>
#include <array>
#include <numeric>
#include <random>
#include <chrono>

#include <openblas/cblas.h>

using namespace QPULib;

void dot(Int N, Ptr<Float> A, Ptr<Float> B, Ptr<Float> result) {
  Int inc = 16;
  Int qpuID = me();
  Ptr<Float> a = A + index() + (qpuID << 4);
  Ptr<Float> b = B + index() + (qpuID << 4);
  Ptr<Float> c = result + index() + (qpuID << 4);
  gather(a);
  gather(b);

  Float aOld, bOld;
  For(Int i = 0, i < N, i = i + inc)
    gather(a + i + inc + (qpuID << 4));
    gather(b + i + inc + (qpuID << 4));
    receive(aOld);
    receive(bOld);
    store(aOld * bOld, c + i + (qpuID << 4));
  End
}

class MillisecondTimer {
public:
  MillisecondTimer();

  void Start();

  void Stop();

  void Restart();

  void Reset();

  std::chrono::milliseconds::rep Elapsed();

private:
  std::chrono::system_clock::time_point Now();
  std::chrono::system_clock::duration TimeSinceStart();

  std::chrono::system_clock::time_point _start;
  std::chrono::system_clock::duration _elapsedTime;
  bool _running;
};

MillisecondTimer::MillisecondTimer()
    : _start(std::chrono::system_clock::now()),
      _elapsedTime(std::chrono::system_clock::duration::zero()),
      _running(true) {}

void MillisecondTimer::Start() {
  Reset();
  _running = true;
}

void MillisecondTimer::Stop() {
  _elapsedTime += TimeSinceStart();
  _running = false;
}

void MillisecondTimer::Restart() {
  _start = Now();
  _running = true;
}

void MillisecondTimer::Reset() {
  _start = Now();
  _elapsedTime = _elapsedTime.zero();
}

std::chrono::milliseconds::rep MillisecondTimer::Elapsed() {
  auto elapsed = _running ? TimeSinceStart() + _elapsedTime : _elapsedTime;
  return std::chrono::duration_cast<std::chrono::milliseconds>(elapsed).count();
}

std::chrono::system_clock::time_point MillisecondTimer::Now() {
  return std::chrono::system_clock::now();
}

std::chrono::system_clock::duration MillisecondTimer::TimeSinceStart() {
  return Now() - _start;
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

   // const int N = 16 * 24; // 192000

   // Construct kernel
   auto k = compile(dot);

   // Use 12 QPUs
   k.setNumQPUs(12);

   const int ITERATIONS = 100;
   for (unsigned i = 8; i <= (2 << 4); i *= 2) {
     const unsigned N = 16 * 12 * i;

     // Allocate and initialise arrays shared between ARM and GPU
     std::array<SharedArray<float>, 5> x;
     std::array<SharedArray<float>, 5> y;
     SharedArray<float> result(N);
     std::fill_n(&result[0], N, 0.0f);
     for (unsigned i = 0; i < x.size(); ++i) {
       x[i].alloc(N);
       std::generate_n(&x[i][0], N, [&] { return dist(engine); });
       y[i].alloc(N);
       std::generate_n(&y[i][0], N, [&] { return dist(engine); });
     }

     // warm-up gpu
     for (unsigned i = 0; i < x.size(); ++i) {
       k((int)N, &x[i], &y[i], &result);
       (void)std::accumulate(&result[0], &result[0] + N, 0.f);
     }

     printf("%u,", N);
     volatile float gpuOut = 0.f;
     // MillisecondTimer timer;
     gettimeofday(&tvStart, NULL);
     for (unsigned j = 0; j < ITERATIONS; ++j) {
       for (unsigned i = 0; i < x.size(); ++i) {
         k((int)N, &x[i], &y[i], &result);
         gpuOut = std::accumulate(&result[0], &result[0] + N, 0.f);
       }
     }
     gettimeofday(&tvEnd, NULL);
     timersub(&tvEnd, &tvStart, &tvDiff);
     printf("%ld.%06lds,", tvDiff.tv_sec, tvDiff.tv_usec);
     // timer.Stop();
     // auto gpuTime = timer.Elapsed() / (ITERATIONS * x.size());

     volatile float blasOut = 0.f;
     // timer.Start();
     gettimeofday(&tvStart, NULL);
     for (unsigned j = 0; j < ITERATIONS; ++j) {
       for (unsigned i = 0; i < x.size(); ++i) {
         blasOut = cblas_sdot(N, &x[i][0], 1, &y[i][0], 1);
       }
     }
     gettimeofday(&tvEnd, NULL);
     timersub(&tvEnd, &tvStart, &tvDiff);
     printf("%ld.%06lds\n", tvDiff.tv_sec, tvDiff.tv_usec);
     // timer.Stop();
     // auto blasTime = timer.Elapsed() / (ITERATIONS * x.size());

     // printf("%u,%llu,%llu\n", N, gpuTime, blasTime);
   }
  // gettimeofday(&tvStart, NULL);
  // k(N, &x, &y, &result);
  // float gpuOut = std::accumulate(&result[0], &result[0] + N, 0.f);
  // gettimeofday(&tvEnd, NULL);
  // timersub(&tvEnd, &tvStart, &tvDiff);
  // printf("GPU: %ld.%06lds\n", tvDiff.tv_sec, tvDiff.tv_usec);
  // printf("gpuOutput = %f\n", gpuOut);

  // gettimeofday(&tvStart, NULL);
  // float blasOut = cblas_sdot(N, &x[0], 1, &y[0], 1);
  // gettimeofday(&tvEnd, NULL);
  // timersub(&tvEnd, &tvStart, &tvDiff);
  // printf("BLAS: %ld.%06lds\n", tvDiff.tv_sec, tvDiff.tv_usec);
  // printf("blasOutput = %f\n", blasOut);

  return 0;
}
