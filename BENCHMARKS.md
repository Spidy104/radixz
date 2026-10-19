# Benchmarks

All numbers come from `zig build bench -Doptimize=ReleaseFast` (source: `src/bench.zig`).
Run it yourself; the results below are from one laptop and FFT timings move around by
5-10% between runs on it.

## Setup

| | |
|---|---|
| CPU | Intel Core i5-1335U (2 P-cores + 8 E-cores, AVX2 + FMA, no AVX-512) |
| OS | Fedora 44, Linux 7.2, power profile `performance` |
| Zig | 0.16.0, `-Doptimize=ReleaseFast`, native CPU |
| FFTW | 3.3.10 from the distribution packages (`fftw-libs-single`, `fftw-libs-double`) |

## Method

- Every case is checked against FFTW's output before it is timed. The `max err` column is
  the largest difference divided by the largest FFTW output value.
- A sample is a batch of repetitions (enough for about 2 million points of work, at least
  one). Eight samples are taken, the first is dropped as warm-up, and the median of the
  rest is reported.
- One thread unless stated. Both libraries run in the same process, in the same loop, on the
  same data.
- GFLOPS uses the usual FFT convention of `5 n log2(n)` flops per complex transform.

### FFT

- radixz: `Fft(T).forward` on split `re`/`im` arrays, in place, natural order.
- FFTW: `fftw_plan_dft_1d`, in place, interleaved, `FFTW_MEASURE`. This is FFTW's fastest
  configuration on this machine (its split-format guru plans are slower), so it is the one
  compared against.
- Data: uniform random in [-1, 1].

### Convolution

The task is filtering `4194304 - taps + 1` samples with a random FIR filter, output length
equal to input length. That length makes FFTW's one-shot transform exactly `2^22` points.

- radixz: `Conv(f32).filter`, block size picked as the fastest of `2^k` from the smallest
  allowed to 16x that.
- FFTW overlap-save: the same overlap-save algorithm written around FFTW `r2c`/`c2r` plans
  (`FFTW_MEASURE`), block size swept the same way. This is the fair comparison.
- FFTW one-shot: one `r2c`, a pointwise multiply and one `c2r` over the whole signal,
  which is how FFTW convolution is usually done.

## FFT, f32

| n | radixz ns | FFTW ns | radixz GFLOPS | FFTW GFLOPS | speed vs FFTW | max err |
|---:|---:|---:|---:|---:|---:|---:|
| 16 | 14.6 | 14.2 | 21.9 | 22.6 | 0.97x | 1.0e-7 |
| 32 | 26.6 | 28.6 | 30.1 | 28.0 | 1.07x | 1.0e-7 |
| 64 | 38.3 | 40.9 | 50.1 | 47.0 | 1.07x | 6.1e-8 |
| 128 | 85.1 | 68.9 | 52.7 | 65.0 | 0.81x | 1.0e-7 |
| 256 | 146.5 | 157.8 | 69.9 | 64.9 | 1.08x | 1.1e-7 |
| 512 | 282.9 | 275.1 | 81.4 | 83.8 | 0.97x | 1.2e-7 |
| 1024 | 583.8 | 590.4 | 87.7 | 86.7 | 1.01x | 1.9e-7 |
| 2048 | 1346.5 | 1250.1 | 83.7 | 90.1 | 0.93x | 1.9e-7 |
| 4096 | 3781.0 | 3107.9 | 65.0 | 79.1 | 0.82x | 1.9e-7 |
| 8192 | 10078.5 | 8778.3 | 52.8 | 60.7 | 0.87x | 1.4e-7 |
| 16384 | 20612.8 | 19234.3 | 55.6 | 59.6 | 0.93x | 2.0e-7 |
| 32768 | 45961.0 | 42386.0 | 53.5 | 58.0 | 0.92x | 1.9e-7 |
| 65536 | 105797 | 123249 | 49.6 | 42.5 | 1.16x | 2.3e-7 |
| 131072 | 243482 | 322583 | 45.8 | 34.5 | 1.32x | 2.6e-7 |
| 262144 | 508713 | 634067 | 46.4 | 37.2 | 1.25x | 2.4e-7 |
| 524288 | 1161309 | 2944928 | 42.9 | 16.9 | 2.54x | 2.6e-7 |
| 1048576 | 3164375 | 6667069 | 33.1 | 15.7 | 2.11x | 2.5e-7 |
| 2097152 | 8744857 | 14412467 | 25.2 | 15.3 | 1.65x | 2.5e-7 |
| 4194304 | 19439808 | 30023440 | 23.7 | 15.4 | 1.54x | 2.8e-7 |

## FFT, f64

| n | radixz ns | FFTW ns | radixz GFLOPS | FFTW GFLOPS | speed vs FFTW | max err |
|---:|---:|---:|---:|---:|---:|---:|
| 16 | 14.7 | 13.7 | 21.8 | 23.3 | 0.93x | 1.8e-16 |
| 32 | 28.3 | 27.2 | 28.3 | 29.5 | 0.96x | 2.2e-16 |
| 64 | 60.6 | 62.6 | 31.7 | 30.7 | 1.03x | 2.6e-16 |
| 128 | 128.0 | 149.2 | 35.0 | 30.0 | 1.17x | 4.2e-16 |
| 256 | 292.0 | 264.7 | 35.1 | 38.7 | 0.91x | 4.0e-16 |
| 512 | 611.5 | 501.1 | 37.7 | 46.0 | 0.82x | 4.7e-16 |
| 1024 | 1388.7 | 1180.3 | 36.9 | 43.4 | 0.85x | 5.0e-16 |
| 2048 | 3803.4 | 3528.1 | 29.6 | 31.9 | 0.93x | 5.4e-16 |
| 4096 | 10368.7 | 7372.6 | 23.7 | 33.3 | 0.71x | 5.7e-16 |
| 8192 | 21617.8 | 17225.3 | 24.6 | 30.9 | 0.80x | 7.2e-16 |
| 16384 | 49140.5 | 39732.8 | 23.3 | 28.9 | 0.81x | 6.8e-16 |
| 32768 | 100958.5 | 96838.9 | 24.3 | 25.4 | 0.96x | 7.5e-16 |
| 65536 | 236050 | 212010 | 22.2 | 24.7 | 0.90x | 7.2e-16 |
| 131072 | 556287 | 482584 | 20.0 | 23.1 | 0.87x | 7.1e-16 |
| 262144 | 1003352 | 1118050 | 23.5 | 21.1 | 1.11x | 7.5e-16 |
| 524288 | 2946914 | 3807416 | 16.9 | 13.1 | 1.29x | 7.7e-16 |
| 1048576 | 7854403 | 8604635 | 13.4 | 12.2 | 1.10x | 8.9e-16 |
| 2097152 | 18011900 | 18068232 | 12.2 | 12.2 | 1.00x | 1.0e-15 |

### What this says

- f32: radixz is faster than FFTW from 65536 points up, by 1.2x to 2.5x, mostly because of
  cache blocking. Below that it is within about 20% either way: ahead at 32, 64, 256 and
  1024, behind at 128 and between 2048 and 32768.
- f64: mostly behind FFTW between 256 and 131072 points (0.71x to 0.96x), ahead at 64, 128
  and from 262144 up.
- FFTW's remaining advantage in the middle sizes comes from its generated, register-scheduled
  codelets. Things tried here that did not close the gap: radix-8 passes (register spills),
  an interleaved-layout Stockham core, keeping passes L1-resident, smaller cache blocks.

## Real-input FFT

zconv: `Rfft(T).forward`, `x` (n reals) to `re`, `im` (n/2 + 1 bins each), out of place.
FFTW: `fftw_plan_dft_r2c_1d`, out of place, `FFTW_MEASURE`, n/2 + 1 interleaved complex outputs.
Each row is the mean of two complete runs. The two runs agreed to about 3% for f64, but
differed by up to 25% in a few small f32 rows (64 to 8192 points, in both libraries at
once), so ratios there are good to about plus or minus 15%.

### Real FFT, f32

| n | zconv ns | FFTW ns | speed vs FFTW | max err |
|---:|---:|---:|---:|---:|
| 16 | 18.5 | 11.9 | 0.65x | 1.3e-07 |
| 32 | 31.5 | 29.0 | 0.92x | 8.7e-08 |
| 64 | 48.3 | 42.6 | 0.88x | 7.9e-08 |
| 128 | 72.8 | 63.4 | 0.87x | 1.7e-07 |
| 256 | 138.2 | 115.3 | 0.83x | 1.3e-07 |
| 512 | 241.0 | 218.3 | 0.91x | 1.8e-07 |
| 1024 | 463.4 | 388.6 | 0.84x | 1.6e-07 |
| 2048 | 878.4 | 852.0 | 0.97x | 2.0e-07 |
| 4096 | 2193 | 2021 | 0.92x | 2.1e-07 |
| 8192 | 4778 | 4385 | 0.92x | 2.0e-07 |
| 16384 | 12256 | 12396 | 1.01x | 2.3e-07 |
| 32768 | 24945 | 26707 | 1.07x | 3.2e-07 |
| 65536 | 54924 | 60000 | 1.09x | 2.8e-07 |
| 131072 | 120036 | 144789 | 1.21x | 2.8e-07 |
| 262144 | 265502 | 345449 | 1.30x | 2.0e-07 |
| 524288 | 597724 | 734222 | 1.23x | 2.6e-07 |
| 1048576 | 1473492 | 1909670 | 1.30x | 2.7e-07 |
| 2097152 | 3897180 | 4373550 | 1.12x | 2.6e-07 |
| 4194304 | 9971563 | 10739204 | 1.08x | 2.7e-07 |

### Real FFT, f64

| n | zconv ns | FFTW ns | speed vs FFTW | max err |
|---:|---:|---:|---:|---:|
| 16 | 32.0 | 18.1 | 0.57x | 1.3e-16 |
| 32 | 40.6 | 33.0 | 0.81x | 2.1e-16 |
| 64 | 59.8 | 51.7 | 0.86x | 2.2e-16 |
| 128 | 107.2 | 86.1 | 0.80x | 3.7e-16 |
| 256 | 182.6 | 153.2 | 0.84x | 3.1e-16 |
| 512 | 392.4 | 315.0 | 0.80x | 5.9e-16 |
| 1024 | 828.8 | 648.0 | 0.78x | 4.2e-16 |
| 2048 | 2092 | 1556 | 0.74x | 4.8e-16 |
| 4096 | 4648 | 3553 | 0.76x | 5.8e-16 |
| 8192 | 12186 | 8851 | 0.73x | 6.6e-16 |
| 16384 | 25600 | 19370 | 0.76x | 6.4e-16 |
| 32768 | 56541 | 41656 | 0.74x | 6.9e-16 |
| 65536 | 121078 | 103230 | 0.85x | 6.9e-16 |
| 131072 | 276559 | 264188 | 0.96x | 7.8e-16 |
| 262144 | 597276 | 540484 | 0.90x | 7.8e-16 |
| 524288 | 1461779 | 1338067 | 0.92x | 8.3e-16 |
| 1048576 | 3764652 | 3570978 | 0.95x | 7.9e-16 |
| 2097152 | 9350931 | 8857788 | 0.95x | 9.8e-16 |

### What this says

- f32: faster than FFTW from 16384 points up, by 1.08x to 1.30x. Below that it is behind,
  between 0.65x and 1.07x (worst at 16 points).
- f64: behind FFTW at every size, between 0.57x and 0.96x. The gap is smallest for large n.
- The real FFT is built on the complex one, so it inherits the same pattern: the cache
  blocking pays off for large sizes, FFTW's generated codelets win in the middle.

## Complex-typed API

`Fft(T).forwardComplex` on a `[]Complex` against `Fft(T).forward` on split arrays, same
plan, same data, in place. The two give identical results. Single run per row.

### f32

| n | split ns | complex ns | complex / split |
|---:|---:|---:|---:|
| 16 | 15.1 | 23.4 | 1.55x |
| 32 | 26.1 | 37.9 | 1.45x |
| 64 | 38.9 | 59.3 | 1.52x |
| 128 | 84.9 | 126.1 | 1.49x |
| 256 | 146.3 | 227.9 | 1.56x |
| 512 | 292.0 | 409.1 | 1.40x |
| 1024 | 607.6 | 836.2 | 1.38x |
| 2048 | 1336 | 1838 | 1.38x |
| 4096 | 3784 | 3926 | 1.04x |
| 8192 | 10147 | 8443 | 0.83x |
| 16384 | 20891 | 17773 | 0.85x |
| 32768 | 45841 | 39043 | 0.85x |
| 65536 | 105279 | 86148 | 0.82x |
| 131072 | 245707 | 209687 | 0.85x |
| 262144 | 545126 | 462568 | 0.85x |
| 524288 | 1272532 | 1059787 | 0.83x |
| 1048576 | 3472347 | 3190696 | 0.92x |

### f64

| n | split ns | complex ns | complex / split |
|---:|---:|---:|---:|
| 16 | 15.3 | 28.3 | 1.85x |
| 32 | 27.6 | 44.2 | 1.60x |
| 64 | 55.3 | 83.5 | 1.51x |
| 128 | 120.4 | 145.3 | 1.21x |
| 256 | 265.3 | 324.6 | 1.22x |
| 512 | 573.1 | 701.9 | 1.22x |
| 1024 | 1272 | 1535 | 1.21x |
| 2048 | 3878 | 3707 | 0.96x |
| 4096 | 10251 | 8361 | 0.82x |
| 8192 | 21497 | 17242 | 0.80x |
| 16384 | 49646 | 39740 | 0.80x |
| 32768 | 103629 | 82257 | 0.79x |
| 65536 | 244154 | 211626 | 0.87x |
| 131072 | 532253 | 469440 | 0.88x |
| 262144 | 1194827 | 1123167 | 0.94x |
| 524288 | 3329518 | 3268214 | 0.98x |
| 1048576 | 8577282 | 8423588 | 0.98x |

Below about 2048 points the conversion between layouts costs 20% to 85% extra. From 4096
points up it is folded into the first and last pass and the typed API is equal or up to
about 20% faster, since those passes read and write interleaved data directly.

## Convolution, f32, one thread

| taps | block | radixz ms | FFTW one-shot ms | FFTW overlap-save ms | vs one-shot | vs overlap-save | max err |
|---:|---:|---:|---:|---:|---:|---:|---:|
| 32 | 128 | 4.21 | 27.62 | 6.93 | 6.56x | 1.64x | 3.2e-7 |
| 128 | 2048 | 4.98 | 27.88 | 7.63 | 5.60x | 1.53x | 3.4e-7 |
| 512 | 4096 | 5.60 | 27.41 | 8.14 | 4.90x | 1.45x | 3.3e-7 |
| 2048 | 16384 | 6.28 | 27.47 | 9.99 | 4.38x | 1.59x | 4.0e-7 |
| 8192 | 65536 | 7.41 | 27.56 | 12.24 | 3.72x | 1.65x | 4.1e-7 |
| 32768 | 262144 | 9.36 | 27.44 | 17.43 | 2.93x | 1.86x | 5.5e-7 |

## Convolution, f32, threaded

radixz uses `filterParallel`. FFTW overlap-save gets the same work-stealing split across
threads, sharing one plan and giving each thread its own buffers (FFTW's thread-safe
new-array execute functions); FFTW one-shot uses `libfftw3f_omp` with the same thread
count. Block sizes are swept per thread count for both.

| threads | taps | radixz ms | FFTW one-shot ms | FFTW overlap-save ms | vs overlap-save | radixz speedup over 1 thread |
|---:|---:|---:|---:|---:|---:|---:|
| 1 | 128 | 5.03 | 27.15 | 7.69 | 1.53x | 1.00x |
| 1 | 2048 | 6.28 | 27.09 | 10.18 | 1.62x | 1.00x |
| 1 | 32768 | 9.40 | 27.07 | 17.42 | 1.85x | 1.00x |
| 2 | 128 | 3.31 | 35.21 | 4.28 | 1.29x | 1.52x |
| 2 | 2048 | 3.75 | 23.47 | 5.47 | 1.46x | 1.67x |
| 2 | 32768 | 6.23 | 22.69 | 11.31 | 1.82x | 1.51x |
| 4 | 128 | 3.14 | 28.03 | 3.48 | 1.11x | 1.60x |
| 4 | 2048 | 3.37 | 26.79 | 4.50 | 1.33x | 1.86x |
| 4 | 32768 | 6.10 | 26.47 | 10.28 | 1.68x | 1.54x |
| 8 | 128 | 2.61 | 27.39 | 2.97 | 1.14x | 1.93x |
| 8 | 2048 | 3.17 | 29.09 | 4.36 | 1.38x | 1.98x |
| 8 | 32768 | 5.10 | 27.80 | 9.01 | 1.77x | 1.84x |
| 12 | 128 | 2.10 | 30.99 | 2.51 | 1.20x | 2.40x |
| 12 | 2048 | 2.65 | 32.56 | 4.10 | 1.55x | 2.37x |
| 12 | 32768 | 5.10 | 27.02 | 10.16 | 1.99x | 1.84x |

Scaling stops around 2.4x on this CPU: it has 2 performance cores (4 threads) and 8 slower
efficiency cores, and with large filters each worker's block no longer fits in its cache.

## Reproducing

```sh
zig build bench -Doptimize=ReleaseFast           # everything
zig build bench -Doptimize=ReleaseFast -- fft    # complex FFT tables only
zig build bench -Doptimize=ReleaseFast -- rfft   # real-input FFT tables only
zig build bench -Doptimize=ReleaseFast -- cx     # complex-typed API vs split API
zig build bench -Doptimize=ReleaseFast -- conv   # convolution tables only
```

FFTW (single and double precision, plus `fftw3f_omp`) must be installed where the system
linker finds it. `FFTW_MEASURE` planning for the large sizes takes a while.
