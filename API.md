# API

```zig
const radixz = @import("radixz");
```

## Conventions

These hold for everything below.

**Element type.** `T` is `f32` or `f64`. Other float types compile but are not tested.

**Sizes.** Transform sizes are powers of two, `n >= 2`. Anything else trips an assertion.

**Layout.** Complex data is split: one slice of real parts, one of imaginary parts, same length.
There is no interleaved (`re, im, re, im`) entry point.

**Definition.** `forward` computes `X[k] = sum_j x[j] * exp(-2*pi*i*j*k/n)`.
`inverse` uses `exp(+2*pi*i*j*k/n)`. Neither scales, so `inverse(forward(x)) == n * x`.
This matches FFTW's `FFTW_FORWARD` / `FFTW_BACKWARD`.

**Memory.** Nothing in the library allocates. Plans are built on top of a buffer you pass in
and keep slices into it, so the buffer must outlive the plan. Plans are small structs of
slices and can be copied freely. Buffers passed to `init` must be 64-byte aligned
(`gpa.alignedAlloc(T, .@"64", len)`); data and scratch slices have no alignment requirement.

**Checks.** Contract violations are `std.debug.assert`s: caught in Debug and ReleaseSafe,
undefined behaviour in ReleaseFast.

---

## `Fft(T)`

A plan for one transform size, chosen at runtime.

### `bufferLen(n: usize) usize`

Number of `T` elements `init` needs (`2 * n`, the twiddle tables).

### `init(buf: []align(64) T, n: usize) Fft(T)`

Fills `buf` with twiddle factors and returns the plan. Costs `O(n)` sine/cosine
evaluations; reuse the plan.

Requires `n` a power of two, `n >= 2`, `buf.len >= bufferLen(n)`.

### `forward(self, re: []T, im: []T, scratch: []T) void`
### `inverse(self, re: []T, im: []T, scratch: []T) void`

In-place transform in natural order. `re.len == im.len == n`, `scratch.len >= 2 * n`.

`scratch` is overwritten; its contents before and after the call mean nothing. It must not
overlap `re` or `im`.

The plan is read-only during a call. Several threads may use one plan at the same time as
long as each passes its own `scratch` and its own data.

### `forwardScrambled(self, re: []T, im: []T) void`
### `inverseScrambled(self, re: []T, im: []T) void`

In-place transforms without the reordering step, so no scratch is needed.
`forwardScrambled` leaves the spectrum in an internal permuted order (bit-reversed for
`n < Fft(T).span`, a related blocked order above that). The only promise about that order:
`inverseScrambled` undoes it, and it is the same for every input of the same size.

That is enough for convolution and correlation, which only multiply spectra pointwise:

```zig
fft.forwardScrambled(ar, ai);
fft.forwardScrambled(br, bi);
for (ar, ai, br, bi) |*xr, *xi, yr, yi| {
    const r = xr.* * yr - xi.* * yi;
    xi.* = xr.* * yi + xi.* * yr;
    xr.* = r;
}
fft.inverseScrambled(ar, ai); // circular convolution, scaled by n
```

Do not read individual bins out of a scrambled spectrum.

### `span: usize` (constant)

Block size the vector kernels work in: 64 for `f32`, 32 for `f64` on AVX2.
Smaller transforms take a different code path.

### Tuning field: `block`

`fft.block` (default 4096) is the sub-transform size, in elements, below which the
transform stops recursing and finishes in cache. Any value is correct; it only affects
speed, and only noticeably for `n` above about 100k. Leave the other fields alone.

---

## `StaticFft(T, comptime n)`

The size is fixed at compile time. Twiddle factors are computed by the compiler and end up
in the binary, and the transforms run fully in place, so there is nothing to allocate,
initialise or pass around.

```zig
const F = radixz.StaticFft(f32, 256);
var re: [256]f32 = ...;
var im: [256]f32 = ...;
F.forward(&re, &im);
F.inverse(&re, &im);
```

### `forward(re: *[n]T, im: *[n]T) void`
### `inverse(re: *[n]T, im: *[n]T) void`

Same definition and scaling as `Fft(T)`. Thread-safe: no shared mutable state.

Notes:

- `n >= 2`, power of two.
- Up to `n = 256` for `f32` and `n = 64` for `f64`, both `StaticFft` and `Fft` run the same
  fully unrolled register code, so they are equally fast there. Above that `StaticFft` does
  an in-place bit reversal instead of the fused reordering `Fft` uses, and is slower.
- The twiddle table costs `2 * n * @sizeOf(T)` bytes of read-only data, and compile time
  grows with `n`. Tested up to `n = 4096`.

---

## `Conv(T)`

FIR filter by overlap-save. Computes

```
y[i] = sum_{j=0}^{taps-1} h[j] * x[i - j]      with x[i - j] = 0 for i - j < 0
```

so the output has the same length as the input and starts with the filter's transient.

### `bufferLen(n: usize, workers: usize) usize`

Elements `init` needs: twiddles, the filter's spectrum and one work area per worker.

### `init(buf: []align(64) T, taps: []const T, n: usize, workers: usize) Conv(T)`

`n` is the FFT block size. Requirements:

- `n` a power of two, `n >= Conv(T).span`, `n > taps.len`
- `taps.len >= 1`
- `workers >= 1` (use 1 unless you call `filterParallel`)
- `buf.len >= bufferLen(n, workers)`

`taps` is only read during `init` and can be freed afterwards.

Each block produces `n - taps.len + 1` outputs, so larger `n` wastes less work but uses
more memory per block. Somewhere between `4 * taps.len` and `16 * taps.len` is usually best;
[BENCHMARKS.md](BENCHMARKS.md) lists the sizes that won on the test machine.

### `filter(self, x: []const T, y: []T) void`

Filters all of `x` into `y`. `y.len == x.len`, and `x` and `y` must not overlap.

Uses the first work area in the plan's buffer, so one `Conv` value must not run two
`filter` calls at the same time. Copying the struct does not help, the copies share
the buffer. Use separate plans, or `filterParallel`.

### `filterParallel(self, io: std.Io, x: []const T, y: []T) !void`

Same result as `filter`, bit for bit, split across up to `workers` concurrent tasks using
`std.Io.Group`. Errors are those of `std.Io.Group.concurrent` and `await`
(`error.ConcurrencyUnavailable`, `error.Canceled`); on error `y` is partially written.

Speedup depends on cores and on `n`: work areas larger than the per-core cache stop
scaling early. On the 2+8 core test machine the best result was about 3x.

### `span: usize` (constant)

Smallest allowed block size `n`.
