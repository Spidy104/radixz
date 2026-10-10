# radixz

Power-of-two complex FFT and FIR convolution in Zig.

- `Fft(T)`: in-place complex FFT, natural order, `f32` or `f64`
- `StaticFft(T, n)`: same, size fixed at compile time, needs no buffers at all
- `Conv(T)`: overlap-save FIR filter, optionally multi-threaded

The library never allocates. You hand it memory, it uses only that.

Requires Zig 0.16.

## Install

```sh
zig fetch --save git+https://github.com/Spidy104/radixz
```

```zig
const radixz = b.dependency("radixz", .{ .target = target, .optimize = optimize });
exe.root_module.addImport("radixz", radixz.module("radixz"));
```

## Example

```zig
const radixz = @import("radixz");

const F = radixz.Fft(f32);
const tables = try gpa.alignedAlloc(f32, .@"64", F.bufferLen(n));
defer gpa.free(tables);
const fft = F.init(tables, n);

fft.forward(re, im, scratch);
fft.inverse(re, im, scratch);
```

`re`, `im` have length `n`, `scratch` has length `2 * n`. Neither direction scales,
so a forward/inverse round trip multiplies by `n`, same as FFTW.

## More

- [API.md](API.md): every public function, its contract and its costs
- [BENCHMARKS.md](BENCHMARKS.md): timings against FFTW and how they were taken

## Tests

```sh
zig build test               # unit tests
zig build stress             # long randomized run
zig build bench -Doptimize=ReleaseFast   # needs FFTW installed
```
