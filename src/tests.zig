const std = @import("std");
const zc = @import("radixz");
const stress = @import("options").stress;

const rounds: usize = if (stress) 4000 else 60;

fn direct(comptime T: type, h: []const T, x: []const T, y: []T) void {
    for (y, 0..) |*o, n| {
        var s: f64 = 0;
        for (h, 0..) |hj, j| {
            if (j > n) break;
            s += @as(f64, hj) * @as(f64, x[n - j]);
        }
        o.* = @floatCast(s);
    }
}

fn fill(comptime T: type, r: std.Random, v: []T) void {
    for (v) |*e| e.* = r.float(T) * 2 - 1;
}

fn expectClose(comptime T: type, h: []const T, got: []const T, want: []const T) !void {
    var scale: f64 = 0;
    for (h) |v| scale += @abs(@as(f64, v));
    const tol = @max(scale, 1) * @as(f64, if (T == f32) 3e-6 else 1e-14) * 8;
    for (got, want, 0..) |g, w, i| {
        const e = @abs(@as(f64, g) - @as(f64, w));
        if (!(e <= tol)) {
            std.debug.print("{s} taps={d} len={d} i={d} got={d} want={d} err={e} tol={e}\n", .{ @typeName(T), h.len, got.len, i, g, w, e, tol });
            return error.TestExpectedApproxEq;
        }
    }
}

fn Owned(comptime T: type) type {
    return struct {
        buf: []align(64) T,
        c: zc.Conv(T),

        fn init(gpa: std.mem.Allocator, h: []const T, n: usize, workers: usize) !@This() {
            const buf = try gpa.alignedAlloc(T, .@"64", zc.Conv(T).bufferLen(n, workers));
            return .{ .buf = buf, .c = .init(buf, h, n, workers) };
        }

        fn deinit(self: *@This(), gpa: std.mem.Allocator) void {
            gpa.free(self.buf);
        }
    };
}

fn run(comptime T: type, gpa: std.mem.Allocator, h: []const T, x: []const T, n: usize) !void {
    const y = try gpa.alloc(T, x.len);
    defer gpa.free(y);
    const want = try gpa.alloc(T, x.len);
    defer gpa.free(want);
    var o = try Owned(T).init(gpa, h, n, 1);
    defer o.deinit(gpa);
    @memset(y, std.math.nan(T));
    o.c.filter(x, y);
    direct(T, h, x, want);
    try expectClose(T, h, y, want);
    @memset(y, std.math.nan(T));
    o.c.filter(x, y);
    try expectClose(T, h, y, want);
}

fn sweep(comptime T: type) !void {
    const gpa = std.testing.allocator;
    var prng = std.Random.DefaultPrng.init(42);
    const r = prng.random();
    const span = zc.Conv(T).span;
    const xs = try gpa.alloc(T, 1 << 15);
    defer gpa.free(xs);
    const hs = try gpa.alloc(T, 1 << 12);
    defer gpa.free(hs);
    fill(T, r, xs);
    fill(T, r, hs);
    var n: usize = span;
    while (n <= 1 << 13) : (n *= 2) {
        const tap_list = [_]usize{ 1, 2, 3, 5, n / 4, n / 2 - 1, n / 2, n / 2 + 1, n - 2, n - 1 };
        for (tap_list) |taps| {
            if (taps == 0 or taps >= n or taps > hs.len) continue;
            const l = n - taps + 1;
            const len_list = [_]usize{ 0, 1, 2, taps, l - 1, l, l + 1, 2 * l - 1, 2 * l, 2 * l + 1, 3 * l + 7, 5 * n + 3 };
            for (len_list) |len| {
                if (len > xs.len or len * taps > 1 << 24) continue;
                try run(T, gpa, hs[0..taps], xs[0..len], n);
            }
        }
    }
}

test "convolution exhaustive sweep f32" {
    try sweep(f32);
}

test "convolution exhaustive sweep f64" {
    try sweep(f64);
}

test "convolution random shapes" {
    const gpa = std.testing.allocator;
    var prng = std.Random.DefaultPrng.init(7);
    const r = prng.random();
    inline for (.{ f32, f64 }) |T| {
        for (0..rounds) |_| {
            const n = @as(usize, 1) << r.intRangeAtMost(u6, std.math.log2_int(usize, zc.Conv(T).span), 12);
            const taps = r.intRangeAtMost(usize, 1, n - 1);
            const len = r.intRangeAtMost(usize, 0, 6000);
            const h = try gpa.alloc(T, taps);
            defer gpa.free(h);
            const x = try gpa.alloc(T, len);
            defer gpa.free(x);
            fill(T, r, h);
            fill(T, r, x);
            try run(T, gpa, h, x, n);
        }
    }
}

test "impulse returns the taps" {
    const gpa = std.testing.allocator;
    const h = [_]f64{ 1, -2, 3, -4, 5 };
    var x: [300]f64 = @splat(0);
    x[0] = 1;
    var y: [300]f64 = undefined;
    var o = try Owned(f64).init(gpa, &h, 64, 1);
    defer o.deinit(gpa);
    o.c.filter(&x, &y);
    for (h, y[0..h.len]) |p, q| try std.testing.expectApproxEqAbs(p, q, 1e-12);
    for (y[h.len..]) |q| try std.testing.expectApproxEqAbs(0, q, 1e-12);
}

test "shifted impulse lands at its offset across block borders" {
    const gpa = std.testing.allocator;
    const h = [_]f64{ 0.5, 0.25, 0.125 };
    var o = try Owned(f64).init(gpa, &h, 64, 1);
    defer o.deinit(gpa);
    for (0..200) |k| {
        var x: [200]f64 = @splat(0);
        var y: [200]f64 = undefined;
        x[k] = 1;
        o.c.filter(&x, &y);
        for (y, 0..) |v, i| {
            const want: f64 = if (i >= k and i - k < h.len) h[i - k] else 0;
            try std.testing.expectApproxEqAbs(want, v, 1e-12);
        }
    }
}

test "dc and nyquist steady state" {
    const gpa = std.testing.allocator;
    const h = [_]f64{ 1, 2, 3, 4 };
    var o = try Owned(f64).init(gpa, &h, 64, 1);
    defer o.deinit(gpa);
    var x: [500]f64 = undefined;
    var y: [500]f64 = undefined;
    @memset(&x, 1);
    o.c.filter(&x, &y);
    for (y[h.len..]) |v| try std.testing.expectApproxEqAbs(10, v, 1e-11);
    for (&x, 0..) |*v, i| v.* = if (i % 2 == 0) 1 else -1;
    o.c.filter(&x, &y);
    for (y[h.len..], h.len..) |v, i| try std.testing.expectApproxEqAbs(if (i % 2 == 0) @as(f64, -2) else 2, v, 1e-11);
}

test "linearity and zero input" {
    const gpa = std.testing.allocator;
    var prng = std.Random.DefaultPrng.init(3);
    const r = prng.random();
    var h: [37]f64 = undefined;
    var a: [1000]f64 = undefined;
    var b: [1000]f64 = undefined;
    var s: [1000]f64 = undefined;
    fill(f64, r, &h);
    fill(f64, r, &a);
    fill(f64, r, &b);
    for (&s, a, b) |*d, p, q| d.* = 2 * p - 3 * q;
    var o = try Owned(f64).init(gpa, &h, 128, 1);
    defer o.deinit(gpa);
    var ya: [1000]f64 = undefined;
    var yb: [1000]f64 = undefined;
    var ys: [1000]f64 = undefined;
    o.c.filter(&a, &ya);
    o.c.filter(&b, &yb);
    o.c.filter(&s, &ys);
    for (ys, ya, yb) |v, p, q| try std.testing.expectApproxEqAbs(2 * p - 3 * q, v, 1e-11);
    @memset(&s, 0);
    o.c.filter(&s, &ys);
    for (ys) |v| try std.testing.expectEqual(@as(f64, 0), v);
}

test "parallel equals serial under random workers and lengths" {
    const gpa = std.testing.allocator;
    var prng = std.Random.DefaultPrng.init(11);
    const r = prng.random();
    inline for (.{ f32, f64 }) |T| {
        for (0..rounds / 4 + 8) |_| {
            const workers = r.intRangeAtMost(usize, 1, 16);
            const n = @as(usize, 1) << r.intRangeAtMost(u6, std.math.log2_int(usize, zc.Conv(T).span), 12);
            const taps = r.intRangeAtMost(usize, 1, n - 1);
            const len = r.intRangeAtMost(usize, 0, 60000);
            const h = try gpa.alloc(T, taps);
            defer gpa.free(h);
            const x = try gpa.alloc(T, len);
            defer gpa.free(x);
            const a = try gpa.alloc(T, len);
            defer gpa.free(a);
            const b = try gpa.alloc(T, len);
            defer gpa.free(b);
            fill(T, r, h);
            fill(T, r, x);
            var o = try Owned(T).init(gpa, h, n, workers);
            defer o.deinit(gpa);
            o.c.filter(x, a);
            @memset(b, std.math.nan(T));
            try o.c.filterParallel(std.testing.io, x, b);
            try std.testing.expectEqualSlices(T, a, b);
        }
    }
}

fn dft(comptime inv: bool, xr: []const f64, xi: []const f64, yr: []f64, yi: []f64) void {
    const n = xr.len;
    for (yr, yi, 0..) |*or_, *oi, k| {
        var sr: f64 = 0;
        var si: f64 = 0;
        for (xr, xi, 0..) |a, b, j| {
            const ang = (if (inv) @as(f64, 2) else -2) * std.math.pi * @as(f64, @floatFromInt((j * k) % n)) / @as(f64, @floatFromInt(n));
            sr += a * @cos(ang) - b * @sin(ang);
            si += a * @sin(ang) + b * @cos(ang);
        }
        or_.* = sr;
        oi.* = si;
    }
}

const FftCase = struct {
    fn check(comptime T: type, gpa: std.mem.Allocator, comptime inv: bool, xr: []const T, xi: []const T) !void {
        const n = xr.len;
        const buf = try gpa.alignedAlloc(T, .@"64", zc.Fft(T).bufferLen(n));
        defer gpa.free(buf);
        const f = zc.Fft(T).init(buf, n);
        const scratch = try gpa.alloc(T, 2 * n);
        defer gpa.free(scratch);
        @memset(scratch, std.math.nan(T));
        const re = try gpa.dupe(T, xr);
        defer gpa.free(re);
        const im = try gpa.dupe(T, xi);
        defer gpa.free(im);
        const r64 = try gpa.alloc(f64, 4 * n);
        defer gpa.free(r64);
        for (xr, xi, r64[0..n], r64[n .. 2 * n]) |a, b, *p, *q| {
            p.* = a;
            q.* = b;
        }
        dft(inv, r64[0..n], r64[n .. 2 * n], r64[2 * n .. 3 * n], r64[3 * n ..]);
        if (inv) f.inverse(re, im, scratch) else f.forward(re, im, scratch);
        var peak: f64 = 1;
        for (r64[2 * n .. 3 * n], r64[3 * n ..]) |a, b| peak = @max(peak, @abs(a), @abs(b));
        const tol = peak * @as(f64, if (T == f32) 4e-7 else 1e-15) * @as(f64, @floatFromInt(std.math.log2_int(usize, n) + 2)) * 4;
        for (re, im, r64[2 * n .. 3 * n], r64[3 * n ..], 0..) |a, b, wa, wb, k| {
            const e = @max(@abs(@as(f64, a) - wa), @abs(@as(f64, b) - wb));
            if (!(e <= tol)) {
                std.debug.print("fft {s} inv={} n={d} k={d} got=({d},{d}) want=({d},{d}) err={e} tol={e}\n", .{ @typeName(T), inv, n, k, a, b, wa, wb, e, tol });
                return error.TestExpectedApproxEq;
            }
        }
    }
};

test "fft matches naive dft for every power of two up to 4096" {
    const gpa = std.testing.allocator;
    var prng = std.Random.DefaultPrng.init(5);
    const r = prng.random();
    inline for (.{ f32, f64 }) |T| {
        var n: usize = 2;
        while (n <= 4096) : (n *= 2) {
            const xr = try gpa.alloc(T, n);
            defer gpa.free(xr);
            const xi = try gpa.alloc(T, n);
            defer gpa.free(xi);
            fill(T, r, xr);
            fill(T, r, xi);
            try FftCase.check(T, gpa, false, xr, xi);
            try FftCase.check(T, gpa, true, xr, xi);
            @memset(xr, 0);
            @memset(xi, 0);
            xr[0] = 1;
            try FftCase.check(T, gpa, false, xr, xi);
            @memset(xr, 1);
            try FftCase.check(T, gpa, false, xr, xi);
            for (xr, 0..) |*v, i| v.* = if (i % 2 == 0) 1 else -1;
            try FftCase.check(T, gpa, false, xr, xi);
            const k = n / 3 + 1;
            for (xr, xi, 0..) |*a, *b, i| {
                const ang = 2 * std.math.pi * @as(f64, @floatFromInt((i * k) % n)) / @as(f64, @floatFromInt(n));
                a.* = @floatCast(@cos(ang));
                b.* = @floatCast(@sin(ang));
            }
            try FftCase.check(T, gpa, false, xr, xi);
        }
    }
}

test "fft roundtrip parseval and linearity up to 2^20" {
    const gpa = std.testing.allocator;
    var prng = std.Random.DefaultPrng.init(8);
    const r = prng.random();
    inline for (.{ f32, f64 }) |T| {
        var n: usize = 2;
        while (n <= 1 << 20) : (n *= 2) {
            const buf = try gpa.alignedAlloc(T, .@"64", zc.Fft(T).bufferLen(n));
            defer gpa.free(buf);
            const f = zc.Fft(T).init(buf, n);
            const scratch = try gpa.alloc(T, 2 * n);
            defer gpa.free(scratch);
            const v = try gpa.alloc(T, 6 * n);
            defer gpa.free(v);
            const xr = v[0..n];
            const xi = v[n .. 2 * n];
            const ar = v[2 * n .. 3 * n];
            const ai = v[3 * n .. 4 * n];
            const br = v[4 * n .. 5 * n];
            const bi = v[5 * n ..];
            fill(T, r, xr);
            fill(T, r, xi);
            fill(T, r, br);
            fill(T, r, bi);
            @memcpy(ar, xr);
            @memcpy(ai, xi);
            f.forward(ar, ai, scratch);
            var ex: f64 = 0;
            var ef: f64 = 0;
            for (xr, xi, ar, ai) |a, b, c, d| {
                ex += @as(f64, a) * a + @as(f64, b) * b;
                ef += @as(f64, c) * c + @as(f64, d) * d;
            }
            const nf: f64 = @floatFromInt(n);
            const eps: f64 = if (T == f32) 1e-5 else 1e-13;
            try std.testing.expectApproxEqRel(ex * nf, ef, eps * 20);
            f.inverse(ar, ai, scratch);
            var worst: f64 = 0;
            for (xr, xi, ar, ai) |a, b, c, d| worst = @max(worst, @abs(@as(f64, c) / nf - a), @abs(@as(f64, d) / nf - b));
            try std.testing.expect(worst < eps * @as(f64, @floatFromInt(std.math.log2_int(usize, n) + 1)));
            @memcpy(ar, xr);
            @memcpy(ai, xi);
            for (ar, ai, br, bi) |*a, *b, c, d| {
                a.* = 2 * a.* - 3 * c;
                b.* = 2 * b.* - 3 * d;
            }
            f.forward(ar, ai, scratch);
            f.forward(xr, xi, scratch);
            f.forward(br, bi, scratch);
            var lin: f64 = 0;
            var peak: f64 = 1;
            for (ar, ai, xr, xi, br, bi) |s, t, a, b, c, d| {
                lin = @max(lin, @abs(@as(f64, s) - (2 * @as(f64, a) - 3 * @as(f64, c))), @abs(@as(f64, t) - (2 * @as(f64, b) - 3 * @as(f64, d))));
                peak = @max(peak, @abs(@as(f64, s)), @abs(@as(f64, t)));
            }
            try std.testing.expect(lin <= peak * eps * 10);
        }
    }
}

test "fft random fuzz against naive dft" {
    const gpa = std.testing.allocator;
    var prng = std.Random.DefaultPrng.init(21);
    const r = prng.random();
    inline for (.{ f32, f64 }) |T| {
        for (0..rounds / 2 + 4) |_| {
            const n = @as(usize, 1) << r.intRangeAtMost(u6, 1, if (stress) 11 else 9);
            const xr = try gpa.alloc(T, n);
            defer gpa.free(xr);
            const xi = try gpa.alloc(T, n);
            defer gpa.free(xi);
            const amp: T = @floatCast(std.math.pow(f64, 10, r.float(f64) * 8 - 4));
            for (xr, xi) |*a, *b| {
                a.* = (r.float(T) * 2 - 1) * amp;
                b.* = (r.float(T) * 2 - 1) * amp;
            }
            if (r.boolean()) try FftCase.check(T, gpa, true, xr, xi) else try FftCase.check(T, gpa, false, xr, xi);
        }
    }
}

test "library source never touches an allocator" {
    const src = @embedFile("radixz.zig");
    try std.testing.expect(std.mem.indexOf(u8, src, "Allocator") == null);
    try std.testing.expect(std.mem.indexOf(u8, src, "alloc") == null);
    try std.testing.expect(std.mem.indexOf(u8, src, "heap") == null);
}

test "cache-blocked recursion is bit-identical to flat passes" {
    const gpa = std.testing.allocator;
    var prng = std.Random.DefaultPrng.init(31);
    const r = prng.random();
    inline for (.{ f32, f64 }) |T| {
        const F = zc.Fft(T);
        var n: usize = F.span;
        while (n <= 1 << 18) : (n *= 2) {
            const buf = try gpa.alignedAlloc(T, .@"64", F.bufferLen(n));
            defer gpa.free(buf);
            var deep = F.init(buf, n);
            deep.block = F.span;
            var flat = deep;
            flat.block = std.math.maxInt(usize);
            const v = try gpa.alloc(T, 6 * n);
            defer gpa.free(v);
            fill(T, r, v[0 .. 2 * n]);
            @memcpy(v[2 * n .. 4 * n], v[0 .. 2 * n]);
            inline for (.{ false, true }) |inv| {
                if (inv) {
                    deep.inverse(v[0..n], v[n .. 2 * n], v[4 * n ..]);
                    flat.inverse(v[2 * n .. 3 * n], v[3 * n .. 4 * n], v[4 * n ..]);
                } else {
                    deep.forward(v[0..n], v[n .. 2 * n], v[4 * n ..]);
                    flat.forward(v[2 * n .. 3 * n], v[3 * n .. 4 * n], v[4 * n ..]);
                }
                try std.testing.expectEqualSlices(T, v[2 * n .. 4 * n], v[0 .. 2 * n]);
            }
        }
        const taps = 300;
        const len = 200000;
        const h = try gpa.alloc(T, taps);
        defer gpa.free(h);
        const x = try gpa.alloc(T, len);
        defer gpa.free(x);
        const a = try gpa.alloc(T, len);
        defer gpa.free(a);
        const b = try gpa.alloc(T, len);
        defer gpa.free(b);
        fill(T, r, h);
        fill(T, r, x);
        var o = try Owned(T).init(gpa, h, 1 << 16, 1);
        defer o.deinit(gpa);
        o.c.fft.block = std.math.maxInt(usize);
        o.c.filter(x, a);
        o.c.fft.block = F.span;
        o.c.filter(x, b);
        try std.testing.expectEqualSlices(T, a, b);
    }
}

test "static fft needs no memory and matches the dynamic plan" {
    const gpa = std.testing.allocator;
    var prng = std.Random.DefaultPrng.init(77);
    const r = prng.random();
    inline for (.{ f32, f64 }) |T| {
        inline for (.{ 2, 4, 8, 16, 32, 64, 128, 256, 512, 1024, 4096 }) |n| {
            const S = zc.StaticFft(T, n);
            var re: [n]T = undefined;
            var im: [n]T = undefined;
            fill(T, r, &re);
            fill(T, r, &im);
            const xr = re;
            const xi = im;
            try FftCase.check(T, gpa, false, &xr, &xi);
            const buf = try gpa.alignedAlloc(T, .@"64", zc.Fft(T).bufferLen(n));
            defer gpa.free(buf);
            const f = zc.Fft(T).init(buf, n);
            const scratch = try gpa.alloc(T, 2 * n);
            defer gpa.free(scratch);
            var dr = xr;
            var di = xi;
            f.forward(&dr, &di, scratch);
            S.forward(&re, &im);
            var peak: f64 = 1;
            for (dr, di) |a, b| peak = @max(peak, @abs(@as(f64, a)), @abs(@as(f64, b)));
            const tol = peak * @as(f64, if (T == f32) 4e-6 else 1e-14);
            for (re, im, dr, di) |a, b, c, d| {
                try std.testing.expect(@abs(@as(f64, a) - c) <= tol and @abs(@as(f64, b) - d) <= tol);
            }
            S.inverse(&re, &im);
            const nf: f64 = @floatFromInt(n);
            for (re, im, xr, xi) |a, b, c, d| {
                try std.testing.expect(@abs(@as(f64, a) / nf - c) <= tol and @abs(@as(f64, b) / nf - d) <= tol);
            }
        }
    }
}
