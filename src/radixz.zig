const std = @import("std");

pub fn Fft(comptime T: type) type {
    return struct {
        const Self = @This();
        const lanes: usize = std.simd.suggestVectorLength(T) orelse 4;
        const V = @Vector(lanes, T);
        const Mask = @Vector(lanes, i32);
        const vecs = 8;
        pub const span = vecs * lanes;
        const span_log2 = std.math.log2_int(usize, span);
        const lanes_log2 = std.math.log2_int(usize, lanes);
        const group_max = 2;

        const Work = struct { re: []T, im: []T };

        n: usize,
        twr: []const T,
        twi: []const T,
        block: usize = 1 << 12,
        fused_min: usize = span * line,

        pub fn bufferLen(n: usize) usize {
            return 2 * n;
        }

        pub fn init(buf: []align(64) T, n: usize) Self {
            std.debug.assert(std.math.isPowerOfTwo(n) and n >= 2 and buf.len >= bufferLen(n));
            const twr = buf[0..n];
            const twi = buf[n .. 2 * n];
            twr[0] = 1;
            twi[0] = 0;
            var h: usize = 1;
            while (h < n) : (h *= 2) {
                for (0..h) |j| {
                    const a = -std.math.pi * @as(f64, @floatFromInt(j)) / @as(f64, @floatFromInt(h));
                    twr[h + j] = @floatCast(@cos(a));
                    twi[h + j] = @floatCast(@sin(a));
                }
            }
            return .{ .n = n, .twr = twr, .twi = twi };
        }

        pub fn forward(self: *const Self, re: []T, im: []T, scratch: []T) void {
            std.debug.assert(re.len == self.n and im.len == self.n and scratch.len >= 2 * self.n);
            if (codelet(false, self.n, re, im)) return;
            if (self.n < span) {
                self.scalar(false, re, im);
                return self.bitReverse(re, im);
            }
            const w: Work = .{ .re = scratch[0..self.n], .im = scratch[self.n..][0..self.n] };
            if (self.n < self.fused_min) {
                self.forwardScrambled(re, im);
                return self.slotPermute(false, re, im, w);
            }
            self.bigForward(w, re, im);
            self.permutedOut(false, w, re, im);
        }

        pub fn inverse(self: *const Self, re: []T, im: []T, scratch: []T) void {
            std.debug.assert(re.len == self.n and im.len == self.n and scratch.len >= 2 * self.n);
            if (codelet(true, self.n, re, im)) return;
            if (self.n < span) {
                self.bitReverse(re, im);
                return self.scalar(true, re, im);
            }
            const w: Work = .{ .re = scratch[0..self.n], .im = scratch[self.n..][0..self.n] };
            if (self.n < self.fused_min) {
                self.slotPermute(true, re, im, w);
                return self.inverseScrambled(re, im);
            }
            self.permutedIn(false, re, im, w);
            self.bigInverse(w, true, re, im, 0);
        }

        pub const Complex = std.math.Complex(T);

        comptime {
            std.debug.assert(@sizeOf(Complex) == 2 * @sizeOf(T) and @offsetOf(Complex, "re") == 0 and @offsetOf(Complex, "im") == @sizeOf(T));
        }

        pub fn forwardComplex(self: *const Self, z: []Complex, scratch: []T) void {
            self.complexRun(false, z, scratch);
        }

        pub fn inverseComplex(self: *const Self, z: []Complex, scratch: []T) void {
            self.complexRun(true, z, scratch);
        }

        fn complexRun(self: *const Self, comptime inv: bool, z: []Complex, scratch: []T) void {
            const n = self.n;
            std.debug.assert(z.len == n and scratch.len >= 4 * n);
            const x: []T = std.mem.bytesAsSlice(T, std.mem.sliceAsBytes(z));
            const w: Work = .{ .re = scratch[0..n], .im = scratch[n .. 2 * n] };
            if (n >= self.fused_min and self.bigStages() > 0) {
                if (inv) {
                    self.permutedIn(true, x, x, w);
                    self.bigInverseInterleaved(w, x);
                } else {
                    self.bigForwardInterleaved(w, x);
                    self.permutedOut(true, w, x, x);
                }
                return;
            }
            deinterleave(x, w.re, w.im);
            if (inv) self.inverse(w.re, w.im, scratch[2 * n ..]) else self.forward(w.re, w.im, scratch[2 * n ..]);
            interleave(w.re, w.im, x);
        }

        fn deinterleave(x: []const T, re: []T, im: []T) void {
            var k: usize = 0;
            while (k + lanes <= re.len) : (k += lanes) {
                const p = ldSplit(x, k);
                re[k..][0..lanes].* = p[0];
                im[k..][0..lanes].* = p[1];
            }
            while (k < re.len) : (k += 1) {
                re[k] = x[2 * k];
                im[k] = x[2 * k + 1];
            }
        }

        fn interleave(re: []const T, im: []const T, x: []T) void {
            var k: usize = 0;
            while (k + lanes <= re.len) : (k += lanes) stMerge(x, k, re[k..][0..lanes].*, im[k..][0..lanes].*);
            while (k < re.len) : (k += 1) {
                x[2 * k] = re[k];
                x[2 * k + 1] = im[k];
            }
        }

        const rev_span: [span]usize = blk: {
            var t: [span]usize = undefined;
            for (&t, 0..) |*v, i| v.* = @bitReverse(@as(std.math.IntFittingRange(0, span - 1), @intCast(i)));
            break :blk t;
        };

        fn rev(x: usize, bits: usize) usize {
            if (bits == 0) return 0;
            return @bitReverse(x) >> @intCast(@as(usize, @bitSizeOf(usize)) - bits);
        }

        const line = lanes;

        fn permutedOut(self: *const Self, comptime il: bool, w: Work, re: []T, im: []T) void {
            const cols = self.n / span;
            const cb = std.math.log2_int(usize, cols);
            var tr: [line][vecs]V = undefined;
            var ti: [line][vecs]V = undefined;
            var c: usize = 0;
            while (c < cols) : (c += line) {
                for (0..line) |j| {
                    const b = rev(c + j, cb) * span;
                    inline for (0..vecs) |q| {
                        tr[j][q] = ld(w.re, b + q * lanes);
                        ti[j][q] = ld(w.im, b + q * lanes);
                    }
                    bigStages2(false, &tr[j], &ti[j]);
                }
                inline for (0..vecs) |q| {
                    var vr: [lanes]V = undefined;
                    var vi: [lanes]V = undefined;
                    inline for (0..lanes) |j| {
                        vr[j] = tr[j][q];
                        vi[j] = ti[j][q];
                    }
                    transpose(&vr);
                    transpose(&vi);
                    smallStages(false, lanes, &vr, &vi);
                    inline for (0..lanes) |t| {
                        if (il) {
                            stMerge(re, rev_span[q * lanes + t] * cols + c, vr[t], vi[t]);
                        } else {
                            st(re, rev_span[q * lanes + t] * cols + c, vr[t]);
                            st(im, rev_span[q * lanes + t] * cols + c, vi[t]);
                        }
                    }
                }
            }
        }

        fn permutedIn(self: *const Self, comptime il: bool, re: []const T, im: []const T, w: Work) void {
            const cols = self.n / span;
            const cb = std.math.log2_int(usize, cols);
            var tr: [line][vecs]V = undefined;
            var ti: [line][vecs]V = undefined;
            var c: usize = 0;
            while (c < cols) : (c += line) {
                inline for (0..vecs) |q| {
                    var vr: [lanes]V = undefined;
                    var vi: [lanes]V = undefined;
                    inline for (0..lanes) |t| {
                        if (il) {
                            const p = ldSplit(re, rev_span[q * lanes + t] * cols + c);
                            vr[t] = p[0];
                            vi[t] = p[1];
                        } else {
                            vr[t] = ld(re, rev_span[q * lanes + t] * cols + c);
                            vi[t] = ld(im, rev_span[q * lanes + t] * cols + c);
                        }
                    }
                    smallStages(true, lanes, &vr, &vi);
                    transpose(&vr);
                    transpose(&vi);
                    inline for (0..lanes) |j| {
                        tr[j][q] = vr[j];
                        ti[j][q] = vi[j];
                    }
                }
                for (0..line) |j| {
                    const b = rev(c + j, cb) * span;
                    bigStages2(true, &tr[j], &ti[j]);
                    inline for (0..vecs) |q| {
                        st(w.re, b + q * lanes, tr[j][q]);
                        st(w.im, b + q * lanes, ti[j][q]);
                    }
                }
            }
        }

        pub fn forwardScrambled(self: *const Self, re: []T, im: []T) void {
            std.debug.assert(re.len == self.n and im.len == self.n);
            if (self.n < span) return self.scalar(false, re, im);
            const w: Work = .{ .re = re, .im = im };
            self.bigForward(w, re, im);
            smallPass(false, w);
        }

        pub fn inverseScrambled(self: *const Self, re: []T, im: []T) void {
            std.debug.assert(re.len == self.n and im.len == self.n);
            if (self.n < span) return self.scalar(true, re, im);
            const w: Work = .{ .re = re, .im = im };
            smallPass(true, w);
            self.bigInverse(w, false, &.{}, &.{}, 0);
        }

        fn Codelet(comptime L: usize) type {
            return struct {
                const W = @Vector(L, T);
                const bits = std.math.log2_int(usize, L);

                fn bf(comptime inv: bool, ar: *W, ai: *W, br: *W, bi: *W, wr: W, wi: W) void {
                    if (inv) {
                        const tr = @mulAdd(W, br.*, wr, bi.* * wi);
                        const ti = @mulAdd(W, bi.*, wr, -(br.* * wi));
                        br.* = ar.* - tr;
                        bi.* = ai.* - ti;
                        ar.* += tr;
                        ai.* += ti;
                    } else {
                        const dr = ar.* - br.*;
                        const di = ai.* - bi.*;
                        ar.* += br.*;
                        ai.* += bi.*;
                        br.* = @mulAdd(W, dr, wr, -(di * wi));
                        bi.* = @mulAdd(W, dr, wi, di * wr);
                    }
                }

                fn ctw(comptime h: usize, comptime j0: usize, comptime step: usize) [2]W {
                    var wr: [L]T = undefined;
                    var wi: [L]T = undefined;
                    for (0..L) |k| {
                        const ang = -std.math.pi * @as(f64, @floatFromInt((j0 + k * step) % h)) / @as(f64, @floatFromInt(h));
                        wr[k] = @floatCast(@cos(ang));
                        wi[k] = @floatCast(@sin(ang));
                    }
                    return .{ wr, wi };
                }

                fn flip(v: *[L]W) void {
                    inline for (0..bits) |t| {
                        const bs = 1 << t;
                        const ma, const mb = comptime blk: {
                            var x: [L]i32 = undefined;
                            var y: [L]i32 = undefined;
                            for (0..L) |p| {
                                x[p] = if (p & bs == 0) @intCast(p) else ~@as(i32, @intCast(p - bs));
                                y[p] = if (p & bs == 0) @intCast(p + bs) else ~@as(i32, @intCast(p));
                            }
                            break :blk .{ @as(@Vector(L, i32), x), @as(@Vector(L, i32), y) };
                        };
                        inline for (0..L) |i| {
                            if (i & bs == 0) {
                                const x = v[i];
                                const y = v[i | bs];
                                v[i] = @shuffle(T, x, y, ma);
                                v[i | bs] = @shuffle(T, x, y, mb);
                            }
                        }
                    }
                }

                fn r(comptime x: usize) usize {
                    return if (bits == 0) 0 else @bitReverse(@as(std.meta.Int(.unsigned, bits), @intCast(x)));
                }

                fn vertical(comptime inv: bool, v: *[L]W, m: *[L]W) void {
                    inline for (0..bits) |k| {
                        const h = if (inv) @as(usize, 1) << @intCast(k) else L >> @intCast(k + 1);
                        inline for (0..L) |q| {
                            if ((q / h) % 2 == 0) {
                                const w = comptime ctw(h * L, (q % h) * L, 1);
                                bf(inv, &v[q], &m[q], &v[q + h], &m[q + h], w[0], w[1]);
                            }
                        }
                    }
                }

                fn small(comptime inv: bool, v: *[L]W, m: *[L]W) void {
                    inline for (0..bits) |k| {
                        const h = if (inv) @as(usize, 1) << @intCast(k) else L >> @intCast(k + 1);
                        inline for (0..L) |t| {
                            if ((t / h) % 2 == 0) {
                                const w = comptime ctw(h, t % h, 0);
                                bf(inv, &v[t], &m[t], &v[t + h], &m[t + h], w[0], w[1]);
                            }
                        }
                    }
                }

                fn square(comptime inv: bool, v: *[L]W, m: *[L]W) void {
                    if (inv) {
                        var a: [L]W = undefined;
                        var b: [L]W = undefined;
                        inline for (0..L) |t| {
                            a[t] = v[r(t)];
                            b[t] = m[r(t)];
                        }
                        small(true, &a, &b);
                        flip(&a);
                        flip(&b);
                        inline for (0..L) |l| {
                            v[r(l)] = a[l];
                            m[r(l)] = b[l];
                        }
                        vertical(true, v, m);
                    } else {
                        vertical(false, v, m);
                        var a: [L]W = undefined;
                        var b: [L]W = undefined;
                        inline for (0..L) |l| {
                            a[l] = v[r(l)];
                            b[l] = m[r(l)];
                        }
                        flip(&a);
                        flip(&b);
                        small(false, &a, &b);
                        inline for (0..L) |t| {
                            v[r(t)] = a[t];
                            m[r(t)] = b[t];
                        }
                    }
                }

                fn run(comptime inv: bool, comptime twice: bool, re: []T, im: []T) void {
                    const halves = if (twice) 2 else 1;
                    var v: [halves][L]W = undefined;
                    var m: [halves][L]W = undefined;
                    inline for (0..halves) |hf| inline for (0..L) |q| {
                        v[hf][q] = re[(hf * L + q) * L ..][0..L].*;
                        m[hf][q] = im[(hf * L + q) * L ..][0..L].*;
                    };
                    if (twice) {
                        if (inv) {
                            var a: [2 * L * L]T = undefined;
                            var b: [2 * L * L]T = undefined;
                            inline for (0..2) |hf| inline for (0..L) |q| {
                                a[(hf * L + q) * L ..][0..L].* = v[hf][q];
                                b[(hf * L + q) * L ..][0..L].* = m[hf][q];
                            };
                            inline for (0..2) |hf| inline for (0..L) |q| {
                                var x: [L]T = undefined;
                                var y: [L]T = undefined;
                                inline for (0..L) |l| {
                                    x[l] = a[2 * (q * L + l) + hf];
                                    y[l] = b[2 * (q * L + l) + hf];
                                }
                                v[hf][q] = x;
                                m[hf][q] = y;
                            };
                            inline for (0..2) |hf| square(true, &v[hf], &m[hf]);
                            inline for (0..L) |q| {
                                const w = comptime ctw(L * L, q * L, 1);
                                bf(true, &v[0][q], &m[0][q], &v[1][q], &m[1][q], w[0], w[1]);
                            }
                        } else {
                            inline for (0..L) |q| {
                                const w = comptime ctw(L * L, q * L, 1);
                                bf(false, &v[0][q], &m[0][q], &v[1][q], &m[1][q], w[0], w[1]);
                            }
                            inline for (0..2) |hf| square(false, &v[hf], &m[hf]);
                            var a: [2 * L * L]T = undefined;
                            var b: [2 * L * L]T = undefined;
                            inline for (0..2) |hf| inline for (0..L) |q| {
                                const x: [L]T = v[hf][q];
                                const y: [L]T = m[hf][q];
                                inline for (0..L) |l| {
                                    a[2 * (q * L + l) + hf] = x[l];
                                    b[2 * (q * L + l) + hf] = y[l];
                                }
                            };
                            inline for (0..2) |hf| inline for (0..L) |q| {
                                v[hf][q] = a[(hf * L + q) * L ..][0..L].*;
                                m[hf][q] = b[(hf * L + q) * L ..][0..L].*;
                            };
                        }
                    } else square(inv, &v[0], &m[0]);
                    inline for (0..halves) |hf| inline for (0..L) |q| {
                        re[(hf * L + q) * L ..][0..L].* = v[hf][q];
                        im[(hf * L + q) * L ..][0..L].* = m[hf][q];
                    };
                }
            };
        }

        fn codelet(comptime inv: bool, n: usize, re: []T, im: []T) bool {
            switch (n) {
                inline 4, 8, 16, 32, 64, 128, 256 => |c| {
                    const lg = comptime std.math.log2_int(usize, c);
                    const L: usize = 1 << (lg / 2);
                    if (L > (if (lg % 2 == 1) lanes else 2 * lanes)) return false;
                    Codelet(L).run(inv, lg % 2 == 1, re, im);
                    return true;
                },
                else => return false,
            }
        }

        fn bitReverse(self: *const Self, re: []T, im: []T) void {
            std.debug.assert(re.len == self.n and im.len == self.n);
            const shift: std.math.Log2Int(usize) = @intCast(@as(usize, @bitSizeOf(usize)) - std.math.log2_int(usize, self.n));
            for (0..self.n) |i| {
                const j = @bitReverse(i) >> shift;
                if (i < j) {
                    std.mem.swap(T, &re[i], &re[j]);
                    std.mem.swap(T, &im[i], &im[j]);
                }
            }
        }

        fn scalar(self: *const Self, comptime inv: bool, re: []T, im: []T) void {
            var h: usize = if (inv) 1 else self.n / 2;
            while (h >= 1 and h < self.n) : (h = if (inv) h * 2 else h / 2) {
                var i: usize = 0;
                while (i < self.n) : (i += 2 * h) {
                    for (i..i + h, self.twr[h..][0..h], self.twi[h..][0..h]) |a, wr, wi| {
                        const b = a + h;
                        if (inv) {
                            const tr = re[b] * wr + im[b] * wi;
                            const ti = im[b] * wr - re[b] * wi;
                            re[b] = re[a] - tr;
                            im[b] = im[a] - ti;
                            re[a] += tr;
                            im[a] += ti;
                        } else {
                            const dr = re[a] - re[b];
                            const di = im[a] - im[b];
                            re[a] += re[b];
                            im[a] += im[b];
                            re[b] = dr * wr - di * wi;
                            im[b] = dr * wi + di * wr;
                        }
                    }
                }
            }
        }

        fn ld(s: []const T, i: usize) V {
            return s[i..][0..lanes].*;
        }

        const even_mask: Mask = blk: {
            var a: [lanes]i32 = undefined;
            for (0..lanes) |i| a[i] = if (2 * i < lanes) @intCast(2 * i) else ~@as(i32, @intCast(2 * i - lanes));
            break :blk a;
        };

        const odd_mask: Mask = blk: {
            var a: [lanes]i32 = undefined;
            for (0..lanes) |i| a[i] = if (2 * i + 1 < lanes) @intCast(2 * i + 1) else ~@as(i32, @intCast(2 * i + 1 - lanes));
            break :blk a;
        };

        fn ldSplit(x: []const T, i: usize) [2]V {
            const a: V = x[2 * i ..][0..lanes].*;
            const b: V = x[2 * i + lanes ..][0..lanes].*;
            return .{ @shuffle(T, a, b, even_mask), @shuffle(T, a, b, odd_mask) };
        }

        const lo_mask: Mask = blk: {
            var a: [lanes]i32 = undefined;
            for (0..lanes) |i| a[i] = if (i % 2 == 0) @intCast(i / 2) else ~@as(i32, @intCast(i / 2));
            break :blk a;
        };

        const hi_mask: Mask = blk: {
            var a: [lanes]i32 = undefined;
            for (0..lanes) |i| a[i] = if (i % 2 == 0) @intCast(lanes / 2 + i / 2) else ~@as(i32, @intCast(lanes / 2 + i / 2));
            break :blk a;
        };

        fn stMerge(x: []T, i: usize, a: V, b: V) void {
            x[2 * i ..][0..lanes].* = @shuffle(T, a, b, lo_mask);
            x[2 * i + lanes ..][0..lanes].* = @shuffle(T, a, b, hi_mask);
        }

        fn st(s: []T, i: usize, v: V) void {
            s[i..][0..lanes].* = v;
        }

        fn group(rem: usize) usize {
            return @min(rem, group_max);
        }

        fn bigStages(self: *const Self) usize {
            return std.math.log2_int(usize, self.n) - span_log2;
        }

        fn bigForward(self: *const Self, w: Work, sr: []const T, si: []const T) void {
            const rem = self.bigStages();
            if (rem == 0) {
                if (sr.ptr != w.re.ptr) {
                    @memcpy(w.re, sr);
                    @memcpy(w.im, si);
                }
                return;
            }
            self.fwdBlock(w, false, sr, si, 0, self.n, rem);
        }

        fn bigForwardInterleaved(self: *const Self, w: Work, x: []const T) void {
            self.fwdBlock(w, true, x, x, 0, self.n, self.bigStages());
        }

        fn fwdBlock(self: *const Self, w: Work, comptime il: bool, sr: []const T, si: []const T, lo: usize, size: usize, rem: usize) void {
            var top = size / 2;
            var left = rem;
            if (left == 0) return;
            const g0 = group(left);
            const q0 = top >> @intCast(g0 - 1);
            self.pass(w, false, il, g0, q0, sr, si, lo, lo + size, false, &.{}, &.{}, 0);
            if (size > self.block and left > group_max) {
                const sub = size >> @intCast(g0);
                for (0..@as(usize, 1) << @intCast(g0)) |k| self.fwdBlock(w, false, w.re, w.im, lo + k * sub, sub, left - g0);
                return;
            }
            top = q0 / 2;
            left -= g0;
            while (left > 0) {
                const g = group(left);
                const q = top >> @intCast(g - 1);
                self.pass(w, false, false, g, q, w.re, w.im, lo, lo + size, false, &.{}, &.{}, 0);
                top = q / 2;
                left -= g;
            }
        }

        fn bigInverse(self: *const Self, w: Work, comptime to_y: bool, yr: []T, yi: []T, skip: usize) void {
            self.invBlock(w, to_y, false, yr, yi, skip, 0, self.n, self.bigStages());
        }

        fn bigInverseInterleaved(self: *const Self, w: Work, x: []T) void {
            self.invBlock(w, true, true, x, x, 0, 0, self.n, self.bigStages());
        }

        fn invBlock(self: *const Self, w: Work, comptime to_y: bool, comptime il: bool, yr: []T, yi: []T, skip: usize, lo: usize, size: usize, rem: usize) void {
            if (size > self.block and rem > group_max) {
                const g = group(rem);
                const sub = size >> @intCast(g);
                for (0..@as(usize, 1) << @intCast(g)) |k| self.invBlock(w, false, false, &.{}, &.{}, 0, lo + k * sub, sub, rem - g);
                self.pass(w, true, il, g, sub, w.re, w.im, lo, lo + size, to_y, yr, yi, skip);
                return;
            }
            var left = rem;
            var q: usize = span;
            while (left > 0) {
                const g: usize = if (left % group_max != 0) left % group_max else group_max;
                left -= g;
                if (to_y and left == 0)
                    self.pass(w, true, il, g, q, w.re, w.im, lo, lo + size, true, yr, yi, skip)
                else
                    self.pass(w, true, false, g, q, w.re, w.im, lo, lo + size, false, &.{}, &.{}, 0);
                q <<= @intCast(g);
            }
        }

        fn pass(self: *const Self, w: Work, comptime inv: bool, comptime il: bool, g: usize, q: usize, sr: []const T, si: []const T, lo: usize, hi: usize, comptime to_y: bool, yr: []T, yi: []T, skip: usize) void {
            switch (g) {
                inline 1...group_max => |c| self.passN(w, inv, il, c, q, sr, si, lo, hi, to_y, yr, yi, skip),
                else => unreachable,
            }
        }

        fn put(d: []T, skip: usize, idx: usize, v: V) void {
            if (idx >= skip) return st(d, idx, v);
            if (idx + lanes <= skip) return;
            const a: [lanes]T = v;
            for (skip - idx..lanes) |e| d[idx + e] = a[e];
        }

        fn tw(self: *const Self, h: usize, j: usize) [2]V {
            return .{ ld(self.twr, h + j), ld(self.twi, h + j) };
        }

        fn passN(self: *const Self, w: Work, comptime inv: bool, comptime il: bool, comptime s: usize, q: usize, sr: []const T, si: []const T, lo: usize, hi: usize, comptime to_y: bool, yr: []T, yi: []T, skip: usize) void {
            const g = 1 << s;
            var i = lo;
            while (i < hi) : (i += g * q) {
                var j: usize = 0;
                while (j < q) : (j += lanes) {
                    var r: [g]V = undefined;
                    var m: [g]V = undefined;
                    inline for (0..g) |k| {
                        if (il and !inv) {
                            const p = ldSplit(sr, i + j + k * q);
                            r[k] = p[0];
                            m[k] = p[1];
                        } else {
                            r[k] = ld(sr, i + j + k * q);
                            m[k] = ld(si, i + j + k * q);
                        }
                    }
                    if (s == 2) {
                        const w2 = self.tw(2 * q, j);
                        const p = w2[0] * w2[1];
                        const w1: [2]V = .{ @mulAdd(V, w2[0], w2[0], -(w2[1] * w2[1])), p + p };
                        const w3r = w2[1];
                        const w3i = -w2[0];
                        if (inv) {
                            butterfly(true, &r[0], &m[0], &r[1], &m[1], w1[0], w1[1]);
                            butterfly(true, &r[2], &m[2], &r[3], &m[3], w1[0], w1[1]);
                            butterfly(true, &r[0], &m[0], &r[2], &m[2], w2[0], w2[1]);
                            butterfly(true, &r[1], &m[1], &r[3], &m[3], w3r, w3i);
                        } else {
                            butterfly(false, &r[0], &m[0], &r[2], &m[2], w2[0], w2[1]);
                            butterfly(false, &r[1], &m[1], &r[3], &m[3], w3r, w3i);
                            butterfly(false, &r[0], &m[0], &r[1], &m[1], w1[0], w1[1]);
                            butterfly(false, &r[2], &m[2], &r[3], &m[3], w1[0], w1[1]);
                        }
                    } else inline for (0..s) |u| {
                        const t = if (inv) u else s - 1 - u;
                        const bit = 1 << t;
                        inline for (0..g) |k| {
                            if (k & bit == 0) {
                                const c = self.tw(q << t, j + (k & (bit - 1)) * q);
                                butterfly(inv, &r[k], &m[k], &r[k | bit], &m[k | bit], c[0], c[1]);
                            }
                        }
                    }
                    inline for (0..g) |k| {
                        if (to_y and il) {
                            stMerge(yr, i + j + k * q, r[k], m[k]);
                        } else if (to_y) {
                            put(yr, skip, i + j + k * q, r[k]);
                            put(yi, skip, i + j + k * q, m[k]);
                        } else {
                            st(w.re, i + j + k * q, r[k]);
                            st(w.im, i + j + k * q, m[k]);
                        }
                    }
                }
            }
        }

        fn inPlace(self: *const Self, comptime inv: bool, re: []T, im: []T) void {
            if (codelet(inv, self.n, re, im)) return;
            if (self.n < span) {
                if (inv) self.bitReverse(re, im);
                self.scalar(inv, re, im);
                if (!inv) self.bitReverse(re, im);
                return;
            }
            const w: Work = .{ .re = re, .im = im };
            if (inv) {
                self.bitReverse(re, im);
                exactPass(true, w);
                self.bigInverse(w, false, &.{}, &.{}, 0);
            } else {
                self.bigForward(w, re, im);
                exactPass(false, w);
                self.bitReverse(re, im);
            }
        }

        fn exactPass(comptime inv: bool, w: Work) void {
            var i: usize = 0;
            while (i < w.re.len) : (i += span) {
                var r: [vecs]V = undefined;
                var m: [vecs]V = undefined;
                inline for (0..vecs) |q| {
                    r[q] = ld(w.re, i + q * lanes);
                    m[q] = ld(w.im, i + q * lanes);
                }
                if (inv) transposeGroups(&r, &m);
                chunk(inv, &r, &m);
                if (!inv) transposeGroups(&r, &m);
                inline for (0..vecs) |q| {
                    st(w.re, i + q * lanes, r[q]);
                    st(w.im, i + q * lanes, m[q]);
                }
            }
        }

        fn smallPass(comptime inv: bool, w: Work) void {
            var i: usize = 0;
            while (i < w.re.len) : (i += span) {
                var r: [vecs]V = undefined;
                var m: [vecs]V = undefined;
                inline for (0..vecs) |q| {
                    r[q] = ld(w.re, i + q * lanes);
                    m[q] = ld(w.im, i + q * lanes);
                }
                chunk(inv, &r, &m);
                inline for (0..vecs) |q| {
                    st(w.re, i + q * lanes, r[q]);
                    st(w.im, i + q * lanes, m[q]);
                }
            }
        }

        fn chunk(comptime inv: bool, r: *[vecs]V, m: *[vecs]V) void {
            if (inv) {
                smallStages(true, vecs, r, m);
                transposeGroups(r, m);
                bigStages2(true, r, m);
            } else {
                bigStages2(false, r, m);
                transposeGroups(r, m);
                smallStages(false, vecs, r, m);
            }
        }

        fn bigStages2(comptime inv: bool, r: *[vecs]V, m: *[vecs]V) void {
            inline for (0..span_log2 - lanes_log2) |k| {
                const h = if (inv) lanes << @intCast(k) else span >> @intCast(k + 1);
                inline for (0..vecs) |q| {
                    if ((q * lanes / h) % 2 == 0) {
                        const w = comptime twiddles(h, (q * lanes) % h);
                        butterfly(inv, &r[q], &m[q], &r[q + h / lanes], &m[q + h / lanes], w[0], w[1]);
                    }
                }
            }
        }

        fn transposeGroups(r: *[vecs]V, m: *[vecs]V) void {
            inline for (0..vecs / lanes) |g| {
                transpose(r[g * lanes ..][0..lanes]);
                transpose(m[g * lanes ..][0..lanes]);
            }
        }

        fn smallStages(comptime inv: bool, comptime count: usize, r: *[count]V, m: *[count]V) void {
            inline for (0..lanes_log2) |k| {
                const h = if (inv) @as(usize, 1) << @intCast(k) else lanes >> @intCast(k + 1);
                inline for (0..lanes) |e| {
                    if ((e / h) % 2 == 0) {
                        const ang = -std.math.pi * @as(f64, @floatFromInt(e % h)) / @as(f64, @floatFromInt(h));
                        const wr: V = @splat(@as(T, @floatCast(@cos(ang))));
                        const wi: V = @splat(@as(T, @floatCast(@sin(ang))));
                        inline for (0..count / lanes) |g| {
                            const i = g * lanes + e;
                            butterfly(inv, &r[i], &m[i], &r[i + h], &m[i + h], wr, wi);
                        }
                    }
                }
            }
        }

        fn transpose(v: *[lanes]V) void {
            inline for (0..lanes_log2) |t| {
                const bs = 1 << t;
                const ma, const mb = comptime blk: {
                    var x: [lanes]i32 = undefined;
                    var y: [lanes]i32 = undefined;
                    for (0..lanes) |p| {
                        x[p] = if (p & bs == 0) @intCast(p) else ~@as(i32, @intCast(p - bs));
                        y[p] = if (p & bs == 0) @intCast(p + bs) else ~@as(i32, @intCast(p));
                    }
                    break :blk .{ @as(Mask, x), @as(Mask, y) };
                };
                inline for (0..lanes) |i| {
                    if (i & bs == 0) {
                        const x = v[i];
                        const y = v[i | bs];
                        v[i] = @shuffle(T, x, y, ma);
                        v[i | bs] = @shuffle(T, x, y, mb);
                    }
                }
            }
        }

        fn slot(v: usize, l: usize) usize {
            return ((v / lanes) * lanes + l) * lanes + v % lanes;
        }

        fn slotPermute(self: *const Self, comptime inv: bool, re: []T, im: []T, w: Work) void {
            const bits = std.math.log2_int(usize, self.n);
            for (0..self.n) |p| {
                const within = p % span;
                const d = rev(p - within + slot(within / lanes, within % lanes), bits);
                if (inv) {
                    w.re[p] = re[d];
                    w.im[p] = im[d];
                } else {
                    w.re[d] = re[p];
                    w.im[d] = im[p];
                }
            }
            @memcpy(re, w.re);
            @memcpy(im, w.im);
        }

        fn butterfly(comptime inv: bool, ar: *V, ai: *V, br: *V, bi: *V, wr: V, wi: V) void {
            if (inv) {
                const tr = @mulAdd(V, br.*, wr, bi.* * wi);
                const ti = @mulAdd(V, bi.*, wr, -(br.* * wi));
                br.* = ar.* - tr;
                bi.* = ai.* - ti;
                ar.* += tr;
                ai.* += ti;
            } else {
                const dr = ar.* - br.*;
                const di = ai.* - bi.*;
                ar.* += br.*;
                ai.* += bi.*;
                br.* = @mulAdd(V, dr, wr, -(di * wi));
                bi.* = @mulAdd(V, dr, wi, di * wr);
            }
        }

        fn twiddles(comptime h: usize, comptime j0: usize) [2]V {
            var wr: [lanes]T = undefined;
            var wi: [lanes]T = undefined;
            for (0..lanes) |k| {
                const ang = -std.math.pi * @as(f64, @floatFromInt(j0 + k)) / @as(f64, @floatFromInt(h));
                wr[k] = @floatCast(@cos(ang));
                wi[k] = @floatCast(@sin(ang));
            }
            return .{ wr, wi };
        }
    };
}

pub fn Conv(comptime T: type) type {
    return struct {
        const Self = @This();
        const F = Fft(T);
        const V = F.V;
        const lanes = F.lanes;
        pub const span = F.span;

        fft: F,
        n: usize,
        taps: usize,
        hr: []T,
        hi: []T,
        works: []T,
        workers: usize,

        pub fn bufferLen(n: usize, workers: usize) usize {
            return F.bufferLen(n) + (2 + 2 * workers) * n;
        }

        pub fn init(buf: []align(64) T, taps: []const T, n: usize, workers: usize) Self {
            std.debug.assert(n >= span and n > taps.len and taps.len >= 1 and workers >= 1 and buf.len >= bufferLen(n, workers));
            const f0 = F.bufferLen(n);
            var self: Self = .{
                .fft = F.init(buf[0..f0], n),
                .n = n,
                .taps = taps.len,
                .hr = buf[f0..][0..n],
                .hi = buf[f0 + n ..][0..n],
                .works = buf[f0 + 2 * n ..][0 .. 2 * workers * n],
                .workers = workers,
            };
            const w = self.work(0);
            @memset(w.re, 0);
            @memset(w.im, 0);
            @memcpy(w.re[0..taps.len], taps);
            self.fft.forwardScrambled(w.re, w.im);
            const s: T = 1 / @as(T, @floatFromInt(n));
            for (self.hr, self.hi, w.re, w.im) |*dr, *di, r, i| {
                dr.* = r * s;
                di.* = i * s;
            }
            return self;
        }

        fn work(self: *const Self, k: usize) F.Work {
            return .{ .re = self.works[2 * k * self.n ..][0..self.n], .im = self.works[(2 * k + 1) * self.n ..][0..self.n] };
        }

        fn pairs(self: *const Self, len: usize) usize {
            const l = self.n - self.taps + 1;
            return (len + 2 * l - 1) / (2 * l);
        }

        pub fn filter(self: *const Self, x: []const T, y: []T) void {
            std.debug.assert(y.len == x.len);
            self.range(self.work(0), x, y, 0, self.pairs(x.len));
        }

        pub fn filterParallel(self: *const Self, io: std.Io, x: []const T, y: []T) (std.Io.ConcurrentError || std.Io.Cancelable)!void {
            std.debug.assert(y.len == x.len);
            const p = self.pairs(x.len);
            const t = @min(self.workers, p);
            if (t <= 1) return self.filter(x, y);
            var next: std.atomic.Value(usize) = .init(0);
            const step = @max(1, p / (8 * t));
            var g: std.Io.Group = .init;
            errdefer g.cancel(io);
            for (1..t) |k| try g.concurrent(io, worker, .{ self, self.work(k), x, y, &next, p, step });
            self.worker(self.work(0), x, y, &next, p, step);
            try g.await(io);
        }

        fn worker(self: *const Self, w: F.Work, x: []const T, y: []T, next: *std.atomic.Value(usize), p: usize, step: usize) void {
            while (true) {
                const p0 = next.fetchAdd(step, .monotonic);
                if (p0 >= p) return;
                self.range(w, x, y, p0, @min(p0 + step, p));
            }
        }

        fn range(self: *const Self, w: F.Work, x: []const T, y: []T, p0: usize, p1: usize) void {
            const m1 = self.taps - 1;
            const l = self.n - m1;
            for (p0..p1) |p| {
                const b = 2 * p;
                const s0: isize = @as(isize, @intCast(b * l)) - @as(isize, @intCast(m1));
                const s1 = s0 + @as(isize, @intCast(l));
                if (s0 >= 0 and s1 + @as(isize, @intCast(self.n)) <= @as(isize, @intCast(x.len))) {
                    self.fft.bigForward(w, x[@intCast(s0)..][0..self.n], x[@intCast(s1)..][0..self.n]);
                } else {
                    self.load(w.re, x, s0);
                    self.load(w.im, x, s1);
                    self.fft.bigForward(w, w.re, w.im);
                }
                self.middle(w);
                const o0 = b * l;
                if (self.n >= 2048 and o0 >= m1 and o0 + 2 * l <= y.len) {
                    self.fft.bigInverse(w, true, y[o0 - m1 ..], y[o0 + l - m1 ..], m1);
                } else {
                    self.fft.bigInverse(w, false, &.{}, &.{}, 0);
                    store(y, w.re[m1..], o0, l);
                    store(y, w.im[m1..], o0 + l, l);
                }
            }
        }

        fn store(y: []T, src: []const T, o: usize, l: usize) void {
            if (o >= y.len) return;
            const cnt = @min(l, y.len - o);
            @memcpy(y[o..][0..cnt], src[0..cnt]);
        }

        fn load(self: *const Self, dst: []T, x: []const T, start: isize) void {
            const len: isize = @intCast(x.len);
            const lo = @max(start, 0);
            const hi = @min(start + @as(isize, @intCast(self.n)), len);
            if (hi <= lo) return @memset(dst, 0);
            const d0: usize = @intCast(lo - start);
            const cnt: usize = @intCast(hi - lo);
            @memset(dst[0..d0], 0);
            @memcpy(dst[d0..][0..cnt], x[@intCast(lo)..][0..cnt]);
            @memset(dst[d0 + cnt ..], 0);
        }

        fn middle(self: *const Self, w: F.Work) void {
            var i: usize = 0;
            while (i < self.n) : (i += span) {
                var r: [F.vecs]V = undefined;
                var m: [F.vecs]V = undefined;
                inline for (0..F.vecs) |q| {
                    r[q] = F.ld(w.re, i + q * lanes);
                    m[q] = F.ld(w.im, i + q * lanes);
                }
                F.chunk(false, &r, &m);
                inline for (0..F.vecs) |q| {
                    const hr = F.ld(self.hr, i + q * lanes);
                    const hi = F.ld(self.hi, i + q * lanes);
                    const zr = r[q];
                    r[q] = @mulAdd(V, zr, hr, -(m[q] * hi));
                    m[q] = @mulAdd(V, zr, hi, m[q] * hr);
                }
                F.chunk(true, &r, &m);
                inline for (0..F.vecs) |q| {
                    F.st(w.re, i + q * lanes, r[q]);
                    F.st(w.im, i + q * lanes, m[q]);
                }
            }
        }
    };
}

pub fn StaticFft(comptime T: type, comptime n: usize) type {
    return struct {
        const F = Fft(T);
        const table: [2 * n]T = blk: {
            @setEvalBranchQuota(64 * n + 1000);
            var t: [2 * n]T = undefined;
            t[0] = 1;
            t[n] = 0;
            var h: usize = 1;
            while (h < n) : (h *= 2) {
                for (0..h) |j| {
                    const a = -std.math.pi * @as(f64, @floatFromInt(j)) / @as(f64, @floatFromInt(h));
                    t[h + j] = @floatCast(@cos(a));
                    t[n + h + j] = @floatCast(@sin(a));
                }
            }
            break :blk t;
        };
        const plan: F = .{ .n = n, .twr = table[0..n], .twi = table[n..] };

        pub fn forward(re: *[n]T, im: *[n]T) void {
            plan.inPlace(false, re, im);
        }

        pub fn inverse(re: *[n]T, im: *[n]T) void {
            plan.inPlace(true, re, im);
        }
    };
}

pub fn Rfft(comptime T: type) type {
    return struct {
        const Self = @This();
        const F = Fft(T);
        const lanes = F.lanes;
        const V = F.V;
        const M = F.Mask;

        n: usize,
        half: F,
        wr: []const T,
        wi: []const T,

        pub fn bufferLen(n: usize) usize {
            return if (n < 4) 0 else F.bufferLen(n / 2) + n;
        }

        pub fn init(buf: []align(64) T, n: usize) Self {
            std.debug.assert(std.math.isPowerOfTwo(n) and n >= 2 and buf.len >= bufferLen(n));
            if (n < 4) return .{ .n = n, .half = undefined, .wr = &.{}, .wi = &.{} };
            const m = n / 2;
            const f0 = F.bufferLen(m);
            const wr = buf[f0..][0..m];
            const wi = buf[f0 + m ..][0..m];
            for (wr, wi, 0..) |*a, *b, k| {
                const ang = -2 * std.math.pi * @as(f64, @floatFromInt(k)) / @as(f64, @floatFromInt(n));
                a.* = @floatCast(@cos(ang));
                b.* = @floatCast(@sin(ang));
            }
            return .{ .n = n, .half = F.init(buf[0..f0], m), .wr = wr, .wi = wi };
        }

        pub fn forward(self: *const Self, x: []const T, re: []T, im: []T, scratch: []T) void {
            const m = self.n / 2;
            std.debug.assert(x.len == self.n and re.len == m + 1 and im.len == m + 1 and scratch.len >= self.n);
            if (self.n == 2) {
                re[0] = x[0] + x[1];
                re[1] = x[0] - x[1];
                im[0] = 0;
                im[1] = 0;
                return;
            }
            if (m >= self.half.fused_min and self.half.bigStages() > 0) {
                const w: F.Work = .{ .re = scratch[0..m], .im = scratch[m..][0..m] };
                self.half.bigForwardInterleaved(w, x);
                self.half.permutedOut(false, w, re[0..m], im[0..m]);
            } else {
                F.deinterleave(x, re[0..m], im[0..m]);
                self.half.forward(re[0..m], im[0..m], scratch);
            }
            const z0r = re[0];
            const z0i = im[0];
            re[0] = z0r + z0i;
            im[0] = 0;
            re[m] = z0r - z0i;
            im[m] = 0;
            self.untangle(false, re, im);
        }

        pub const Complex = F.Complex;

        pub fn forwardComplex(self: *const Self, x: []const T, out: []Complex, scratch: []T) void {
            const m = self.n / 2;
            std.debug.assert(out.len == m + 1 and scratch.len >= 2 * self.n + 2);
            const re = scratch[0 .. m + 1];
            const im = scratch[m + 1 ..][0 .. m + 1];
            self.forward(x, re, im, scratch[2 * m + 2 ..][0..self.n]);
            F.interleave(re, im, std.mem.bytesAsSlice(T, std.mem.sliceAsBytes(out)));
        }

        pub fn inverseComplex(self: *const Self, in: []const Complex, x: []T, scratch: []T) void {
            const m = self.n / 2;
            std.debug.assert(in.len == m + 1 and scratch.len >= 2 * self.n + 2);
            const re = scratch[0 .. m + 1];
            const im = scratch[m + 1 ..][0 .. m + 1];
            F.deinterleave(std.mem.bytesAsSlice(T, std.mem.sliceAsBytes(in)), re, im);
            self.inverse(re, im, x, scratch[2 * m + 2 ..][0..self.n]);
        }

        pub fn inverse(self: *const Self, re: []T, im: []T, x: []T, scratch: []T) void {
            const m = self.n / 2;
            std.debug.assert(x.len == self.n and re.len == m + 1 and im.len == m + 1 and scratch.len >= self.n);
            if (self.n == 2) {
                x[0] = re[0] + re[1];
                x[1] = re[0] - re[1];
                return;
            }
            const x0 = re[0];
            const xm = re[m];
            re[0] = x0 + xm;
            im[0] = x0 - xm;
            self.untangle(true, re, im);
            if (m >= self.half.fused_min and self.half.bigStages() > 0) {
                const w: F.Work = .{ .re = scratch[0..m], .im = scratch[m..][0..m] };
                self.half.permutedIn(false, re[0..m], im[0..m], w);
                self.half.bigInverseInterleaved(w, x);
            } else {
                self.half.inverse(re[0..m], im[0..m], scratch);
                F.interleave(re[0..m], im[0..m], x);
            }
        }

        fn untangle(self: *const Self, comptime inv: bool, re: []T, im: []T) void {
            const m = self.n / 2;
            const sgn: T = if (inv) -1 else 1;
            var k: usize = 1;
            while (k + lanes <= m / 2) : (k += lanes) {
                const j = m - k - lanes + 1;
                const ar: V = re[k..][0..lanes].*;
                const ai: V = im[k..][0..lanes].*;
                const br = rev(re[j..][0..lanes].*);
                const bi = rev(im[j..][0..lanes].*);
                const wr: V = self.wr[k..][0..lanes].*;
                const wi: V = self.wi[k..][0..lanes].*;
                const o = pair(V, inv, ar, ai, br, bi, wr, wi * @as(V, @splat(sgn)));
                re[k..][0..lanes].* = o[0];
                im[k..][0..lanes].* = o[1];
                re[j..][0..lanes].* = rev(o[2]);
                im[j..][0..lanes].* = rev(o[3]);
            }
            while (k <= m / 2) : (k += 1) {
                const o = pair(T, inv, re[k], im[k], re[m - k], im[m - k], self.wr[k], sgn * self.wi[k]);
                re[k] = o[0];
                im[k] = o[1];
                re[m - k] = o[2];
                im[m - k] = o[3];
            }
        }

        fn pair(comptime E: type, comptime inv: bool, ar: E, ai: E, br: E, bi: E, wr: E, wi: E) [4]E {
            const er = ar + br;
            const ei = ai - bi;
            const dr = ar - br;
            const di = ai + bi;
            const half: E = if (E == T) 0.5 else @splat(@as(T, 0.5));
            if (inv) {
                const tr = di * wr + dr * wi;
                const ti = di * wi - dr * wr;
                return .{ er - tr, ei - ti, er + tr, -(ei + ti) };
            }
            const tr = (di * wr + dr * wi) * half;
            const ti = (di * wi - dr * wr) * half;
            return .{ er * half + tr, ei * half + ti, er * half - tr, -(ei * half - ti) };
        }

        fn rev(v: V) V {
            const mask: M = comptime blk: {
                var a: [lanes]i32 = undefined;
                for (0..lanes) |i| a[i] = @intCast(lanes - 1 - i);
                break :blk a;
            };
            return @shuffle(T, v, undefined, mask);
        }
    };
}
