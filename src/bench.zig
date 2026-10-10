const std = @import("std");
const zc = @import("radixz");
const Conv = zc.Conv(f32);

const Plan = *anyopaque;
extern fn fftwf_malloc(n: usize) ?*anyopaque;
extern fn fftwf_free(p: ?*anyopaque) void;
extern fn fftwf_plan_dft_r2c_1d(n: c_int, in: [*]f32, out: [*]f32, flags: c_uint) ?Plan;
extern fn fftwf_plan_dft_c2r_1d(n: c_int, in: [*]f32, out: [*]f32, flags: c_uint) ?Plan;
extern fn fftwf_execute(p: Plan) void;
extern fn fftwf_execute_dft_r2c(p: Plan, in: [*]f32, out: [*]f32) void;
extern fn fftwf_execute_dft_c2r(p: Plan, in: [*]f32, out: [*]f32) void;
extern fn fftwf_destroy_plan(p: Plan) void;
extern fn fftwf_init_threads() c_int;
extern fn fftwf_plan_with_nthreads(n: c_int) void;
extern fn fftwf_cleanup_threads() void;

const total: usize = 1 << 22;
const runs = 7;
const thread_counts = [_]usize{ 1, 2, 4, 8, 12 };
const parallel_taps = [_]usize{ 128, 2048, 32768 };

fn fbuf(n: usize) []f32 {
    const p: [*]align(16) f32 = @ptrCast(@alignCast(fftwf_malloc(n * 4).?));
    return p[0..n];
}

const Bufs = struct {
    in: []f32,
    spec: []f32,
    out: []f32,

    fn init(n: usize) Bufs {
        return .{ .in = fbuf(n), .spec = fbuf(n + 2), .out = fbuf(n) };
    }

    fn deinit(b: Bufs) void {
        fftwf_free(b.in.ptr);
        fftwf_free(b.spec.ptr);
        fftwf_free(b.out.ptr);
    }
};

const Fft = struct {
    n: usize,
    fwd: Plan,
    inv: Plan,
    hs: []f32,
    bufs: []Bufs,

    fn init(gpa: std.mem.Allocator, n: usize, workers: usize, threads: usize) !Fft {
        const bufs = try gpa.alloc(Bufs, workers);
        for (bufs) |*b| b.* = .init(n);
        if (threads > 1) fftwf_plan_with_nthreads(@intCast(threads));
        const fwd = fftwf_plan_dft_r2c_1d(@intCast(n), bufs[0].in.ptr, bufs[0].spec.ptr, 0).?;
        const inv = fftwf_plan_dft_c2r_1d(@intCast(n), bufs[0].spec.ptr, bufs[0].out.ptr, 0).?;
        return .{ .n = n, .fwd = fwd, .inv = inv, .hs = fbuf(n + 2), .bufs = bufs };
    }

    fn deinit(self: *Fft, gpa: std.mem.Allocator) void {
        fftwf_destroy_plan(self.fwd);
        fftwf_destroy_plan(self.inv);
        fftwf_free(self.hs.ptr);
        for (self.bufs) |b| b.deinit();
        gpa.free(self.bufs);
    }

    fn setTaps(self: *Fft, h: []const f32) void {
        const b = self.bufs[0];
        @memset(b.in, 0);
        @memcpy(b.in[0..h.len], h);
        fftwf_execute_dft_r2c(self.fwd, b.in.ptr, b.spec.ptr);
        const s: f32 = 1 / @as(f32, @floatFromInt(self.n));
        for (self.hs, b.spec) |*d, v| d.* = v * s;
    }

    fn convolveBlock(self: *const Fft, b: Bufs) void {
        const own = b.in.ptr == self.bufs[0].in.ptr;
        if (own) fftwf_execute(self.fwd) else fftwf_execute_dft_r2c(self.fwd, b.in.ptr, b.spec.ptr);
        const sp: [][2]f32 = @as([*][2]f32, @ptrCast(b.spec.ptr))[0 .. self.n / 2 + 1];
        const hs: []const [2]f32 = @as([*]const [2]f32, @ptrCast(self.hs.ptr))[0 .. self.n / 2 + 1];
        for (sp, hs) |*z, w| {
            const a = z.*;
            z.* = .{ a[0] * w[0] - a[1] * w[1], a[0] * w[1] + a[1] * w[0] };
        }
        if (own) fftwf_execute(self.inv) else fftwf_execute_dft_c2r(self.inv, b.spec.ptr, b.out.ptr);
    }

    fn oneShot(self: *const Fft, x: []const f32, y: []f32) void {
        const b = self.bufs[0];
        @memcpy(b.in[0..x.len], x);
        @memset(b.in[x.len..], 0);
        self.convolveBlock(b);
        @memcpy(y, b.out[0..y.len]);
    }

    fn osRange(self: *const Fft, b: Bufs, taps: usize, x: []const f32, y: []f32, k0: usize, k1: usize) void {
        const m1 = taps - 1;
        const l = self.n - m1;
        for (k0..k1) |k| {
            const o = k * l;
            const start: isize = @as(isize, @intCast(o)) - @as(isize, @intCast(m1));
            const lo: usize = @intCast(@max(start, 0));
            const hi = @min(o + l, x.len);
            const d0: usize = @intCast(@as(isize, @intCast(lo)) - start);
            @memset(b.in[0..d0], 0);
            @memcpy(b.in[d0..][0 .. hi - lo], x[lo..hi]);
            @memset(b.in[d0 + hi - lo ..], 0);
            self.convolveBlock(b);
            const cnt = @min(l, x.len - o);
            @memcpy(y[o..][0..cnt], b.out[m1..][0..cnt]);
        }
    }

    fn overlapSave(self: *const Fft, io: std.Io, threads: usize, taps: usize, x: []const f32, y: []f32) !void {
        const blocks = (x.len + self.n - taps) / (self.n - taps + 1);
        const t = @min(threads, self.bufs.len, blocks);
        if (t <= 1) return self.osRange(self.bufs[0], taps, x, y, 0, blocks);
        var next: std.atomic.Value(usize) = .init(0);
        const step = @max(1, blocks / (8 * t));
        var g: std.Io.Group = .init;
        errdefer g.cancel(io);
        for (1..t) |k| try g.concurrent(io, osWorker, .{ self, self.bufs[k], taps, x, y, &next, blocks, step });
        self.osWorker(self.bufs[0], taps, x, y, &next, blocks, step);
        try g.await(io);
    }

    fn osWorker(self: *const Fft, b: Bufs, taps: usize, x: []const f32, y: []f32, next: *std.atomic.Value(usize), blocks: usize, step: usize) void {
        while (true) {
            const k0 = next.fetchAdd(step, .monotonic);
            if (k0 >= blocks) return;
            self.osRange(b, taps, x, y, k0, @min(k0 + step, blocks));
        }
    }
};

fn clock(io: std.Io, comptime f: anytype, args: anytype) !u64 {
    const errs = @typeInfo(@typeInfo(@TypeOf(f)).@"fn".return_type.?) == .error_union;
    var t: [runs + 1]u64 = undefined;
    for (&t) |*r| {
        const t0 = std.Io.Clock.awake.now(io);
        if (errs) try @call(.auto, f, args) else @call(.auto, f, args);
        r.* = @intCast(std.Io.Clock.awake.now(io).nanoseconds - t0.nanoseconds);
    }
    std.mem.sort(u64, t[1..], {}, std.sort.asc(u64));
    return t[1 + runs / 2];
}

fn maxRelErr(want: []const f32, got: []const f32) f32 {
    var err: f32 = 0;
    var peak: f32 = 0;
    for (want, got) |a, b| {
        err = @max(err, @abs(a - b));
        peak = @max(peak, @abs(a));
    }
    return err / peak;
}

fn ms(ns: u64) f64 {
    return @as(f64, @floatFromInt(ns)) / 1e6;
}

fn ratio(a: u64, b: u64) f64 {
    return @as(f64, @floatFromInt(a)) / @as(f64, @floatFromInt(b));
}

const Iodim = extern struct { n: c_int, is: c_int, os: c_int };

fn Fftw(comptime T: type) type {
    const p = if (T == f32) "fftwf_" else "fftw_";
    return struct {
        const guru = @extern(*const fn (c_int, *const Iodim, c_int, ?*const Iodim, [*]T, [*]T, [*]T, [*]T, c_uint) callconv(.c) ?Plan, .{ .name = p ++ "plan_guru_split_dft" });
        const plan1d = @extern(*const fn (c_int, [*]T, [*]T, c_int, c_uint) callconv(.c) ?Plan, .{ .name = p ++ "plan_dft_1d" });
        const execute = @extern(*const fn (Plan) callconv(.c) void, .{ .name = p ++ "execute" });
        const destroy = @extern(*const fn (Plan) callconv(.c) void, .{ .name = p ++ "destroy_plan" });

        fn run(plan: Plan, reps: usize) void {
            for (0..reps) |_| execute(plan);
        }
    };
}

fn zigRun(comptime T: type) type {
    return struct {
        fn run(f: *const zc.Fft(T), re: []T, im: []T, scratch: []T, reps: usize) void {
            for (0..reps) |_| f.forward(re, im, scratch);
        }
    };
}

fn fftRace(comptime T: type, gpa: std.mem.Allocator, io: std.Io, max_log2: u6) !void {
    const W = Fftw(T);
    std.debug.print("\ncomplex fft {s}, in place, natural order, vs FFTW_MEASURE interleaved (its fastest)\n{s:>9} {s:>11} {s:>11} {s:>8} {s:>8} {s:>10} {s:>8}\n", .{ @typeName(T), "n", "zig ns", "fftw ns", "zig GF", "fftw GF", "zig/fftw", "max err" });
    var prng = std.Random.DefaultPrng.init(2);
    var lg: u6 = 4;
    while (lg <= max_log2) : (lg += 1) {
        const n = @as(usize, 1) << lg;
        const fb = try gpa.alignedAlloc(T, .@"64", zc.Fft(T).bufferLen(n));
        defer gpa.free(fb);
        const f = zc.Fft(T).init(fb, n);
        const d = try gpa.alignedAlloc(T, .@"64", 8 * n);
        defer gpa.free(d);
        const sr = d[0..n];
        const si = d[n .. 2 * n];
        const re = d[2 * n .. 3 * n];
        const im = d[3 * n .. 4 * n];
        const cx = d[4 * n .. 6 * n];
        const scratch = d[6 * n .. 8 * n];
        const inter = W.plan1d(@intCast(n), cx.ptr, cx.ptr, -1, 0).?;
        defer W.destroy(inter);
        for (sr, si) |*a, *b| {
            a.* = prng.random().float(T) * 2 - 1;
            b.* = prng.random().float(T) * 2 - 1;
        }
        @memcpy(re, sr);
        @memcpy(im, si);
        for (sr, si, 0..) |a, b, k| {
            cx[2 * k] = a;
            cx[2 * k + 1] = b;
        }
        f.forward(re, im, scratch);
        W.execute(inter);
        var err: f64 = 0;
        var peak: f64 = 0;
        for (re, im, 0..) |a, b, k| {
            err = @max(err, @abs(@as(f64, a) - cx[2 * k]), @abs(@as(f64, b) - cx[2 * k + 1]));
            peak = @max(peak, @abs(@as(f64, cx[2 * k])), @abs(@as(f64, cx[2 * k + 1])));
        }
        const reps = @max(1, (@as(usize, 1) << 21) / n);
        const tz = try clock(io, zigRun(T).run, .{ &f, re, im, scratch, reps });
        const ti = try clock(io, W.run, .{ inter, reps });
        const per = @as(f64, @floatFromInt(reps));
        const flops = 5 * @as(f64, @floatFromInt(n)) * @as(f64, @floatFromInt(lg));
        const nz = @as(f64, @floatFromInt(tz)) / per;
        const ni = @as(f64, @floatFromInt(ti)) / per;
        std.debug.print("{d:>9} {d:>11.1} {d:>11.1} {d:>8.2} {d:>8.2} {d:>9.2}x {e:.1}\n", .{ n, nz, ni, flops / nz, flops / ni, ni / nz, err / peak });
    }
}

pub fn main(init: std.process.Init) !void {
    var dbg: std.heap.DebugAllocator(.{}) = .init;
    defer std.debug.assert(dbg.deinit() == .ok);
    const gpa = dbg.allocator();
    const io = init.io;
    var args = std.process.Args.Iterator.init(init.minimal.args);
    _ = args.skip();
    const mode = args.next() orelse "all";
    if (std.mem.eql(u8, mode, "all") or std.mem.eql(u8, mode, "fft")) {
        try fftRace(f32, gpa, io, 22);
        try fftRace(f64, gpa, io, 21);
    }
    if (!std.mem.eql(u8, mode, "all") and !std.mem.eql(u8, mode, "conv")) return;

    const x = try gpa.alloc(f32, total);
    defer gpa.free(x);
    const y = try gpa.alloc(f32, total);
    defer gpa.free(y);
    const ref = try gpa.alloc(f32, total);
    defer gpa.free(ref);
    const h = try gpa.alloc(f32, 1 << 16);
    defer gpa.free(h);
    var prng = std.Random.DefaultPrng.init(1);
    for (x) |*v| v.* = prng.random().float(f32) * 2 - 1;
    for (h) |*v| v.* = prng.random().float(f32) * 2 - 1;

    var cache: [21]?Fft = @splat(null);
    defer for (&cache) |*c| if (c.*) |*f| f.deinit(gpa);
    var best_zig_n: [16]usize = undefined;
    var best_os_n: [16]usize = undefined;

    std.debug.print("\nconvolution, single thread\n{s:>7} {s:>8} {s:>9} {s:>10} {s:>9} {s:>8} {s:>9} {s:>9} {s:>9}\n", .{ "taps", "block", "zig ms", "1-shot ms", "OS ms", "Msamp/s", "vs 1-shot", "vs OS", "max err" });
    {
        var big = try Fft.init(gpa, total, 1, 1);
        defer big.deinit(gpa);
        var m: usize = 32;
        while (m <= 1 << 15) : (m *= 4) {
            const len = total - m + 1;
            const taps = h[0..m];
            const xs = x[0..len];
            const ys = y[0..len];
            var best_zig: u64 = std.math.maxInt(u64);
            var best_os: u64 = std.math.maxInt(u64);
            var os_err: f32 = 0;
            var n = @max(std.math.ceilPowerOfTwoAssert(usize, m + 1), Conv.span);
            const n_end = @min(n * 16, 1 << 20);
            while (n <= n_end) : (n *= 2) {
                const cb = try gpa.alignedAlloc(f32, .@"64", Conv.bufferLen(n, 1));
                defer gpa.free(cb);
                const c = Conv.init(cb, taps, n, 1);
                const tz = try clock(io, Conv.filter, .{ &c, xs, ys });
                if (tz < best_zig) {
                    best_zig = tz;
                    best_zig_n[std.math.log2_int(usize, m)] = n;
                    @memcpy(ref[0..len], ys);
                }
                const lg = std.math.log2_int(usize, n);
                if (cache[lg] == null) cache[lg] = try Fft.init(gpa, n, thread_counts[thread_counts.len - 1], 1);
                const f = &cache[lg].?;
                f.setTaps(taps);
                const to = try clock(io, Fft.overlapSave, .{ f, io, 1, m, xs, ys });
                if (to < best_os) {
                    best_os = to;
                    best_os_n[std.math.log2_int(usize, m)] = n;
                }
                os_err = @max(os_err, maxRelErr(ref[0..len], ys));
            }
            big.setTaps(taps);
            const one = try clock(io, Fft.oneShot, .{ &big, xs, ys });
            const err = @max(os_err, maxRelErr(ref[0..len], ys));
            std.debug.print("{d:>7} {d:>8} {d:>9.2} {d:>10.2} {d:>9.2} {d:>8.0} {d:>8.2}x {d:>8.2}x {e:.1}\n", .{ m, best_zig_n[std.math.log2_int(usize, m)], ms(best_zig), ms(one), ms(best_os), @as(f64, @floatFromInt(len)) / (ms(best_zig) * 1e3), ratio(one, best_zig), ratio(best_os, best_zig), err });
        }
    }

    if (fftwf_init_threads() == 0) return error.FftwThreads;
    defer fftwf_cleanup_threads();
    std.debug.print("\nthreads: block size swept per thread count for both sides\n{s:>7} {s:>7} {s:>8} {s:>8} {s:>9} {s:>10} {s:>9} {s:>9} {s:>9} {s:>11} {s:>8}\n", .{ "threads", "taps", "zig blk", "OS blk", "zig ms", "1-shot ms", "OS ms", "vs 1-shot", "vs OS", "zig scaling", "max err" });
    var zig_single: [parallel_taps.len]u64 = undefined;
    for (thread_counts) |t| {
        var big = try Fft.init(gpa, total, 1, t);
        defer big.deinit(gpa);
        for (parallel_taps, 0..) |m, i| {
            const len = total - m + 1;
            const taps = h[0..m];
            const xs = x[0..len];
            const ys = y[0..len];
            var tz: u64 = std.math.maxInt(u64);
            var to: u64 = std.math.maxInt(u64);
            var zn: usize = 0;
            var on: usize = 0;
            var err: f32 = 0;
            var n = std.math.ceilPowerOfTwoAssert(usize, m + 1);
            const n_end = @min(n * 16, 1 << 20);
            while (n <= n_end) : (n *= 2) {
                const cb = try gpa.alignedAlloc(f32, .@"64", Conv.bufferLen(n, t));
                defer gpa.free(cb);
                const c = Conv.init(cb, taps, n, t);
                const a = try clock(io, Conv.filterParallel, .{ &c, io, xs, ys });
                if (a < tz) {
                    tz = a;
                    zn = n;
                }
                @memcpy(ref[0..len], ys);
                const lg = std.math.log2_int(usize, n);
                if (cache[lg] == null) cache[lg] = try Fft.init(gpa, n, thread_counts[thread_counts.len - 1], 1);
                const f = &cache[lg].?;
                f.setTaps(taps);
                const b = try clock(io, Fft.overlapSave, .{ f, io, t, m, xs, ys });
                if (b < to) {
                    to = b;
                    on = n;
                }
                err = @max(err, maxRelErr(ref[0..len], ys));
            }
            if (t == 1) zig_single[i] = tz;
            big.setTaps(taps);
            const one = try clock(io, Fft.oneShot, .{ &big, xs, ys });
            err = @max(err, maxRelErr(ref[0..len], ys));
            std.debug.print("{d:>7} {d:>7} {d:>8} {d:>8} {d:>9.2} {d:>10.2} {d:>9.2} {d:>8.2}x {d:>8.2}x {d:>10.2}x {e:.1}\n", .{ t, m, zn, on, ms(tz), ms(one), ms(to), ratio(one, tz), ratio(to, tz), ratio(zig_single[i], tz), err });
        }
    }
}
