const std = @import("std");

pub fn build(b: *std.Build) void {
    const target = b.standardTargetOptions(.{});
    const optimize = b.standardOptimizeOption(.{});
    const radixz = b.addModule("radixz", .{
        .root_source_file = b.path("src/radixz.zig"),
        .target = target,
        .optimize = optimize,
    });

    const bench = b.addExecutable(.{
        .name = "bench",
        .root_module = b.createModule(.{
            .root_source_file = b.path("src/bench.zig"),
            .target = target,
            .optimize = optimize,
            .link_libc = true,
            .imports = &.{.{ .name = "radixz", .module = radixz }},
        }),
    });
    bench.root_module.linkSystemLibrary("fftw3f", .{});
    bench.root_module.linkSystemLibrary("fftw3", .{});
    bench.root_module.linkSystemLibrary("fftw3f_omp", .{});
    const run_bench = b.addRunArtifact(bench);
    if (b.args) |a| run_bench.addArgs(a);
    b.step("bench", "run benchmark").dependOn(&run_bench.step);

    inline for (.{ .{ "test", "run tests", false }, .{ "stress", "run long randomized stress tests", true } }) |cfg| {
        const opts = b.addOptions();
        opts.addOption(bool, "stress", cfg[2]);
        const tests = b.addTest(.{ .root_module = b.createModule(.{
            .root_source_file = b.path("src/tests.zig"),
            .target = target,
            .optimize = optimize,
            .imports = &.{
                .{ .name = "radixz", .module = radixz },
                .{ .name = "options", .module = opts.createModule() },
            },
        }) });
        b.step(cfg[0], cfg[1]).dependOn(&b.addRunArtifact(tests).step);
    }
}
