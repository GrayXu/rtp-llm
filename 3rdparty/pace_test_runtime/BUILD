load("//bazel:defs.bzl", "runfiles_alias")

package(default_visibility = ["//visibility:public"])

# Service binaries only: the KVCM SDK remains the branch's existing dependency.
genrule(
    name = "meta_binary",
    testonly = True,
    tags = ["manual"],
    srcs = select({
        "@//:using_cuda12_arm": ["@pace_test_meta_rpm_aarch64//file"],
        "@//:using_arm": ["@pace_test_meta_rpm_aarch64//file"],
        "//conditions:default": ["@pace_test_meta_rpm//file"],
    }),
    outs = ["tair_mempool_meta_service"],
    cmd = """
        stage=$$(mktemp -d)
        trap 'rm -rf "$$stage"' EXIT
        rpm2cpio $(SRCS) | (cd "$$stage" && cpio -idm --quiet)
        cp "$$stage/usr/local/tair_mempool/bin/tair_mempool_meta_service" "$@"
    """,
)

genrule(
    name = "server_binary_x86_64",
    testonly = True,
    tags = ["manual"],
    srcs = ["@pace_test_server_rpm//file"],
    outs = ["tair_mempool_server", "libtent_shared.so.5"],
    cmd = """
        stage=$$(mktemp -d)
        trap 'rm -rf "$$stage"' EXIT
        rpm2cpio $(SRCS) | (cd "$$stage" && cpio -idm --quiet)
        cp "$$stage/usr/local/tair_mempool/bin/tair_mempool_server" "$(@D)/tair_mempool_server"
        cp -L "$$stage/usr/local/tair_mempool/lib/libtent_shared.so.5" "$(@D)/libtent_shared.so.5"
    """,
)

genrule(
    name = "server_binary_aarch64",
    testonly = True,
    tags = ["manual"],
    srcs = ["@pace_test_server_rpm_aarch64//file"],
    outs = ["aarch64/tair_mempool_server", "aarch64/libtent_shared.so.5"],
    cmd = """
        stage=$$(mktemp -d)
        trap 'rm -rf "$$stage"' EXIT
        rpm2cpio $(SRCS) | (cd "$$stage" && cpio -idm --quiet)
        mkdir -p "$(@D)/aarch64"
        cp "$$stage/usr/local/tair_mempool/bin/tair_mempool_server" "$(@D)/aarch64/tair_mempool_server"
        cp -L "$$stage/usr/local/tair_mempool/lib/libtent_shared.so.5" "$(@D)/aarch64/libtent_shared.so.5"
    """,
)

runfiles_alias(
    name = "server_binary_aarch64_runfile",
    testonly = True,
    src = ":aarch64/tair_mempool_server",
    runfile_path = "3rdparty/pace_test_runtime/tair_mempool_server",
)

filegroup(
    name = "server_binary",
    testonly = True,
    tags = ["manual"],
    srcs = select({
        "@//:using_cuda12_arm": [":server_binary_aarch64", ":server_binary_aarch64_runfile"],
        "@//:using_arm": [":server_binary_aarch64", ":server_binary_aarch64_runfile"],
        "//conditions:default": [":server_binary_x86_64"],
    }),
)

filegroup(
    name = "runtime",
    testonly = True,
    tags = ["manual"],
    srcs = [":meta_binary", ":server_binary"],
)
