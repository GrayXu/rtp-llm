package(default_visibility = ["//visibility:public"])

# Service binaries only: the KVCM SDK remains the branch's existing dependency.
genrule(
    name = "meta_binary",
    testonly = True,
    tags = ["manual"],
    srcs = ["@pace_test_meta_rpm//file"],
    outs = ["tair_mempool_meta_service"],
    cmd = """
        stage=$$(mktemp -d)
        trap 'rm -rf "$$stage"' EXIT
        rpm2cpio $(location @pace_test_meta_rpm//file) | (cd "$$stage" && cpio -idm --quiet)
        cp "$$stage/usr/local/tair_mempool/bin/tair_mempool_meta_service" "$@"
    """,
)

genrule(
    name = "server_binary",
    testonly = True,
    tags = ["manual"],
    srcs = ["@pace_test_server_rpm//file"],
    outs = ["tair_mempool_server", "libtent_shared.so.4"],
    cmd = """
        stage=$$(mktemp -d)
        trap 'rm -rf "$$stage"' EXIT
        rpm2cpio $(location @pace_test_server_rpm//file) | (cd "$$stage" && cpio -idm --quiet)
        cp "$$stage/usr/local/tair_mempool/bin/tair_mempool_server" "$(@D)/tair_mempool_server"
        cp -L "$$stage/usr/local/tair_mempool/lib/libtent_shared.so.4" "$(@D)/libtent_shared.so.4"
    """,
)

filegroup(
    name = "runtime",
    testonly = True,
    tags = ["manual"],
    srcs = [":meta_binary", ":server_binary"],
)
