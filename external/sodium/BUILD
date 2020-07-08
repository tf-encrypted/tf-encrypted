load("@rules_foreign_cc//tools/build_defs:configure.bzl", "configure_make")

configure_make(
    name = "lib",
    configure_options = [
        "--disable-shared",
        "--enable-static",
        "--disable-debug",
        "--disable-dependency-tracking",
        "--with-pic",
    ],
    lib_source = "@sodium//:.",
    static_libraries = ["libsodium.a"],
    visibility = ["//visibility:public"],
    # libsodium script uses its own linker, so explicitly unset the default
    # libtool from rules_foreign_cc
    # (otherwise, if the libtool from bazel's toolchain is supplied,
    # the build script has problems with passing output file to libtool)
    # see bazelbuild/rules_foreign_cc#315
    configure_env_vars = {
        "AR": "",
    },
)
