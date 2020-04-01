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
)
