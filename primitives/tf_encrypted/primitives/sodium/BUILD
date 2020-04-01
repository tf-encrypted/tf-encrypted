package(default_visibility = ["//visibility:public"])

cc_binary(
    name = "python/_sodium_module.so",
    srcs = [
        "cc/ops.cc",
        "cc/easy_box_kernels.cc",
    ],
    linkshared = 1,
    deps = [
        "@local_config_tf//:libtensorflow_framework",
        "@local_config_tf//:tf_header_lib",
        "@sodium//:lib",
    ],
    # copts = ["-pthread", "-std=c++11", "-D_GLIBCXX_USE_CXX11_ABI=0", "-fPIC"],
)

py_library(
    name = "sodium_py",
    srcs = ([
        "__init__.py",
        "python/__init__.py",
        "python/easy_box.py",
    ]),
    data = [
        ":python/_sodium_module.so"
    ],
)

py_test(
    name = "sodium_py_test",
    srcs = [
        "python/easy_box_test.py",
    ],
    main = "python/easy_box_test.py",
    deps = [
        ":sodium_py",
    ],
)