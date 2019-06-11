package(default_visibility = ["//visibility:public"])

cc_binary(
    name = 'python/ops/_ntl_ops.so',
    srcs = ["cc/kernels/ntl_matrix_ops.cc",
            "cc/kernels/ntl_matrix.cc",
            "cc/kernels/ntl_matrix.h"] +
            glob(["deps/include/NTL/*.h"]),
    linkshared = 1,
    deps = [
        "@local_config_tf//:libtensorflow_framework",
        "@local_config_tf//:tf_header_lib",
    ],
    copts = ["-pthread", "-std=c++11", "-D_GLIBCXX_USE_CXX11_ABI=0", "-Intl/deps/include"],
    linkopts = ["-Lntl/deps/lib", "-lntl", "-lgmp", "-lm"]
)

cc_test (
    name = "ntl_matrix_test",
    size = "small",
    srcs = ["cc/kernels/ntl_matrix_test.cc"],
    deps = [
        "@local_config_tf//:libtensorflow_framework",
        "@local_config_tf//:tf_header_lib",
    ],
    copts = ["-pthread", "-std=c++11", "-D_GLIBCXX_USE_CXX11_ABI=0", "-Intl/deps/include"],
    linkopts = ["-Lntl/deps/lib", "-lntl", "-lgmp", "-lm"]
)

py_library(
    name = "ntl_ops_py",
    srcs = ([
        "python/ops/ntl_ops.py",
    ]),
    data = [
        ":python/ops/_ntl_ops.so"
    ],
)

py_test(
    name = "ntl_ops_py_test",
    srcs = [
        "python/ops/ntl_ops_test.py"
    ],
    main = "python/ops/ntl_ops_test.py",
    deps = [
        ":ntl_ops_py",
    ],
    srcs_version = "PY2AND3",
)

py_library(
    name = "ntl_py",
    srcs = ([
        "__init__.py",
        "python/__init__.py",
        "python/ops/__init__.py",
    ]),
    deps = [
        ":ntl_ops_py"
    ],
)