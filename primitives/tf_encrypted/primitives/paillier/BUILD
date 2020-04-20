package(default_visibility = ["//visibility:public"])

py_library(
    name = "paillier_py",
    srcs = ([
        "__init__.py",
        "primitives.py",
    ]),
)

py_test(
    name = "paillier_py_test",
    srcs = [
        "primitives_test.py",
    ],
    main = "primitives_test.py",
    deps = [
        ":paillier_py",
    ],
)
