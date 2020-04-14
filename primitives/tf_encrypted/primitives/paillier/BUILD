package(default_visibility = ["//visibility:public"])

py_library(
    name = "paillier_py",
    srcs = ([
        "__init__.py",
    ]),
)

py_test(
    name = "paillier_py_test",
    srcs = [
        "python/easy_box_test.py",
    ],
    main = "python/easy_box_test.py",
    deps = [
        ":paillier_py",
    ],
)
