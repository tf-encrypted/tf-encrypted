package(default_visibility = ["//visibility:public"])


py_library(
    name = "test_py",
    srcs = ([
        "__init__.py",
        "execution_context.py",
    ]),
)

py_test(
    name = "execution_context_py_test",
    srcs = [
        "execution_context_test.py",
    ],
    main = "execution_context_test.py",
    deps = [
        ":test_py",
    ],
)
