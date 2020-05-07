package(default_visibility = ["//visibility:public"])

sh_binary(
    name = "build_pip_package",
    srcs = ["build_pip_package.sh"],
    data = [
        "//tf_encrypted/primitives/paillier:paillier_py",
        "//tf_encrypted/primitives/sodium:sodium_py",
        "//tf_encrypted/test:test_py",
    ] + glob([
        "setup.py",
        "MANIFEST.in",
        "**/__init__.py",
    ]),
)
