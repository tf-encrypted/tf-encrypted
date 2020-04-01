load("@bazel_tools//tools/build_defs/repo:http.bzl", "http_archive")

http_archive(
   name = "rules_foreign_cc",
   url = "https://github.com/bazelbuild/rules_foreign_cc/archive/master.zip",
   strip_prefix = "rules_foreign_cc-master",
)

load("@rules_foreign_cc//:workspace_definitions.bzl", "rules_foreign_cc_dependencies")

rules_foreign_cc_dependencies([])

load("//external/tf:tf_configure.bzl", "tf_configure")

tf_configure(name = "local_config_tf")

http_archive(
    name = "sodium",
    build_file = "sodium/BUILD",
    patch_cmds = ["./autogen.sh"],
    url = "https://github.com/jedisct1/libsodium/archive/1.0.17.tar.gz",
    strip_prefix = "libsodium-1.0.17",
    sha256 = "602e07029c780e154347fb95495b13ce48709ae705c6cff927ecb0c485b95672",
)

local_repository(
    name = "primitives",
    path = "primitives/",
)
