all: ntl-op

EXT=dylib

ntl/deps/lib/libgmp.$(EXT):
	wget https://gmplib.org/download/gmp/gmp-6.1.2.tar.xz
	tar -xvf gmp-6.1.2.tar.xz
	mkdir -p ntl/deps
	cd gmp-6.1.2 && ./configure --prefix=$(shell pwd)/ntl/deps --enable-static=no
	$(MAKE) -C gmp-6.1.2
	$(MAKE) -C gmp-6.1.2 install
	rm -R gmp-6.1.2.tar.xz gmp-6.1.2

ntl/deps/lib/libntl.$(EXT):
	wget https://www.shoup.net/ntl/ntl-11.3.2.tar.gz
	tar -xvf ntl-11.3.2.tar.gz
	mkdir -p ntl/deps
	cd ntl-11.3.2/src && ./configure PREFIX=$(shell pwd)/ntl/deps GMP_PREFIX=$(shell pwd)/ntl/deps SHARED=on
	$(MAKE) -C ntl-11.3.2/src
	$(MAKE) -C ntl-11.3.2/src install
	rm -R ntl-11.3.2.tar.gz ntl-11.3.2

ntl-op: ntl/deps/lib/libgmp.$(EXT) ntl/deps/lib/libntl.$(EXT)
	bazel build ntl:python/ops/_ntl_ops.so

.PHONY: ntl-op

test:
	bazel build ntl:ntl_ops_py_test
	./bazel-bin/ntl/ntl_ops_py_test

clean:
	rm -R ntl/deps
	bazel clean

.PHONY: clean test