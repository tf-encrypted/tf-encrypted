LIBSODIUM_VER_TAG=1.0.16

TF_CFLAGS=$(shell python -c 'import tensorflow as tf; print(" ".join(tf.sysconfig.get_compile_flags()))' 2>/dev/null)
TF_LFLAGS=$(shell python -c 'import tensorflow as tf; print(" ".join(tf.sysconfig.get_link_flags()))' 2>/dev/null)

all: build-op

build-lib-sodium:
	# TODO CHECK IF SODIUM LIB EXISTS FIRST
	git clone https://github.com/jedisct1/libsodium.git
	cd libsodium && git checkout $(LIBSODIUM_VER_TAG)
	cd libsodium && ./autogen.sh && ./configure --disable-shared --enable-static \
		--disable-debug --disable-dependency-tracking --with-pic
	$(MAKE) -C libsodium
	$(MAKE) -C libsodium install
	rm -Rf libsodium

build-op:
	g++ -std=c++11 -shared secure_random.cc -o secure_random.so -fPIC $(TF_CFLAGS) $(TF_LFLAGS) -O2 -lsodium
