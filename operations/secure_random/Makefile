LIBSODIUM_VER_TAG=1.0.16
LIBSODIUM_DIR=build/libsodium-$(LIBSODIUM_VER_TAG)

TF_CFLAGS=$(shell python -c 'import tensorflow as tf; print(" ".join(tf.sysconfig.get_compile_flags()))' 2>/dev/null)
TF_LFLAGS=$(shell python -c 'import tensorflow as tf; print(" ".join(tf.sysconfig.get_link_flags()))' 2>/dev/null)
PACKAGE_DIR=../../tf_encrypted/operations

WGET_PATH=$(shell which wget)

all: build-op

wgetcheck:
ifeq (,$(WGET_PATH))
ifeq (,$(BYPASS_WGET_CHECK))
	$(error "wget must be installed")
endif
endif

build-libsodium: wgetcheck
ifeq ("$(wildcard /usr/local/lib/libsodium.a)","")
		wget https://github.com/jedisct1/libsodium/archive/$(LIBSODIUM_VER_TAG).tar.gz
		mkdir -p build
		tar -xvf $(LIBSODIUM_VER_TAG).tar.gz -C build
		cd $(LIBSODIUM_DIR) && ./autogen.sh && ./configure --disable-shared --enable-static \
			--disable-debug --disable-dependency-tracking --with-pic
		$(MAKE) -C $(LIBSODIUM_DIR)
		$(MAKE) -C $(LIBSODIUM_DIR) install
		rm -f $(LIBSODIUM_VER_TAG).tar.gz
endif


build-op: secure_random.cc build-libsodium
	mkdir -p $(PACKAGE_DIR)/secure_random
	g++ -std=c++11 -shared secure_random.cc -o $(PACKAGE_DIR)/secure_random/secure_random.so \
		-fPIC $(TF_CFLAGS) $(TF_LFLAGS) -O2 -lsodium

clean:
	$(MAKE) -C $(LIBSODIUM_DIR) uninstall
	rm -fR build
	rm -fR $(PACKAGE_DIR)/secure_random/secure_random.so
