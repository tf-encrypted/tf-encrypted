LIBSODIUM_VER_TAG=1.0.16
LIBSODIUM_DIR=build/libsodium-$(LIBSODIUM_VER_TAG)

TF_CFLAGS=$(shell python -c 'import tensorflow as tf; print(" ".join(tf.sysconfig.get_compile_flags()))' 2>/dev/null)
TF_LFLAGS=$(shell python -c 'import tensorflow as tf; print(" ".join(tf.sysconfig.get_link_flags()))' 2>/dev/null)
PACKAGE_DIR=../../tf_encrypted/operations

SODIUM_INSTALL = $(shell pwd)/build

SECURE_OUT = $(PACKAGE_DIR)/secure_random/secure_random_module.so
LIBSODIUM_OUT = $(SODIUM_INSTALL)/lib/libsodium.a

all: $(SECURE_OUT)

.PHONY: all

$(LIBSODIUM_OUT):
	wget https://github.com/jedisct1/libsodium/archive/$(LIBSODIUM_VER_TAG).tar.gz
	mkdir -p build
	tar -xvf $(LIBSODIUM_VER_TAG).tar.gz -C build
	cd $(LIBSODIUM_DIR) && ./autogen.sh && ./configure --disable-shared --enable-static \
		--disable-debug --disable-dependency-tracking --with-pic --prefix=$(SODIUM_INSTALL)
	$(MAKE) -C $(LIBSODIUM_DIR)
	$(MAKE) -C $(LIBSODIUM_DIR) install
	rm -f $(LIBSODIUM_VER_TAG).tar.gz


$(SECURE_OUT): $(LIBSODIUM_OUT) secure_random.cc
	mkdir -p $(PACKAGE_DIR)/secure_random
	g++ -std=c++11 -shared secure_random.cc -o $(SECURE_OUT) \
		-fPIC $(TF_CFLAGS) $(TF_LFLAGS) -O2 -I$(SODIUM_INSTALL)/include -L$(SODIUM_INSTALL)/lib -lsodium

clean:
	$(MAKE) -C $(LIBSODIUM_DIR) uninstall
	rm -fR build
	rm -fR $(SECURE_OUT)

.PHONY: clean
