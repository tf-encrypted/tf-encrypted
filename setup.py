import setuptools

with open("README.md", "r") as fh:
    long_description = fh.read()

setuptools.setup(
    name="tf-encrypted",
    version="0.5.2",
    packages=setuptools.find_packages(),
    package_data={'tf_encrypted': [
        'operations/secure_random/*.so',
        'convert/*.yaml'
    ]},
    python_requires=">=3.5",
    install_requires=[
        "tensorflow>=1.12.0,<2",
        "numpy>=1.14.0",
        "pyyaml>=5.1"
    ],
    extra_requires={
        "tf": ["tensorflow>=1.12.0,<2"]
    },
    license="Apache License 2.0",
    url="https://github.com/tf-encrypted/tf-encrypted",
    description="Layer on top of TensorFlow for doing machine learning on encrypted data.",
    long_description=long_description,
    long_description_content_type="text/markdown",
    author="The TF Encrypted Authors",
    author_email="tfencrypted@gmail.com",
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: Apache Software License",
        "Development Status :: 2 - Pre-Alpha",
        "Operating System :: OS Independent",
        "Topic :: Scientific/Engineering :: Artificial Intelligence",
        "Topic :: Security :: Cryptography"
    ]
)
