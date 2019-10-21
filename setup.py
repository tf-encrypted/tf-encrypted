"""Installing with setuptools."""
import setuptools

with open("README.md", "r") as fh:
  long_description = fh.read()

setuptools.setup(
    name="tf-encrypted",
    version="0.6.0-rc0",
    packages=setuptools.find_packages(),
    package_data={'tf_encrypted': [
        'operations/secure_random/*.so',
        'convert/*.yaml',
    ]},
    python_requires=">=3.5",
    install_requires=[
        "tensorflow ==2.0.0",
        "numpy >=1.14.0",
        "pyyaml >=5.1",
    ],
    extras_require={
        "tf": ["tensorflow ==2.0.0"],
    },
    license="Apache License 2.0",
    url="https://github.com/tf-encrypted/tf-encrypted",
    description="A Framework for Machine Learning on Encrypted Data.",
    long_description=long_description,
    long_description_content_type="text/markdown",
    author="The TF Encrypted Authors",
    author_email="contact@tf-encrypted.io",
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: Apache Software License",
        "Development Status :: 2 - Pre-Alpha",
        "Operating System :: OS Independent",
        "Topic :: Scientific/Engineering :: Artificial Intelligence",
        "Topic :: Security :: Cryptography",
    ]
)
