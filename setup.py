import setuptools

with open("README.md", "r") as fh:
    long_description = fh.read()

setuptools.setup(
    name="tf-encrypted",
    version="0.2.0",
    packages=setuptools.find_packages(),
    python_requires=">=3.5",
    install_requires=[
        "tensorflow>=1.9.0",
        "numpy>=1.14.0"
    ],
    extra_requires={
        "tf": ["tensorflow>=1.9.0"]
    },
    license="Apache License 2.0",
    url="https://github.com/mortendahl/tf-encrypted",
    description="Layer on top of TensorFlow for doing machine learning on encrypted data.",
    long_description=long_description,
    long_description_content_type="text/markdown",
    author="Morten Dahl",
    author_email="morten@dropoutlabs.com",
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: Apache Software License",
        "Development Status :: 2 - Pre-Alpha",
        "Operating System :: OS Independent",
        "Topic :: Scientific/Engineering :: Artificial Intelligence",
        "Topic :: Security :: Cryptography"
    ]
)
