"""Installing with setuptools."""
import setuptools

setuptools.setup(
    name="tf-encrypted-primitives",
    version="0.0.1",
    packages=setuptools.find_namespace_packages(include=["tf_encrypted.*"]),
    package_data={"": ["*.so"]},
    python_requires=">=3.6",
    install_requires=["tensorflow ==2.1.0", "numpy >=1.14.0"],
    license="Apache License 2.0",
    url="https://github.com/tf-encrypted/tf-encrypted",
    description="A Framework for Machine Learning on Encrypted Data.",
    long_description="",
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
    ],
)
