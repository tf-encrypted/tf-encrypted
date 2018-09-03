from setuptools import setup

setup(
    name='tf-encrypted',
    version='0.0.1rc',
    packages=['tensorflow_encrypted',],
    python_requires='>=3.6',
    install_requires=[
        'tensorflow>=1.10.0',
        'numpy>=1.14.0',
    ],
    extra_requires={
        'tf': ["tensorflow>=1.0.0"],
        'tf_gpu': ["tensorflow-gpu>=1.0.0"],
    },
    license='Apache License 2.0',
    url="https://github.com/mortendahl/tf-encrypted",
    description='Layer on top of TensorFlow for doing machine learning on encrypted data.',
)
