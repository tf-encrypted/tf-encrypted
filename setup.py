from setuptools import setup

setup(
    name='tf-encrypted',
    version='0.0.1-rc',
    packages=['tensorflow_encrypted', ],
    install_requires=['numpy>=1.14.0'],
    extra_requires={
        'tf': ["tensorflow>=1.10.0"],
        'tf_gpu': ["tensorflow-gpu>=1.10.0"],
    },
    license='',
    long_description=open('README.md').read(),
)
