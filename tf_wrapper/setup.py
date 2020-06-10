import setuptools

setuptools.setup(
    name="tf_wrapper",
    version="0.5",
    author="Di Chai",
    author_email="dchai@connect.ust.hk",
    description="A useful tool for doing ML using tensorflow",
    long_description="## TF Wrapper",
    long_description_content_type="text/markdown",
    url="https://github.com/Di-Chai/tf_wrapper",
    packages=setuptools.find_packages(),
    include_package_data=True,
    install_requires=['numpy', 'scipy'],
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
    ],
)