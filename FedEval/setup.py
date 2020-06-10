import setuptools


setuptools.setup(
    name="FedEval",
    version="0.5",
    author="Di Chai",
    author_email="dchai@connect.ust.hk",
    description="A Comprehensive Evaluation Model for Federated Learning",
    long_description="## FedEval\n",
    long_description_content_type="text/markdown",
    url="https://github.com/Di-Chai/FMLs",
    packages=setuptools.find_packages(),
    include_package_data=True,
    install_requires=['numpy', 'scipy'],
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
    ],
)