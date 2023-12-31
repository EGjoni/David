from setuptools import setup, find_packages

setup(
    name="David",
    version="0.0.1",
    author="Eron Gjoni",
    author_email="rufsketch1@gmail.com",
    description="A huggingface transformers module for memory-efficient, on-the-fly creation of Goliath style frankenmerges",
    long_description=open("README.md").read(),
    long_description_content_type="text/markdown",
    url="https://github.com/EGjoni/David",
    package_dir={"": "."},
    packages=find_packages(),
    install_requires=open("requirements.txt").read().splitlines(),
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
    ],
    python_requires=">=3.7",
)
