import setuptools

with open("README.md", "r") as fh:
    long_description = fh.read()

setuptools.setup(
    name="video-cassette",
    version="0.0.3",
    author="Jake Ledoux",
    author_email="contactjakeledoux@gmail.com",
    description="A simple python package for encoding any file into video form.",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/jakeledoux/video_cassette",
    packages=setuptools.find_packages(),
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
    ],
    python_requires='>=3.6'
)