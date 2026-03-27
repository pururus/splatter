from setuptools import setup, find_packages

setup(
    name="splatter-a-video",
    version="0.1.0",
    description="Splatter A Video - Dynamic Gaussian Splatting for Video",
    author="",
    packages=find_packages(exclude=["tests", "*.tests", "*.tests.*", "tests.*"]),
    python_requires=">=3.8",
    install_requires=[
        # Core dependencies - adjust versions as needed
        "torch>=2.0.0",
        "torchvision>=0.15.0",
        "torchaudio>=2.0.0",
        "numpy",
        "opencv-python",
        "scipy",
        "pillow",
        "pyyaml",
        "imageio",
        "imageio-ffmpeg",
    ],
    extras_require={
        "dev": [
            "pytest",
            "black",
            "isort",
            "pylint",
        ],
    },
    include_package_data=True,
    zip_safe=False,
)
