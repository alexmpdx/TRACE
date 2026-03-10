"""Setup for editable installs (pip install -e .)."""

from setuptools import find_packages, setup

setup(
    name="landmark-locator",
    version="0.1.0",
    description="Heatmap-based landmark detection for Drosophila wing images",
    python_requires=">=3.9",
    packages=find_packages(include=["landmark_locator*"]),
    install_requires=[
        "torch",
        "torchvision",
        "numpy",
        "opencv-python",
        "albumentations",
        "pyyaml",
        "tqdm",
        "scikit-learn",
    ],
    extras_require={
        "gui": ["PyQt5", "matplotlib"],
        "dev": ["pandas", "matplotlib", "pre-commit", "black", "isort", "flake8"],
    },
    entry_points={
        "console_scripts": [
            "landmark-train=landmark_locator.scripts.train:main",
            "landmark-predict=landmark_locator.scripts.predict:main",
            "landmark-visualize=landmark_locator.scripts.visualize:main",
        ],
    },
)
