from setuptools import setup, find_packages

setup(
    name="quantum-multimodal-framework",
    version="1.0.0",
    description="Hybrid Quantum-Classical Multimodal Framework for Sentiment Analysis and Image Captioning",
    author="Hieu Nguyen",
    packages=find_packages(),
    python_requires=">=3.9",
    install_requires=[
        "pennylane>=0.40",
        "pennylane-lightning>=0.40",
        "torch>=2.0",
        "torchvision>=0.15",
        "transformers>=4.30",
        "datasets",
        "nltk",
        "tokenizers",
        "Pillow",
        "opencv-python",
        "scikit-learn",
        "matplotlib",
        "seaborn",
        "tqdm",
        "tensorboard",
        "pyyaml",
        "scipy",
    ],
)
