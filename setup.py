from pathlib import Path

from setuptools import find_packages, setup


# Collect top-level scripts so they can be imported after installation.
root = Path(__file__).parent
py_modules = [
    p.stem
    for p in root.glob("*.py")
    if p.is_file() and p.suffix == ".py" and p.stem not in {"setup", "__init__"}
]

setup(
    name="physical-atari",
    version="0.0.0",
    description="Research scripts for the physical Atari setup and Swift-Sarsa experiments.",
    packages=find_packages(include=["framework", "framework.*"]),
    py_modules=py_modules,
    package_dir={"": "."},
    python_requires=">=3.9",
)
