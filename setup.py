from setuptools import setup, find_packages

setup(
    name="ur5e_kinematics",
    version="0.1.0",
    description="Forward/inverse kinematics, Jacobians, dynamics, and control for the UR5e",
    packages=find_packages(),
    python_requires=">=3.8",
    install_requires=[
        "numpy>=1.20",
    ],
    extras_require={
        "sim": ["mujoco>=3.0"],
        "test": ["pytest"],
    },
)
