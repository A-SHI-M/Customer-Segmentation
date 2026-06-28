from setuptools import setup, find_packages

HYPEN_E_DOT = "-e ."


def get_requirements(file_path: str) -> list:
    requirements = []
    with open(file_path) as f:
        requirements = [req.strip() for req in f.readlines()]
        if HYPEN_E_DOT in requirements:
            requirements.remove(HYPEN_E_DOT)
    return requirements


setup(
    name="customer-segmentation",
    version="0.0.1",
    author="A-SHI-M",
    author_email="1647dasashim1647@gmail.com",
    packages=find_packages(where="src"),
    package_dir={"": "src"},
    install_requires=get_requirements("requirements.txt"),
)
