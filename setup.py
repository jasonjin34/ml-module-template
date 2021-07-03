from os import path


try:
    from setuptools import setup, find_packages
except ImportError:
    print("setuptools version 40 or higher required")
    raise

here = path.abspath(path.dirname(__file__))
requirements_path = path.join(here, 'src', 'main', 'python', 'requirements.txt')

if path.isfile(requirements_path):
    with open(requirements_path, encoding='utf-8') as f:
        REQUIRED = f.read().split('\n')
else:
    REQUIRED = []

print(REQUIRED)


def read(fname):
    this_directory = path.abspath(path.dirname(__file__))
    with open(path.join(this_directory, fname), encoding='utf-8') as f:
        long_description = f.read()
    return long_description

setup(
    name="ml-module-template",
    version=versioneer.get_version(),
    cmdclass=versioneer.get_cmdclass(),
    description="Pytorch Training Module Template",
    packages=find_packages(),
    package_dir={'': 'src/main/python'},
    install_requires=REQUIRED
)
