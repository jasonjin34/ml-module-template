# -*- coding: utf-8 -*-
import base64
import os
from datetime import datetime
from distutils.command.upload import upload as upload_orig
from os import path

try:
    from setuptools import setup, find_packages
except ImportError:
    print("setuptools version 40 or higher required")
    raise

here = path.abspath(path.dirname(__file__))

requirements_path = path.join(here, 'requirements.txt')
if path.isfile(requirements_path):
    with open(requirements_path, encoding='utf-8') as f:
        REQUIRED = f.read().split('\n')
else:
    REQUIRED = []
print("required", REQUIRED)


def get_pypi_fixed_version(version):
    """
    TC dev builds will be snapshots, named <version>-SNAPSHOT
    This causes warnings for the pypi versioning schema, so we will just
    replace it with .dev + timestamp

    :param version: version string
    :returns: fixed version string
    """
    return version.replace(
        "-SNAPSHOT", ".dev{}".format(datetime.utcnow().strftime("%Y%m%d%H%M"))
    )


class upload(upload_orig):
    def _read_pypirc(self):
        config = super(upload, self)._read_pypirc()
        if config != {}:
            config["password"] = str(base64.b64decode(config["password"]), "utf-8")
            config["username"] = str(base64.b64decode(config["username"]), "utf-8")
        return config


setup(
    name="${PROJECT_NAME}",
    version=get_pypi_fixed_version("${VERSION}"),
    description="GoG model segmentation",
    package_dir={"": "."},
    packages=find_packages(),
    setup_requires=["wheel"],
    install_requires=REQUIRED,
    cmdclass={"upload": upload},
)
