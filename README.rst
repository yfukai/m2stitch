M2Stitch
========

|PyPI| |Python Version| |License|

|Read the Docs| |Tests| |Codecov|

|pre-commit| |Black| |Zenodo|

.. |PyPI| image:: https://img.shields.io/pypi/v/m2stitch.svg
   :target: https://pypi.org/project/m2stitch/
   :alt: PyPI
.. |Python Version| image:: https://img.shields.io/pypi/pyversions/m2stitch
   :target: https://pypi.org/project/m2stitch
   :alt: Python Version
.. |License| image:: https://img.shields.io/pypi/l/m2stitch
   :target: https://opensource.org/licenses/MIT
   :alt: License
.. |Read the Docs| image:: https://img.shields.io/readthedocs/m2stitch/latest.svg?label=Read%20the%20Docs
   :target: https://m2stitch.readthedocs.io/
   :alt: Read the documentation at https://m2stitch.readthedocs.io/
.. |Tests| image:: https://github.com/yfukai/m2stitch/workflows/Tests/badge.svg
   :target: https://github.com/yfukai/m2stitch/actions?workflow=Tests
   :alt: Tests
.. |Codecov| image:: https://codecov.io/gh/yfukai/m2stitch/branch/master/graph/badge.svg
   :target: https://codecov.io/gh/yfukai/m2stitch
   :alt: Codecov
.. |pre-commit| image:: https://img.shields.io/badge/pre--commit-enabled-brightgreen?logo=pre-commit&logoColor=white
   :target: https://github.com/pre-commit/pre-commit
   :alt: pre-commit
.. |Black| image:: https://img.shields.io/badge/code%20style-black-000000.svg
   :target: https://github.com/psf/black
   :alt: Black
.. |Zenodo| image:: https://zenodo.org/badge/DOI/10.5281/zenodo.5139597.svg
   :target: https://doi.org/10.5281/zenodo.5139597
   :alt: Zenodo

**Note** :memo: : one might also be interested in another Python-written stitching tool
ASHLAR_ (bioRxiv_),
with a comparable performance to that of MIST and additional features.

.. _ASHLAR: https://github.com/labsyspharm/ashlar
.. _bioRxiv: https://www.biorxiv.org/content/10.1101/2021.04.20.440625v1

Features
--------

- Provides robust stitching of tiled microscope images on a regular grid, mostly following algorithm by MIST_.
- Supports missing tiles.

Installation
------------

You can install *M2Stitch* via pip_ from PyPI_:

.. code:: console

   $ pip install m2stitch


Usage
-----

Please see the Usage_ for details.


Contributing
------------

Contributions are very welcome.
To learn more, see the `Contributor Guide`_.


License
-------

Distributed under the terms of the `MIT license`_,
*M2Stitch* is free and open source software.


Issues
------

If you encounter any problems,
please `file an issue`_ along with a detailed description.


Credits
-------

This program is an unofficial implementation of MIST_ stitching algorithm on GitHub_. The original paper is here_.

This project was generated from `@cjolowicz`_'s `Hypermodern Python Cookiecutter`_ template.

.. _MIST: https://pages.nist.gov/MIST
.. _GitHub: https://github.com/usnistgov/MIST
.. _here: https://github.com/USNISTGOV/MIST/wiki/assets/mist-algorithm-documentation.pdf
.. _@cjolowicz: https://github.com/cjolowicz
.. _Cookiecutter: https://github.com/audreyr/cookiecutter
.. _MIT license: https://opensource.org/licenses/MIT
.. _PyPI: https://pypi.org/
.. _Hypermodern Python Cookiecutter: https://github.com/cjolowicz/cookiecutter-hypermodern-python
.. _file an issue: https://github.com/yfukai/m2stitch/issues
.. _pip: https://pip.pypa.io/
.. github-only
.. _Contributor Guide: CONTRIBUTING.rst
.. _Usage: https://m2stitch.readthedocs.io/en/latest/usage.html
