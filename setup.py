from setuptools import setup, find_packages
setup(name='EMspecPy',
      version='0.3.1',
      zip_safe=False,
      packages=find_packages(),
      include_package_data=False,
      install_requires=['numpy','scipy','pandas','matplotlib'],
      )
