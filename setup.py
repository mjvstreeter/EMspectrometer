from setuptools import setup, find_packages
import os
import EMspecPy


# print(data_files)
setup(name='EMspecPy',
      version='0.2.1',
      zip_safe=False,
      packages=find_packages(),
      include_package_data=False,
      install_requires=['opencv-python','scipy','scikit-learn',
                        'scikit-image','scikit-optimize','Pillow'],
      )
