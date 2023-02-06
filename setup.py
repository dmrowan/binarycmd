from setuptools import setup, find_packages
from distutils.util import convert_path

setup(name = "AstroCMD",
    version = 0.1,
    description = "Color-Magnitude Diagram Analysis"
    author = "Dom Rowan",
    author_email = "",
    url = "https://github.com/dmrowan/AstroCMD",
    packages = find_packages(include=['binarycmd', 'binarycmd.*']),
    package_data = {'binarycmd':['data/*.dat']},
    include_package_data = True,
    classifiers=[
      'Intended Audience :: Science/Research',
      'Operating System :: OS Independent',
      'Programming Language :: Python :: 3',
      'License :: OSI Approved :: MIT License',
      'Topic :: Scientific/Engineering :: Astronomy'
      ],
    zip_safe=False,
    python_requires=">=3.6",
    install_requires=["numpy", "scipy", "matplotlib", "pandas", "astropy"]
)
