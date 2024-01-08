import os
from setuptools import setup, find_packages

def src(pth):
    return os.path.join(os.path.dirname(__file__), pth)

# Project description
descr = 'Arid Multi Component Interpolation'

setup(
    name="AridMCSlopes",
    description=descr,
    long_description=open(src('README.md')).read(),
    keywords=['inverse problems',
              'optimization',
              'seismic demultiple'],
    classifiers=[
        'Intended Audience :: Science/Research',
        'Natural Language :: English',
        'Programming Language :: Python :: 3.9',
        'Topic :: Scientific/Engineering :: Mathematics'
    ],
    author='Muhammad Iqbal Khatami',
    author_email='muhammadiqbal.khatami@kaust.edu.sa',
    install_requires=['numpy>=1.18.4',
                      'torch>=2.0.1',
                      'pylops==1.18.3',
                      'devito>=4.7.1',
                      'cupy>=8.3.0'],
    packages=find_packages(),
    use_scm_version=dict(root='.',
                         relative_to=__file__,
                         write_to=src('AridMCSlopes/version.py')),
    setup_requires=['setuptools_scm'],

)
