# C:\PycharmProjects\Peternity\modules_and_data\setup.py
from setuptools import setup

setup(
    name='peternity',
    version='0.1.0',    
    description='Peternity Python package',
    url='https://github.com/Fabbernat/Peternity',
    author='Bernát Fábián',
    author_email='fabbernat@gmail.com',
    license='BSD 2-clause',
    packages=['peternity'],
    install_requires=['mpi4py>=2.0',
                      'numpy',                     
                      ],

    classifiers=[
        'Development Status :: 1 - Planning',
        'Intended Audience :: Science/Research',
        'License :: OSI Approved :: BSD License',  
        'Operating System :: POSIX :: Linux',        
        'Programming Language :: Python :: 2',
        'Programming Language :: Python :: 2.7',
        'Programming Language :: Python :: 3',
        'Programming Language :: Python :: 3.4',
        'Programming Language :: Python :: 3.5',
    ],
)