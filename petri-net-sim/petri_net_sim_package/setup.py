from setuptools import setup, find_packages

setup(
    name='petri_net_sim_package',
    version='0.1',
    author='Ronaldo Tsela',
    author_email='ron-tsela@di.uoa.gr',
    description='A simple Petri network simulator with minimal graphical representation.',
    long_description=open('README.md').read(),
    long_description_content_type='text/markdown',
    packages=find_packages(),
    url='https://github.com/RonT23/CIM/petri_net_sim_package',
    classifiers=[
        'Programming Language :: Python :: 3',
        'License :: OSI Approved :: MIT License',
        'Operating System :: OS Independent',
    ],
    python_requires='>=3.6',
    install_requires=[ 
        'os', 'random', 'json', 'networkx', 'matplotlib'
    ],
)