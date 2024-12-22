from setuptools import setup, find_packages

setup(
    name='hot_rolling_fuzzy_package',
    version='0.1',
    author='Ronaldo Tsela',
    author_email='rontsela@mail.ntua.gr',
    description='Implementation of the Fuzzy Logic System for hot rolling manufacturing process control based on the paper: "Fuzzy control algorithm for the prediction of tension variations in hot rolling - Jong-Yeob Jung, Yong-Taek Im"',
    long_description=open('README.md').read(),
    long_description_content_type='text/markdown',
    packages=find_packages(),
    url='https://github.com/RonT23/CIM/hot-rolling-fuzzy/hot_rolling_fuzzy_package',

    classifiers=[
        'Programming Language :: Python :: 3',
        'License :: OSI Approved :: MIT License',
        'Operating System :: OS Independent',
    ],

    python_requires='>=3.5',
    
    install_requires=[  
        'numpy',
        'matplotlib'
    ],
)