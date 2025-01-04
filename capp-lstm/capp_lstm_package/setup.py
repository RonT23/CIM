from setuptools import setup, find_packages

setup(
    name='hot_rolling_fuzzy_package',
    version='0.1',
    author='Ronaldo Tsela',
    author_email='rontsela@mail.ntua.gr',
    description='CAPP System developed using LSTM ANN for generating process chains. The package provides both training and inference API/ functionalities.',
    long_description=open('README.md').read(),
    long_description_content_type='text/markdown',
    packages=find_packages(),
    url='https://github.com/RonT23/CIM/capp-lstm/capp_lstm_package',

    classifiers=[
        'Programming Language :: Python :: 3',
        'License :: OSI Approved :: MIT License',
        'Operating System :: OS Independent',
    ],

    python_requires='>=3.5',
    
    install_requires=[  
        'numpy',
        'matplotlib',
        # Tensorflow.Keras
        # pandas
        #  ...
    ],
)