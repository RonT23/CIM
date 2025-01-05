from setuptools import setup, find_packages

setup(
    name='capp_lstm_package',
    version='0.1.0',
    author='Ronaldo Tsela',
    author_email='rontsela@mail.ntua.gr',
    description='CAPP System using LSTM-based ANN for generating process chains. Provides training and inference APIs.',
    long_description=open('README.md').read(),
    long_description_content_type='text/markdown',
    packages=find_packages(),
    url='https://github.com/RonT23/CIM/capp-lstm',
    
    classifiers=[
        'Programming Language :: Python :: 3',
        'License :: OSI Approved :: MIT License',
        'Operating System :: OS Independent',
        'Intended Audience :: Developers',
        'Topic :: Scientific/Engineering :: Artificial Intelligence',
        'Topic :: Scientific/Engineering :: Manufacturing',
    ],
    
    # Minimum Python version updated for TensorFlow compatibility
    python_requires='>=3.7',  

    install_requires=[  
        'numpy>=1.19.5',
        'matplotlib>=3.2.2',
        'tensorflow>=2.6.0',
        'pandas>=1.1.5',
        'scikit-learn>=0.24.2',
        'joblib>=1.0.0'
    ],

    include_package_data=True, 
)
