from setuptools import setup

setup(
    name='speciesselector',
    version='0.1.0',
    py_modules=['speciesselector', 'speciesselector.database'],
    install_requires=[
        'Click',
    ],
    entry_points={
        'console_scripts': [
            'spselec = speciesselector.cli:cli',
        ],
    },
)
