import setuptools

with open( 'README.md', 'r' ) as f:
    long_desc = f.read()

# get __version__
exec( open( 'bric_analysis_libraries/_version.py' ).read() )

setuptools.setup(
    name='bric-analysis-libraries',
    version = __version__,
    author='Brian Carlsen',
    author_email = 'carlsen.bri@gmail.com',
    description = 'An assortment of analysis libraries.',
    long_description = long_desc,
    long_description_content_type = 'text/markdown',
    url = 'https://github.com/bicarlsen/analysis-libraries.git',
    packages = setuptools.find_packages(),
    classifiers = [
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
    ],
    install_requires = [
        'numpy',
        'pandas',
        'matplotlib',
        'scipy',
        'h5py'
    ],
    package_data = {
        'pl': [ 'data/*' ]
    },

)
