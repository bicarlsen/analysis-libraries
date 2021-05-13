import setuptools

with open( 'README.md', 'r' ) as f:
    long_desc = f.read()

setuptools.setup(
    name='bric-analysis-libraries',
    version = '0.0.8',
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
    package_data = {
        'pl': [ 'data/*' ]
    }
)
