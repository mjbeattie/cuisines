from setuptools import setup, find_packages
setup(
        name='cuisines',
        version='1.0',
        author='Matthew Beattie',
        author_email='mjbeattie@ou.edu',
        packages=find_packages(exclude=('tests','docs')),
        setup_requires=['pytest-runner'],
        tests_require=['pytest']
        )


