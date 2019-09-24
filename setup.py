import setuptools
# Reminder: build with python setup.py bdist_wheel, deploy with twine upload dist/*

with open("README.md", "r") as file:
    long_description = file.read()

setuptools.setup(
    name='pygrank',
    version='0.1.2',
    author="Emmanouil Krasanakis",
    author_email="maniospas@hotmail.com",
    description="Recommendation algorithms for large graphs",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/maniospas/pygrank",
    packages=setuptools.find_packages(),
    classifiers=[
         "Programming Language :: Python :: 3",
         "License :: OSI Approved :: MIT License",
         "Operating System :: OS Independent",
     ],
    install_requires=[
          'sklearn', 'scipy', 'numpy', 'networkx',
      ],
 )