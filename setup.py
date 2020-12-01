from setuptools import setup, find_packages
#print(find_packages())
#required_packages = []
#with open('requirements.txt') as f:
#    required_packages = f.read().splitlines()
#required_packages += ['corner.py @ https://github.com/jiwoncpark/corner.py/archive/master.zip']
#print(required_packages)

setup(
      name='h0rton',
      version='v1.0',
      author='Ji Won Park',
      author_email='jiwon.christine.park@gmail.com',
      packages=find_packages(),
      license='LICENSE.md',
      description='Bayesian neural network for hierarchical inference of the Hubble constant',
      long_description=open("README.rst").read(),
      long_description_content_type='text/x-rst',
      url='https://github.com/jiwoncpark/h0rton',
      #install_requires=required_packages,
      #dependency_links=['http://github.com/jiwoncpark/corner.py/tarball/master#egg=corner_jiwoncpark'],
      include_package_data=True,
      entry_points={
      'console_scripts': ['train=h0rton.train:main', 'infer_h0=h0rton.infer_h0_mcmc_default:main'],
      },
      test_suite='nose.collector',
      tests_require=['nose'],
      classifiers=['Development Status :: 4 - Beta',
      'License :: OSI Approved :: BSD License',
      'Intended Audience :: Developers',
      'Intended Audience :: Science/Research',
      'Operating System :: OS Independent',
      'Programming Language :: Python'],
      keywords='physics'
      )
