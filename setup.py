from setuptools import setup

setup(name='seqstein',
      version='1.0.1',
      description='A Sequential Stein\'s Method for Faster Training of Additive Index Models',
      url='https://github.com/BobZhangHT/SeqStein',
      author='Hengtao Zhang and Zebin Yang',
      author_email='zhanght@connect.hku.hk, yangzb2010@hku.hk',
      license='GPL',
      packages=['seqstein'],
      install_requires=['pandas',
                        'rpy2',
                        'matplotlib', 
                        'numpy', 
                        'sklearn', 
                        'pygam'],
      zip_safe=False)