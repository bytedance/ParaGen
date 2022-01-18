from setuptools import setup, find_packages

setup(
    name="paragen",
    version="0.1.0",
    keywords=["Natural Language Processing", "Machine Learning"],
    description="Bytedance Parallel Generation Toolkit",
    long_description="ParaGen is a PyTorch deep learning framework for parallel sequence generation."
                     "Apart from parallel sequence generation, ParaGen also enhances various NLP tasks, including "
                     "sequence-level classification, extraction and generation.",
    license="MIT Licence",
    author="Jiangtao Feng, Yi Zhou, Xian Qian, Liwei Wu, Jun Zhang, Yanming Liu, Zhexi Zhang, Mingxuan Wang, Hao Zhou",
    author_email="fengjiangtao@bytedance.com,"
                 "zhouyi.naive@bytedance.com,"
                 "qian.xian@bytedance.com"
                 "wuliwei.000@bytedance.com,",

    packages=find_packages(),
    include_package_data=True,
    platforms="any",
    install_requires=open("requirements.txt").readlines(),
    zip_safe=False,

    scripts=[],
    entry_points={
        'console_scripts': [
            'paragen-run = paragen.entries.run:main',
            'paragen-export = paragen.entries.export:main',
            'paragen-preprocess = paragen.entries.preprocess:main',
            'paragen-serve = paragen.entries.serve:main',
            'paragen-serve-model = paragen.entries.serve_model:main',
            'paragen-build-tokenizer = paragen.entries.build_tokenizer:main',
            'paragen-binarize-data = paragen.entries.binarize_data:main'
        ]
    }
)
