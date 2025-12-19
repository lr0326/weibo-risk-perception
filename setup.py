"""
安装配置文件
"""

from setuptools import setup, find_packages

with open("README.md", "r", encoding="utf-8") as fh:
    long_description = fh.read()

with open("requirements.txt", "r", encoding="utf-8") as fh:
    requirements = [line.strip() for line in fh if line.strip() and not line.startswith("#")]

setup(
    name="weibo-risk-perception",
    version="1.0.0",
    author="lr0326",
    author_email="your.email@example.com",
    description="基于微博数据的社会风险感知与舆情预测系统",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/yourusername/weibo-risk-perception",
    packages=find_packages(),
    classifiers=[
        "Development Status :: 4 - Beta",
        "Intended Audience :: Developers",
        "Intended Audience :: Science/Research",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.8",
        "Programming Language :: Python :: 3.9",
        "Programming Language :: Python :: 3.10",
        "Programming Language :: Python :: 3.11",
        "Topic :: Scientific/Engineering :: Artificial Intelligence",
        "Topic :: Text Processing :: Linguistic",
    ],
    python_requires=">=3.8",
    install_requires=requirements,
    extras_require={
        "dev": [
            "pytest>=7.2.0",
            "pytest-cov>=4.0.0",
            "black>=23.1.0",
            "flake8>=6.0.0",
            "isort>=5.12.0",
        ],
        "gpu": [
            "torch>=2.0.0",
        ],
    },
    entry_points={
        "console_scripts": [
            "weibo-risk=main:main",
        ],
    },
    include_package_data=True,
    package_data={
        "": ["*.yaml", "*.yml", "*.txt"],
    },
)

