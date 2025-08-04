from setuptools import find_packages, setup
from glob import glob
import os

package_name = 'spca_llm_ur5'

data_files=[
    ('share/ament_index/resource_index/packages',
        ['resource/' + package_name]),
    ('share/' + package_name, ['package.xml']),
    ('share/' + package_name + '/launch', ['launch/spca_bringup.launch.py']),
    ('share/' + package_name + '/config',
        ['config/curriculum.yaml'] +
        glob('config/levels/**/*.yaml', recursive=True)),
]
for dirpath, _, filenames in os.walk('config'):
    yaml_files = [os.path.join(dirpath, f)
                  for f in filenames if f.endswith('.yaml')]
    if yaml_files:
        install_dir = os.path.join('share', package_name, dirpath)
        data_files.append((install_dir, yaml_files))

setup(
    name=package_name,
    version='0.0.1',
    packages=find_packages(exclude=['test']),
    data_files=data_files,
    install_requires=[
        # LLM + parsing stack
        "openai>=1.84.0",
        "pydantic>=2.9.2",
        "python-dotenv>=1.0.1",

        # Classical-planning stack
        "unified-planning[fast-downward,tamer]>=1.2.0",
    ],
    zip_safe=True,
    maintainer='drew99',
    maintainer_email='drejcpesjak.pesjak@gmail.com',
    description='TODO: Package description',
    license='TODO: License declaration',
    tests_require=['pytest'],
    entry_points={
        'console_scripts': [
            'executor_node = spca_llm_ur5.nodes.executor_node:main',
            'referee_node = spca_llm_ur5.nodes.referee_node:main',
            'supervisor_node = spca_llm_ur5.nodes.supervisor_node:main',
        ],
    },
)
