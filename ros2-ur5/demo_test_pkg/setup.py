from setuptools import find_packages, setup
package_name = "demo_test_pkg"

setup(
    name=package_name,
    version="0.0.0",
    packages=find_packages(exclude=["test"]),
    data_files=[
        ("share/ament_index/resource_index/packages", ["resource/" + package_name]),
        ("share/" + package_name,               ["package.xml"]),
        ("share/" + package_name + "/launch",   ["launch/ur5_simple_demo.launch.py"]),
    ],
    install_requires=["setuptools"],
    zip_safe=True,
    maintainer="drew99",
    maintainer_email="drejcpesjak.pesjak@gmail.com",
    description="Simple MoveItPy demo for UR5",
    license="TODO: License declaration",
    tests_require=["pytest"],
    entry_points={
        "console_scripts": [
            "cart_runner_node   = demo_test_pkg.cart_runner_node:main",
            "moveit_test_node   = demo_test_pkg.moveit_test_node:main",
            "ur5_simple_demo_node = demo_test_pkg.ur5_simple_demo_node:main",
            "moveit_multitest_node = demo_test_pkg.moveit_multitest_node:main",
            "moveit_posetest_node = demo_test_pkg.moveit_posetest_node:main",
            "image_proc_node    = demo_test_pkg.image_proc_node:main",
            "img_depth_node     = demo_test_pkg.img_depth_node:main",
            "closest_point_node = demo_test_pkg.closest_point_node:main",
        ],
    },
)
