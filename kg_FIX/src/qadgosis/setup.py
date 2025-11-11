from setuptools import setup, Extension
import pybind11

ext_modules = [
    Extension(
        "main",  # Ensure the module name matches the one defined in `PYBIND11_MODULE`
        ["main.cpp"],  # Include only the file with the `PYBIND11_MODULE` definition
        include_dirs=[pybind11.get_include()],
        language="c++",
        extra_compile_args=["-std=c++20"]
    )
]

setup(
    name="main", 
    version="0.1",
    ext_modules=ext_modules,
)