from setuptools import setup, Extension
import pybind11
from setuptools.command.build_ext import build_ext

class BuildExtWithGPP(build_ext):
    def build_extensions(self):
        compiler = self.compiler
        for ext in self.extensions:
            ext.extra_compile_args = ["-O2", "-D_GLIBCXX_USE_CXX11_ABI=0"]  # Use C++20 and ensure ABI compatibility
        compiler.compiler_so[0] = "g++"  # Set the compiler to g++
        super().build_extensions()


ext_modules = [
    Extension(
        "walks",
        ["walks.cpp"],
        include_dirs=[pybind11.get_include()],  # Include pybind11 headers
        #extra_compile_args=["-std=c++20"],  # Ensure C++20 support
        extra_link_args=["-O2", "-lstdc++", "-luuid"],  # Link against libstdc++
    )
]

setup(
    name="walks",
    version="0.2",
    ext_modules=ext_modules,
    cmdclass={'build_ext': BuildExtWithGPP},
)
