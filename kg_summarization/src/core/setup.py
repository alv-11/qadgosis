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
        "sankofa",  # Ensure the module name matches the one defined in `PYBIND11_MODULE`
        ["sankofa.cpp"],  # Include only the file with the `PYBIND11_MODULE` definition
        include_dirs=[pybind11.get_include()],
        extra_link_args=["-O2", "-lstdc++", "-luuid"]
    )
]

setup(
    name="sankofa", 
    version="0.1",
    ext_modules=ext_modules,
    cmdclass={'build_ext': BuildExtWithGPP},
)

"""for MACOS users uncomment this and comment the above code"""
# from setuptools import setup, Extension
# import pybind11

# ext_modules = [
#     Extension(
#         "sankofa",  # Ensure the module name matches the one defined in `PYBIND11_MODULE`
#         ["sankofa.cpp"],  # Include only the file with the `PYBIND11_MODULE` definition
#         include_dirs=[pybind11.get_include()],
#         language="c++",
#         extra_compile_args=["-std=c++20"]
#     )
# ]

# setup(
#     name="sankofa", 
#     version="0.1",
#     ext_modules=ext_modules,
# )