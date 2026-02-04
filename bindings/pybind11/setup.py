import os
from pathlib import Path

from pybind11.setup_helpers import Pybind11Extension, build_ext
from setuptools import setup

ROOT = Path(__file__).resolve().parent.parent.parent
STOCKFISH_SRC = ROOT / "Stockfish_hybrid" / "src"

ext_modules = [
    Pybind11Extension(
        "stockfish_hybrid_binding",
        ["stockfish_hybrid_binding.cpp"],
        include_dirs=[str(STOCKFISH_SRC)],
        library_dirs=[str(STOCKFISH_SRC)],
        libraries=["stockfish"],
        extra_link_args=[f"-Wl,-rpath,{STOCKFISH_SRC}"],
        language="c++",
    )
]

setup(
    name="stockfish_hybrid_binding",
    version="0.0.1",
    description="Pybind11 bindings for the hybrid Stockfish fork (evaluation API)",
    ext_modules=ext_modules,
    cmdclass={"build_ext": build_ext},
)
