# libpolicyts

A C++23 library for policy tree search algorithms and auxiliary utilities.

## Implemented Algorithms
The following algorithms are implemented in `include/libpolicyts/algorithm/`, 
which support both user-defined policy/heuristics and neural network defined versions.
- __Best First Search__: A general search algorithm with controlled weights on the g-cost and h-cost
- __LubyTS__: Orseau, Laurent, et al. "Single-agent policy tree search with guarantees." Advances in Neural Information Processing Systems 31 (2018).
- __MultiTS__: Orseau, Laurent, et al. "Single-agent policy tree search with guarantees." Advances in Neural Information Processing Systems 31 (2018).
- __LevinTS__: Orseau, Laurent, et al. "Single-agent policy tree search with guarantees." Advances in Neural Information Processing Systems 31 (2018).
- __PHS*__: Orseau, Laurent, and Levi HS Lelis. "Policy-guided heuristic search with guarantees." Proceedings of the AAAI Conference on Artificial Intelligence. Vol. 35. No. 14. 2021.


## Implemented Environments
The following environment wrappers are implemented in `include/libpolicyts/env/`, 
which support all of the concepts required for all implemented algorithms.
This can be enabled by setting the `LIBPOLICYTS_BUILD_ENVIRONMENTS` CMake flag.
- [boulderdash](https://github.com/tuero/boulderdash_cpp)
- [craftworld](https://github.com/tuero/craftworld_cpp_v2)
- [sokoban](https://github.com/tuero/sokoban_cpp)
- [tsp](https://github.com/tuero/tsp_cpp)

## Implemented Neural Models
The following environment wrappers are implemented in `include/libpolicyts/model/`, 
which support all of the concepts required for all implemented algorithms.
This can be enabled by setting the `LIBPOLICYTS_BUILD_TORCH` CMake flag, with libtorch in your path through the environment variable `LIBTORCH_ROOT`
(see the section below about building)
- `HeuristicConvNetWrapper`: A ResNet style heuristic net
- `PolicyConvNetWrapper`: A ResNet style policy net
- `TwoHeadedConvNetWrapper`: A ResNet style two headed policy + heuristic net

## Auxiliary Utilities
- Neural Network policy and heuristic functions through `libtorch` which are compatible with all implemented algorithms, which can be found in `include/libpolicyts/model/`
- Training and Testing bootstrap functions which include threaded runners with neural network support


## CMake Flags
The following CMake flags are supported:
- `LIBPOLICYTS_BUILD_ENVIRONMENTS`: Set to `ON` to build the environment wrappers
- `LIBPOLICYTS_BUILD_EXAMPLES`: Set to `ON` to build the examples (implicitly enables `LIBPOLICYTS_BUILD_ENVIRONMENTS`)
- `LIBPOLICYTS_BUILD_TORCH`: Set to `ON` to build neural models backed my libtorch


## Examples
- Simple examples using both custom and libtorch neural moddesl are given in `examples/`.
- TODO: Add more examples using the neural models + training/testing functionality


## Include to Your Project: CMake

### VCPKG
`libpolicyts` is not part of the official registry for vcpkg,
but is supported in my personal registry [here](https://github.com/tuero/vcpkg-registry).
This is by far the easier way to use this library as it will pull in dependencies, and is really the only documented way.
To add `tuero/vcpkg-registry` as a git registry to your vcpkg project:
```json
"registries": [
...
{
    "kind": "git",
    "repository": "https://github.com/tuero/vcpkg-registry",
    "reference": "master",
    "baseline": "<COMMIT_SHA>",
    "packages": ["libpolicyts", "boulderdash", "craftworld", "sokoban", "tsp"]
}
]
...
```
where `<COMMIT_SHA>` is the 40-character git commit sha in the registry's repository (you can find 
this by clicking on the latest commit [here](https://github.com/tuero/vcpkg-registry) and looking 
at the URL.

To enable the Libtorch utilities, you can use the `torch` feature option when adding the library to vcpkg.
To enable the provided environment wrappers, you can use the `environment` feature option when adding the library to vcpkg.
```shell
vcpkg add port libpolicyts
vcpkg add port libpolicyts[torch,environment]
```

Note that `torch` will look for the `libtorch` project in the environment variable `LIBTORCH_ROOT`, which is not part of the included dependencies.
The easiest way to get `libtorch` is through the python package.
First, create a virtual environment and install pytorch:
```shell
conda create -n <MY_ENV> python=3.12
conda activate <MY_ENV>
pip3 install torch torchvision
```

Then set the `LIBTORCH_ROOT` environment variable:
```shell
export LIBTORCH_ROOT=`python3 -c 'import torch;print(torch.utils.cmake_prefix_path)'`
```

Then in your project cmake:
```cmake
cmake_minimum_required(VERSION 3.25)
project(my_project LANGUAGES CXX)

find_package(libpolicyts CONFIG REQUIRED)
add_executable(main main.cpp)
target_link_libraries(main PRIVATE libpolicyts::libpolicyts)
```


## Building Examples
The `CMakePrests.json` defines build options for the examples and libtorch neural models.
You will first need to set the following environment variables for your `CC`, `CC`, and `FC` compiler.
You will also need to set `LIBTORCH_ROOT` for the libtorch path.
The easiest way for this is to install as a python package, then locate.

For example:
```shell
export CC=gcc-15.2
export CXX=g++-15.2
export FC=gfortrain-15.2
export LIBTORCH_ROOT=`python3 -c 'import torch;print(torch.utils.cmake_prefix_path)'`
```

Using the supplied `CMakePresets.json`:
```shell
cmake --preset=release
cmake --build --preset=release -- -j8
```
