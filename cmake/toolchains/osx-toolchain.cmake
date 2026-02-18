# Uses environment variables for the following compilers
# - CC -> C compiler
# - CXX -> C++ compiler
# - FC -> Fortrain compiler
# We also disable cross compiling
if(NOT _VCPKG_OSX_TOOLCHAIN)
    set(_VCPKG_OSX_TOOLCHAIN 1)

    set(CMAKE_C_COMPILER "$ENV{CC}")
    set(CMAKE_CXX_COMPILER "$ENV{CXX}")
    set(CMAKE_Fortran_COMPILER "$ENV{FC}")

    set(CMAKE_SYSTEM_NAME Darwin CACHE STRING "")
    set(CMAKE_MACOSX_RPATH ON CACHE BOOL "")
    set(CMAKE_CROSSCOMPILING OFF CACHE BOOL "")

    # --- macOS + Homebrew GCC(libstdc++) workaround --------------------------
    # libstdc++ expects quick_exit/at_quick_exit to be visible in the SDK headers.
    # On macOS, that can require _DARWIN_C_SOURCE.
    #
    # Only apply for GCC/g++ (not clang) to avoid surprising other toolchains.
    get_filename_component(_cxx_name "${CMAKE_CXX_COMPILER}" NAME)
    set(_using_gcc FALSE)
    if(_cxx_name MATCHES "^g\\+\\+([0-9]+)?(-[0-9]+(\\.[0-9]+)*)?$")
        set(_using_gcc TRUE)
    endif()

    if(_using_gcc)
        if(NOT DEFINED _VCPKG_OSX_ADDED_DARWIN_C_SOURCE)
            set(_VCPKG_OSX_ADDED_DARWIN_C_SOURCE 1 CACHE INTERNAL "")
            string(APPEND CMAKE_C_FLAGS_INIT   " -D_DARWIN_C_SOURCE")
            string(APPEND CMAKE_CXX_FLAGS_INIT " -D_DARWIN_C_SOURCE")
        endif()
    endif()

    if(POLICY CMP0056)
        cmake_policy(SET CMP0056 NEW)
    endif()
    if(POLICY CMP0066)
        cmake_policy(SET CMP0066 NEW)
    endif()
    if(POLICY CMP0067)
        cmake_policy(SET CMP0067 NEW)
    endif()
    if(POLICY CMP0137)
        cmake_policy(SET CMP0137 NEW)
    endif()

    list(APPEND CMAKE_TRY_COMPILE_PLATFORM_VARIABLES
        VCPKG_CRT_LINKAGE VCPKG_TARGET_ARCHITECTURE
        VCPKG_C_FLAGS VCPKG_CXX_FLAGS
        VCPKG_C_FLAGS_DEBUG VCPKG_CXX_FLAGS_DEBUG
        VCPKG_C_FLAGS_RELEASE VCPKG_CXX_FLAGS_RELEASE
        VCPKG_LINKER_FLAGS VCPKG_LINKER_FLAGS_RELEASE VCPKG_LINKER_FLAGS_DEBUG
    )

    if(NOT DEFINED CMAKE_SYSTEM_PROCESSOR)
        if(VCPKG_TARGET_ARCHITECTURE STREQUAL "x64")
           set(CMAKE_SYSTEM_PROCESSOR x86_64 CACHE STRING "")
        elseif(VCPKG_TARGET_ARCHITECTURE STREQUAL "x86")
           set(CMAKE_SYSTEM_PROCESSOR x86 CACHE STRING "")
        elseif(VCPKG_TARGET_ARCHITECTURE STREQUAL "arm64")
           set(CMAKE_SYSTEM_PROCESSOR arm64 CACHE STRING "")
        else()
           set(CMAKE_SYSTEM_PROCESSOR "${CMAKE_HOST_SYSTEM_PROCESSOR}" CACHE STRING "")
        endif()
    endif()

    if(DEFINED VCPKG_CMAKE_SYSTEM_VERSION)
        set(CMAKE_SYSTEM_VERSION "${VCPKG_CMAKE_SYSTEM_VERSION}" CACHE STRING "" FORCE)
    endif()

    if(CMAKE_HOST_SYSTEM_NAME STREQUAL "Darwin")
        if(NOT DEFINED CMAKE_SYSTEM_VERSION)
            set(CMAKE_SYSTEM_VERSION "${CMAKE_HOST_SYSTEM_VERSION}" CACHE STRING "")
        endif()
    endif()

    string(APPEND CMAKE_C_FLAGS_INIT " -fPIC ${VCPKG_C_FLAGS} ")
    string(APPEND CMAKE_CXX_FLAGS_INIT " -fPIC ${VCPKG_CXX_FLAGS} ")
    string(APPEND CMAKE_C_FLAGS_DEBUG_INIT " ${VCPKG_C_FLAGS_DEBUG} ")
    string(APPEND CMAKE_CXX_FLAGS_DEBUG_INIT " ${VCPKG_CXX_FLAGS_DEBUG} ")
    string(APPEND CMAKE_C_FLAGS_RELEASE_INIT " ${VCPKG_C_FLAGS_RELEASE} ")
    string(APPEND CMAKE_CXX_FLAGS_RELEASE_INIT " ${VCPKG_CXX_FLAGS_RELEASE} ")

    string(APPEND CMAKE_MODULE_LINKER_FLAGS_INIT " ${VCPKG_LINKER_FLAGS} ")
    string(APPEND CMAKE_SHARED_LINKER_FLAGS_INIT " ${VCPKG_LINKER_FLAGS} ")
    string(APPEND CMAKE_EXE_LINKER_FLAGS_INIT " ${VCPKG_LINKER_FLAGS} ")
    string(APPEND CMAKE_MODULE_LINKER_FLAGS_DEBUG_INIT " ${VCPKG_LINKER_FLAGS_DEBUG} ")
    string(APPEND CMAKE_SHARED_LINKER_FLAGS_DEBUG_INIT " ${VCPKG_LINKER_FLAGS_DEBUG} ")
    string(APPEND CMAKE_EXE_LINKER_FLAGS_DEBUG_INIT " ${VCPKG_LINKER_FLAGS_DEBUG} ")
    string(APPEND CMAKE_MODULE_LINKER_FLAGS_RELEASE_INIT " ${VCPKG_LINKER_FLAGS_RELEASE} ")
    string(APPEND CMAKE_SHARED_LINKER_FLAGS_RELEASE_INIT " ${VCPKG_LINKER_FLAGS_RELEASE} ")
    string(APPEND CMAKE_EXE_LINKER_FLAGS_RELEASE_INIT " ${VCPKG_LINKER_FLAGS_RELEASE} ")
endif()

