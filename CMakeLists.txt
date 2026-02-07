cmake_minimum_required(VERSION 3.25)

project(libpolicyts VERSION 1.1.4 LANGUAGES CXX)


# Build options
option(LIBPOLICYTS_BUILD_ENVIRONMENTS "Build libpolicyts environments" OFF)
option(LIBPOLICYTS_BUILD_EXAMPLES "Build libpolicyts examples" OFF)
option(LIBPOLICYTS_BUILD_TORCH "Build libpolicyts with libtorch models" OFF)

# Let libpolicyts_SHARED_LIBS override BUILD_SHARED_LIBS
if(DEFINED libpolicyts_SHARED_LIBS)
    set(BUILD_SHARED_LIBS "${libpolicyts_SHARED_LIBS}")
endif()


# CPP library
add_library(libpolicyts)
add_library(libpolicyts::libpolicyts ALIAS libpolicyts)
set_target_properties(libpolicyts PROPERTIES
                      VERSION ${libpolicyts_VERSION}
                      SOVERSION ${libpolicyts_VERSION_MAJOR}
                      POSITION_INDEPENDENT_CODE ON
)
target_include_directories(
    libpolicyts PUBLIC "$<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include>")
target_compile_features(libpolicyts PUBLIC cxx_std_23)


# Include the install rules if the user wanted them (included by default when top-level)
string(COMPARE EQUAL "${CMAKE_SOURCE_DIR}" "${CMAKE_CURRENT_SOURCE_DIR}" is_top_level)
option(libpolicyts_INCLUDE_PACKAGING "Include packaging rules for libpolicyts" "${is_top_level}")
if(libpolicyts_INCLUDE_PACKAGING)
    add_subdirectory(packaging)
endif()

# Public headers
set(LIBPOLICYTS_PUBLIC_HEADERS
    include/libpolicyts/algorithm/bfs.h
    include/libpolicyts/algorithm/levints.h
    include/libpolicyts/algorithm/lubyts.h
    include/libpolicyts/algorithm/multits.h
    include/libpolicyts/algorithm/phs.h
    include/libpolicyts/algorithm/detail/luby_impl.h
    include/libpolicyts/algorithm/detail/phs_impl.h
    include/libpolicyts/block_allocator.h
    include/libpolicyts/cluster_graph.h
    include/libpolicyts/concepts.h
    include/libpolicyts/libpolicyts.h
    include/libpolicyts/logging.h
    include/libpolicyts/math_util.h
    include/libpolicyts/metrics_tracker.h
    include/libpolicyts/observation.h
    include/libpolicyts/resource.h
    include/libpolicyts/signaller.h
    include/libpolicyts/static_string.h
    include/libpolicyts/stop_token.h
    include/libpolicyts/test_runner.h
    include/libpolicyts/thread_pool.h
    include/libpolicyts/timer.h
    include/libpolicyts/train_bootstrap.h
    # Environment headers
    include/libpolicyts/env/boulderdash.h
    include/libpolicyts/env/craftworld.h
    include/libpolicyts/env/env_loader.h
    include/libpolicyts/env/sokoban.h
    include/libpolicyts/env/tsp.h
    # Headers which depend on libtorch
    include/libpolicyts/model/detail/heuristic_convnet.h
    include/libpolicyts/model/detail/layers.h
    include/libpolicyts/model/detail/policy_convnet.h
    include/libpolicyts/model/detail/twoheaded_convnet.h
    include/libpolicyts/model/base_model_wrapper.h
    include/libpolicyts/model/heuristic_convnet_wrapper.h
    include/libpolicyts/model/policy_convnet_wrapper.h
    include/libpolicyts/model/torch_init.h
    include/libpolicyts/model/torch_util.h
    include/libpolicyts/model/twoheaded_convnet_wrapper.h
)
# Suppress preprocessor #warning diagnostics for this one TU
set_source_files_properties(src/cluster_graph.cpp APPEND PROPERTIES
  COMPILE_OPTIONS "-Wno-cpp"
)


# External libs from 
find_package(absl CONFIG REQUIRED)
find_package(spdlog CONFIG REQUIRED)
find_package(nlohmann_json CONFIG REQUIRED)
find_package(igraph CONFIG REQUIRED)
find_package(libleidenalg CONFIG REQUIRED)


# Link external libs
target_link_libraries(libpolicyts PUBLIC spdlog::spdlog_header_only)
target_link_libraries(libpolicyts PUBLIC 
    absl::base 
    absl::flat_hash_map 
    absl::flat_hash_set 
    absl::hash
    absl::strings 
    absl::flags
    absl::flags_parse
)
target_link_libraries(libpolicyts PUBLIC nlohmann_json::nlohmann_json)
target_link_libraries(libpolicyts PUBLIC igraph::igraph)
target_link_libraries(libpolicyts PUBLIC libleidenalg::libleidenalg)


if (LIBPOLICYTS_BUILD_EXAMPLES)
    message("LIBPOLICYTS_BUILD_EXAMPLES=ON requires environments to be built. Setting LIBPOLICYTS_BUILD_ENVIRONMENTS=ON")
    set(LIBPOLICYTS_BUILD_ENVIRONMENTS ON)
endif()

# External libs for environments
set(LIBPOLICYTS_BUILD_ENVIRONMENT_SOURCES OFF)
if(LIBPOLICYTS_BUILD_ENVIRONMENTS)
    find_package(boulderdash CONFIG QUIET)
    if(NOT ${boulderdash_FOUND})
        message(FATAL_ERROR "LIBPOLICYTS_BUILD_ENVIRONMENTS=ON but boulderdash not found.")
    endif()
    find_package(craftworld CONFIG QUIET)
    if(NOT ${craftworld_FOUND})
        message(FATAL_ERROR "LIBPOLICYTS_BUILD_ENVIRONMENTS=ON but craftworld not found.")
    endif()
    find_package(sokoban CONFIG QUIET)
    if(NOT ${sokoban_FOUND})
        message(FATAL_ERROR "LIBPOLICYTS_BUILD_ENVIRONMENTS=ON but sokoban not found.")
    endif()
    find_package(tsp CONFIG QUIET)
    if(NOT ${tsp_FOUND})
        message(FATAL_ERROR "LIBPOLICYTS_BUILD_ENVIRONMENTS=ON but tsp not found.")
    endif()

    message("Found environment dependencies, building environment wrappers")
    set(LIBPOLICYTS_BUILD_ENVIRONMENT_SOURCES ON)
    # Macro definition so source files can conditionally set interact with torch-dep files
    target_compile_definitions(libpolicyts PUBLIC LIBPTS_ENVS_FOUND)

    target_link_libraries(libpolicyts PUBLIC boulderdash::boulderdash)
    target_link_libraries(libpolicyts PUBLIC craftworld::craftworld)
    target_link_libraries(libpolicyts PUBLIC sokoban::sokoban)
    target_link_libraries(libpolicyts PUBLIC tsp::tsp)
else()
    message(WARNING "LIBPOLICYTS_BUILD_ENVIRONMENTS=OFF, skipping building environment wrappers.")
endif()


# Torch is an optional dependency
# If its found, then we will also build the included model wrappers
set(LIBPOLICYTS_BUILD_TORCH_SOURCES OFF)
if(LIBPOLICYTS_BUILD_TORCH)
    find_package(Torch CONFIG QUIET)
    if(${TORCH_FOUND})
        message("Found Torch, building Torch model wrappers")
        set(LIBPOLICYTS_BUILD_TORCH_SOURCES ON)
        set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} ${TORCH_CXX_FLAGS}")
        # Macro definition so source files can conditionally set interact with torch-dep files
        target_compile_definitions(libpolicyts PUBLIC LIBPTS_TORCH_FOUND)
        # Add libtorch dependent headers
        set(LIBPOLICYTS_PUBLIC_HEADERS ${LIBPOLICYTS_PUBLIC_HEADERS})
        # Link to torch
        target_link_libraries(libpolicyts PUBLIC ${TORCH_LIBRARIES})
    else()
        message(FATAL_ERROR "LIBPOLICYTS_BUILD_TORCH=ON but libtorch was not found. Please see the readme for how to set the libtorch path.")
    endif()
else()
    message(WARNING "LIBPOLICYTS_BUILD_TORCH=OFF, skipping building Torch model wrappers. If you want this support, set the LIBTORCH_ROOT environment variable.")
endif()



# Add headers to source listing
target_sources(libpolicyts PRIVATE ${LIBPOLICYTS_PUBLIC_HEADERS})

# Library source
add_subdirectory(src)


# Build examples
if (LIBPOLICYTS_BUILD_EXAMPLES)
    message("LIBPOLICYTS_BUILD_EXAMPLES=ON, building examples")
    add_subdirectory(examples)
else()
    message("LIBPOLICYTS_BUILD_EXAMPLES=OFF, skipping building examples")
endif()
