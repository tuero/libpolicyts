target_include_directories(libpolicyts PRIVATE ${PROJECT_SOURCE_DIR}/src)

target_sources(libpolicyts
    PRIVATE 
        logging.cpp
        math_util.cpp
        metrics_tracker.cpp
        resource.cpp
        signaller.cpp
        timer.cpp
)

# Conditionally add libtorch dependent source files
if(LIBPOLICYTS_BUILD_TORCH_SOURCES)
target_sources(libpolicyts
    PRIVATE 
        model/base_model_wrapper.cpp
        model/heuristic_convnet.cpp
        model/heuristic_convnet_wrapper.cpp
        model/layers.cpp
        model/policy_convnet.cpp
        model/policy_convnet_wrapper.cpp
        model/twoheaded_convnet.cpp
        model/twoheaded_convnet_wrapper.cpp
)
endif()
