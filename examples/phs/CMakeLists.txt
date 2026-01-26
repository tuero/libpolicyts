add_executable(phs phs.cpp)
target_link_libraries(phs PRIVATE libpolicyts)
target_compile_features(phs PUBLIC cxx_std_23)

# Build version using torch model
if(LIBPOLICYTS_BUILD_TORCH_SOURCES)
    add_executable(phs_torch phs_torch.cpp)
    target_link_libraries(phs_torch PRIVATE libpolicyts)
    target_compile_features(phs_torch PUBLIC cxx_std_23)
endif()
