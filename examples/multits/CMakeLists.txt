add_executable(multits multits.cpp)
target_link_libraries(multits PRIVATE libpolicyts)
target_compile_features(multits PUBLIC cxx_std_23)
