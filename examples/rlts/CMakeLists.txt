add_executable(rlts rlts.cpp)
target_link_libraries(rlts PRIVATE libpolicyts)
target_compile_features(rlts PUBLIC cxx_std_23)
