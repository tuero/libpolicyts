add_executable(levints levints.cpp)
target_link_libraries(levints PRIVATE libpolicyts)
target_compile_features(levints PUBLIC cxx_std_23)
