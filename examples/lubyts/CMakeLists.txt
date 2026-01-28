add_executable(lubyts lubyts.cpp)
target_link_libraries(lubyts PRIVATE libpolicyts)
target_compile_features(lubyts PUBLIC cxx_std_23)
