add_executable(bfs bfs.cpp)
target_link_libraries(bfs PRIVATE libpolicyts)
target_compile_features(bfs PUBLIC cxx_std_23)
