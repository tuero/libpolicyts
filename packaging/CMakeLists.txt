include(GNUInstallDirs)
include(CMakePackageConfigHelpers)

if (NOT DEFINED libpolicyts_INSTALL_CMAKEDIR)
   set(libpolicyts_INSTALL_CMAKEDIR "${CMAKE_INSTALL_LIBDIR}/cmake/libpolicyts"
       CACHE STRING "Path to libpolicyts CMake files")
endif ()

install(TARGETS libpolicyts EXPORT libpolicyts_Targets
        RUNTIME COMPONENT libpolicyts_Runtime
        LIBRARY COMPONENT libpolicyts_Runtime
        NAMELINK_COMPONENT libpolicyts_Development
        ARCHIVE COMPONENT libpolicyts_Development
        INCLUDES DESTINATION "${CMAKE_INSTALL_INCLUDEDIR}")

install(DIRECTORY "${libpolicyts_SOURCE_DIR}/include/" 
        TYPE INCLUDE
        COMPONENT libpolicyts_Development)

if (BUILD_SHARED_LIBS)
    set(type shared)
    message(STATUS "Building libpolicyts shared lib")
else ()
    set(type static)
    message(STATUS "Building libpolicyts static lib")
endif ()

install(EXPORT libpolicyts_Targets
        DESTINATION "${libpolicyts_INSTALL_CMAKEDIR}"
        NAMESPACE libpolicyts::
        FILE libpolicyts-${type}-targets.cmake
        COMPONENT libpolicyts_Development)

configure_package_config_file(
    "${CMAKE_CURRENT_SOURCE_DIR}/libpolicytsConfig.cmake.in"
    "${CMAKE_CURRENT_BINARY_DIR}/libpolicytsConfig.cmake"
    INSTALL_DESTINATION "${libpolicyts_INSTALL_CMAKEDIR}"
)

write_basic_package_version_file(
    "${CMAKE_CURRENT_BINARY_DIR}/libpolicytsConfigVersion.cmake"
    COMPATIBILITY SameMajorVersion
)

install(FILES
        "${CMAKE_CURRENT_BINARY_DIR}/libpolicytsConfig.cmake"
        "${CMAKE_CURRENT_BINARY_DIR}/libpolicytsConfigVersion.cmake"
        DESTINATION "${libpolicyts_INSTALL_CMAKEDIR}"
        COMPONENT libpolicyts_Development)
