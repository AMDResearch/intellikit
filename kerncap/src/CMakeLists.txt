list(APPEND CMAKE_MODULE_PATH "${CMAKE_CURRENT_SOURCE_DIR}")

find_package(HSA REQUIRED)
find_package(hip REQUIRED)

# Vendored header-only dependencies
set(VENDOR_DIR "${CMAKE_SOURCE_DIR}/vendor")

add_library(kerncap SHARED kerncap.hip)

set_target_properties(kerncap PROPERTIES
    OUTPUT_NAME "kerncap"
    PREFIX "lib"
    CXX_STANDARD 20
    HIP_STANDARD 20
    POSITION_INDEPENDENT_CODE ON
    CXX_VISIBILITY_PRESET hidden
)

target_include_directories(kerncap PRIVATE
    "${CMAKE_CURRENT_SOURCE_DIR}"
    "${VENDOR_DIR}"
    "${HSA_INCLUDE_DIRS}"
)

target_link_libraries(kerncap PRIVATE
    hsa::hsa
)

target_compile_options(kerncap PRIVATE
    -Wall -Wextra
    $<$<CONFIG:Debug>:-O0 -ggdb>
    $<$<CONFIG:Release>:-O2>
)

install(TARGETS kerncap
    LIBRARY DESTINATION "${CMAKE_INSTALL_LIBDIR}"
)

# kerncap-replay: VA-faithful HSA kernel replay binary
add_executable(kerncap-replay replay.cpp)

set_target_properties(kerncap-replay PROPERTIES
    CXX_STANDARD 20
    CXX_STANDARD_REQUIRED ON
)

target_include_directories(kerncap-replay PRIVATE
    "${HSA_INCLUDE_DIRS}"
    "${VENDOR_DIR}"
)

target_link_libraries(kerncap-replay PRIVATE
    hsa::hsa
    hip::host
)

# Allow unresolved symbols from hsa-runtime64's transitive dependencies
# (drm, elf, numa) which are resolved at runtime but may not be in the
# build-time linker's search path (e.g., conda's compiler_compat ld).
target_link_options(kerncap-replay PRIVATE "LINKER:--allow-shlib-undefined")

target_compile_options(kerncap-replay PRIVATE
    -Wall -Wextra
    $<$<CONFIG:Debug>:-O0 -ggdb>
    $<$<CONFIG:Release>:-O2>
)

install(TARGETS kerncap-replay RUNTIME DESTINATION "${CMAKE_INSTALL_BINDIR}")
