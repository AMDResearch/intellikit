target_sources(proboscis PRIVATE
    ${CMAKE_CURRENT_SOURCE_DIR}/interceptor.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/probe_buffer.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/kernarg_repack.cpp
)
