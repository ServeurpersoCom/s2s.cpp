# cpp-httplib: HTTP server and client.
#
# The server runs plain HTTP behind a reverse proxy, but the client side talks
# to whatever endpoint the user configures, and that includes https: a cloud
# API, or a llama-server behind their own TLS proxy. OpenSSL is therefore
# linked when the system provides it.

add_library(httplib STATIC httplib.cpp)
target_include_directories(httplib PUBLIC ${CMAKE_CURRENT_SOURCE_DIR})

# suppress warnings in third-party code
if(MSVC)
    target_compile_options(httplib PRIVATE /w)
else()
    target_compile_options(httplib PRIVATE -w)
endif()

if(WIN32)
    target_link_libraries(httplib PRIVATE ws2_32)
endif()

find_package(OpenSSL QUIET)
if(OpenSSL_FOUND)
    target_compile_definitions(httplib PUBLIC CPPHTTPLIB_OPENSSL_SUPPORT)
    target_link_libraries(httplib PUBLIC OpenSSL::SSL OpenSSL::Crypto)
    # httplib.h loads the macOS Keychain root certificates when TLS is enabled
    if(APPLE AND CMAKE_SYSTEM_NAME STREQUAL "Darwin")
        find_library(CORE_FOUNDATION_FRAMEWORK CoreFoundation REQUIRED)
        find_library(SECURITY_FRAMEWORK Security REQUIRED)
        target_link_libraries(httplib PUBLIC ${CORE_FOUNDATION_FRAMEWORK} ${SECURITY_FRAMEWORK})
    endif()
    message(STATUS "httplib: https client enabled (OpenSSL ${OPENSSL_VERSION})")
else()
    message(STATUS "httplib: no OpenSSL, https endpoints will be refused")
endif()
