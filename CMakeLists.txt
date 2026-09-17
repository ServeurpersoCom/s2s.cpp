cmake_minimum_required(VERSION 3.14)
project(s2s-ggml LANGUAGES C CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

# version.h: embed git commit hash into all binaries.
# runs on every build, only rewrites if the hash changed.
set(VERSION_OUTPUT "${CMAKE_CURRENT_BINARY_DIR}/version.h")
add_custom_target(version ALL
    COMMAND "${CMAKE_COMMAND}" "-DSRC_DIR=${CMAKE_CURRENT_SOURCE_DIR}" "-DOUTPUT=${VERSION_OUTPUT}"
            -P "${CMAKE_CURRENT_SOURCE_DIR}/tools/version.cmake"
    BYPRODUCTS "${VERSION_OUTPUT}"
    COMMENT "Checking git version"
)

# pthread: required explicitly on older glibc (< 2.34) where libpthread
# is not merged into libc.
find_package(Threads REQUIRED)

# Force UTF-8 source and execution charsets so non-ASCII string literals
# (SentencePiece pieces, transcripts in 25 languages) survive the MSVC
# compile without a BOM. Restricted to C and C++ since nvcc treats a bare
# /utf-8 as an input filename.
if(MSVC)
    add_compile_definitions(_CRT_SECURE_NO_WARNINGS)
    add_compile_options($<$<COMPILE_LANGUAGE:C,CXX>:/utf-8>)
endif()

# Executables and backend .so share the build root so ggml_backend_load_all()
# finds the backends at runtime.
set(CMAKE_RUNTIME_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR})
set(CMAKE_LIBRARY_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR})

# FastConformer GGUF tensor names can exceed the default GGML_MAX_NAME of 64.
add_compile_definitions(GGML_MAX_NAME=128)

# Harden: mark fread/fwrite with warn_unused_result. SYCL excluded since
# _FORTIFY_SOURCE swaps memcpy for __memcpy_chk, unresolved in device code.
if(NOT MSVC AND NOT GGML_SYCL)
    add_compile_definitions(_FORTIFY_SOURCE=2)
endif()

# CUDA architectures: Turing to Blackwell for distributed binaries.
# Override with -DCMAKE_CUDA_ARCHITECTURES=native for local builds.
if(NOT DEFINED CMAKE_CUDA_ARCHITECTURES)
    find_package(CUDAToolkit QUIET)
    if(CUDAToolkit_FOUND AND CUDAToolkit_VERSION VERSION_GREATER_EQUAL "12.8")
        set(CMAKE_CUDA_ARCHITECTURES "75-virtual;80-virtual;86-real;89-real;120a-real;121a-real")
    else()
        set(CMAKE_CUDA_ARCHITECTURES "75-virtual;80-virtual;86-real;89-real")
    endif()
endif()

# ggml as subdirectory, inherits GGML_CUDA, GGML_METAL, etc. from cmake flags.
# CUDA graphs default on: standalone ggml ships them off. Overridable with
# -DGGML_CUDA_GRAPHS=OFF or at runtime with GGML_CUDA_DISABLE_GRAPHS=1.
if(NOT DEFINED GGML_CUDA_GRAPHS)
    set(GGML_CUDA_GRAPHS_DEFAULT ON)
endif()
add_subdirectory(ggml)

# qwentts.cpp as a submodule: the TTS runs in process through its C ABI.
# Its own CMakeLists builds a standalone project with its own ggml, its own
# vendored httplib and its own tool targets, so the sources are compiled here
# instead: one ggml, one device context, no target name collision. The
# submodule keeps its include directory to itself, ahead of src/, since both
# trees carry a backend.h and a gguf-weights.h of their own.
set(QWENTTS_VERSION_OUTPUT "${CMAKE_CURRENT_BINARY_DIR}/qwentts/version.h")
add_custom_target(qwentts-version ALL
    COMMAND "${CMAKE_COMMAND}" "-DSRC_DIR=${CMAKE_CURRENT_SOURCE_DIR}/qwentts.cpp"
            "-DOUTPUT=${QWENTTS_VERSION_OUTPUT}" -DMACRO=QWEN_VERSION
            -P "${CMAKE_CURRENT_SOURCE_DIR}/tools/version.cmake"
    BYPRODUCTS "${QWENTTS_VERSION_OUTPUT}"
    COMMENT "Checking qwentts version"
)

add_library(qwen-core STATIC
    qwentts.cpp/src/qwen.cpp
    qwentts.cpp/src/pipeline-tts.cpp
    qwentts.cpp/src/pipeline-codec.cpp
)
target_compile_definitions(qwen-core PUBLIC QWEN_STATIC)
target_include_directories(qwen-core PUBLIC ${CMAKE_CURRENT_SOURCE_DIR}/qwentts.cpp/src)
target_include_directories(qwen-core PRIVATE ${CMAKE_CURRENT_BINARY_DIR}/qwentts)
target_include_directories(qwen-core SYSTEM PRIVATE ${CMAKE_CURRENT_SOURCE_DIR}/ggml/include)
target_link_libraries(qwen-core PUBLIC ggml Threads::Threads)
if(TARGET ggml-base)
    target_link_libraries(qwen-core PUBLIC ggml-base)
endif()
add_dependencies(qwen-core qwentts-version)

# cpp-httplib: HTTP and WebSocket server for s2s-server.
add_subdirectory(vendor/cpp-httplib)

# yyjson (MIT, fast JSON parser and writer): Realtime protocol frames.
add_library(yyjson STATIC vendor/yyjson/yyjson.c)
target_include_directories(yyjson PUBLIC ${CMAKE_CURRENT_SOURCE_DIR}/vendor/yyjson)
if(MSVC)
    target_compile_options(yyjson PRIVATE /W0)
else()
    target_compile_options(yyjson PRIVATE -w)
endif()

# Shared compile options and ggml linkage
macro(link_ggml_backends target)
    target_include_directories(${target} PRIVATE
        ${CMAKE_SOURCE_DIR}/src
        ${CMAKE_SOURCE_DIR}
        ${CMAKE_BINARY_DIR}
    )
    target_include_directories(${target} SYSTEM PRIVATE
        ${CMAKE_SOURCE_DIR}/ggml/include
    )
    if(MSVC)
        target_compile_options(${target} PRIVATE /W4 /wd4100 /wd4505)
    else()
        target_compile_options(${target} PRIVATE -Wall -Wextra -Wshadow -Wconversion
                              -Wno-unused-parameter -Wno-unused-function -Wno-sign-conversion)
    endif()
    target_link_libraries(${target} PRIVATE ggml Threads::Threads)
    if(TARGET ggml-base)
        target_link_libraries(${target} PRIVATE ggml-base)
    endif()
    foreach(backend cpu blas cuda metal vulkan sycl)
        if(TARGET ggml-${backend})
            get_target_property(CURRENT_BACKEND_TYPE ggml-${backend} TYPE)
            if (CURRENT_BACKEND_TYPE STREQUAL "MODULE_LIBRARY")
                # DL mode: backend is loaded at runtime via dlopen,
                # skip all link-time deps.
                continue()
            endif()
            target_link_libraries(${target} PRIVATE ggml-${backend})
        endif()
    endforeach()

    # SYCL links its runtime PRIVATE, consumers need -fsycl to resolve libsycl.so
    if(TARGET ggml-sycl AND GGML_SYCL)
        target_link_options(${target} PRIVATE -fsycl)
    endif()
    add_dependencies(${target} version)
endmacro()


# parakeet-transcribe: audio file to text, plus the stage dump debug path
add_executable(parakeet-transcribe tools/parakeet-transcribe.cpp src/parakeet.cpp src/pipeline-asr.cpp)
link_ggml_backends(parakeet-transcribe)

# webui: convert tools/public/index.html.gz to a C header for embedding.
# the .gz is committed to git so the C++ build works without npm.
# to update: cd tools/webui && npm install && npm run build, then rebuild.
set(WEBUI_INPUT "${CMAKE_CURRENT_SOURCE_DIR}/tools/public/index.html.gz")
set(WEBUI_OUTPUT "${CMAKE_CURRENT_BINARY_DIR}/index.html.gz.hpp")
add_custom_command(
    OUTPUT "${WEBUI_OUTPUT}"
    COMMAND "${CMAKE_COMMAND}" "-DINPUT=${WEBUI_INPUT}" "-DOUTPUT=${WEBUI_OUTPUT}" -P "${CMAKE_CURRENT_SOURCE_DIR}/tools/xxd.cmake"
    DEPENDS "${WEBUI_INPUT}"
    COMMENT "Embedding webui into index.html.gz.hpp"
)
set_source_files_properties(${WEBUI_OUTPUT} PROPERTIES GENERATED TRUE)

# s2s.js: the same client as a standalone ES module, embedded next to the page
# and served on /s2s.js for integrators who host their own front end.
set(WEBUI_LIB_INPUT "${CMAKE_CURRENT_SOURCE_DIR}/tools/public/s2s.js.gz")
set(WEBUI_LIB_OUTPUT "${CMAKE_CURRENT_BINARY_DIR}/s2s.js.gz.hpp")
add_custom_command(
    OUTPUT "${WEBUI_LIB_OUTPUT}"
    COMMAND "${CMAKE_COMMAND}" "-DINPUT=${WEBUI_LIB_INPUT}" "-DOUTPUT=${WEBUI_LIB_OUTPUT}" -P "${CMAKE_CURRENT_SOURCE_DIR}/tools/xxd.cmake"
    DEPENDS "${WEBUI_LIB_INPUT}"
    COMMENT "Embedding s2s.js into s2s.js.gz.hpp"
)
set_source_files_properties(${WEBUI_LIB_OUTPUT} PROPERTIES GENERATED TRUE)

# s2s-server: the product, WebSocket voice loop plus the embedded webui
add_executable(s2s-server tools/s2s-server.cpp
    src/s2s-session.cpp
    src/silero.cpp
    src/smart-turn.cpp
    src/parakeet.cpp
    src/pipeline-asr.cpp
    src/llm-client.cpp
    src/tts-bridge.cpp
    ${WEBUI_OUTPUT}
    ${WEBUI_LIB_OUTPUT})
target_include_directories(s2s-server PRIVATE ${CMAKE_CURRENT_BINARY_DIR})
link_ggml_backends(s2s-server)
target_link_libraries(s2s-server PRIVATE qwen-core httplib yyjson)

# quantize: GGUF requantizer (F32 -> K-quants)
add_executable(quantize tools/quantize.cpp)
link_ggml_backends(quantize)

add_subdirectory(tests)
