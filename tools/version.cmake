# Generate version.h with the current git commit hash and date.
# Only rewrites the file if the content changed (avoids rebuild cascade).
# MACRO names the symbol, so the qwentts submodule gets the QWEN_VERSION it
# includes while the project keeps S2S_VERSION.
# Usage: cmake -DSRC_DIR=... -DOUTPUT=... [-DMACRO=...] -P version.cmake

execute_process(
    COMMAND git rev-parse --short HEAD
    WORKING_DIRECTORY "${SRC_DIR}"
    OUTPUT_VARIABLE GIT_HASH
    OUTPUT_STRIP_TRAILING_WHITESPACE
    ERROR_QUIET
    RESULT_VARIABLE GIT_RESULT
)
if(NOT GIT_RESULT EQUAL 0)
    set(GIT_HASH "unknown")
endif()

execute_process(
    COMMAND git show -s --format=%cs HEAD
    WORKING_DIRECTORY "${SRC_DIR}"
    OUTPUT_VARIABLE GIT_DATE
    OUTPUT_STRIP_TRAILING_WHITESPACE
    ERROR_QUIET
    RESULT_VARIABLE DATE_RESULT
)
if(NOT DATE_RESULT EQUAL 0)
    set(GIT_DATE "unknown")
endif()

if(NOT DEFINED MACRO)
    set(MACRO "S2S_VERSION")
endif()

set(CONTENT "#pragma once\n#define ${MACRO} \"${GIT_HASH} (${GIT_DATE})\"\n")

if(EXISTS "${OUTPUT}")
    file(READ "${OUTPUT}" EXISTING)
    if("${EXISTING}" STREQUAL "${CONTENT}")
        return()
    endif()
endif()

file(WRITE "${OUTPUT}" "${CONTENT}")
