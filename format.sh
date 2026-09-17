#!/bin/bash

find . -name "*.cpp" -o -name "*.h" | grep -v -e build/ -e ggml/ -e qwentts.cpp -e vendor/ | xargs clang-format -i
