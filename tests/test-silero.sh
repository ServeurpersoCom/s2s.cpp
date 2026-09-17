#!/bin/bash

set -eo pipefail

./test-silero.py 2>&1 | tee silero.log
