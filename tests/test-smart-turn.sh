#!/bin/bash

set -eo pipefail

./test-smart-turn.py 2>&1 | tee smart-turn.log
