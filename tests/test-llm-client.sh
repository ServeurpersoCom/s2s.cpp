#!/bin/bash

set -eo pipefail

./test-llm-client.py 2>&1 | tee llm-client.log
