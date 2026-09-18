#!/bin/bash

set -eo pipefail

./test-localvqe.py 2>&1 | tee localvqe.log
