@echo off

cd ggml
git pull --rebase
cd ../qwentts.cpp
git pull --rebase
cd ..
git pull --rebase
