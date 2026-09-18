@echo off

set PATH=%~dp0build\Release;%PATH%

rem Multi-GPU: set GGML_BACKEND to pick a device (CUDA0, CUDA1, Vulkan0...)
rem set GGML_BACKEND=CUDA0
rem set GGML_BACKEND=Vulkan0

rem Everything else belongs to the client: mode, endpoint, prompt, voice,
rem sampling and turn detection travel in session.update, and their defaults
rem are published on /props.

s2s-server.exe ^
    --host 0.0.0.0 ^
    --port 8088 ^
    --models .\models ^
    --origin http://localhost:8088 ^
    --origin http://127.0.0.1:8088 ^
    --llm-host 127.0.0.1:8080

pause
