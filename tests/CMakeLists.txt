# Parity harnesses: one target per model stage, each dumping tensors that
# the matching Python script compares against the reference implementation.

# test-silero: speech probability series against the ONNX reference.
add_executable(test-silero test-silero.cpp ${CMAKE_SOURCE_DIR}/src/silero.cpp)
link_ggml_backends(test-silero)

# test-smart-turn: completion probability series against the ONNX reference.
add_executable(test-smart-turn test-smart-turn.cpp ${CMAKE_SOURCE_DIR}/src/smart-turn.cpp)
link_ggml_backends(test-smart-turn)

# test-localvqe: echo canceller output against the PyTorch reference.
add_executable(test-localvqe test-localvqe.cpp ${CMAKE_SOURCE_DIR}/src/localvqe.cpp)
link_ggml_backends(test-localvqe)

# The Parakeet parity harness drives parakeet-transcribe --dump directly, so
# it needs no target of its own.

# test-session: turn timeline of the state machine, VAD to recognizer.
add_executable(test-session test-session.cpp
    ${CMAKE_SOURCE_DIR}/src/s2s-session.cpp
    ${CMAKE_SOURCE_DIR}/src/silero.cpp
    ${CMAKE_SOURCE_DIR}/src/smart-turn.cpp)
link_ggml_backends(test-session)
target_link_libraries(test-session PRIVATE parakeet-core)

# test-tts-bridge: streaming synthesis and barge-in through the qwentts ABI.
add_executable(test-tts-bridge test-tts-bridge.cpp ${CMAKE_SOURCE_DIR}/src/tts-bridge.cpp)
link_ggml_backends(test-tts-bridge)
target_link_libraries(test-tts-bridge PRIVATE qwen-core)

# test-llm-client: streaming, splitting and cancellation against a mock endpoint.
add_executable(test-llm-client test-llm-client.cpp ${CMAKE_SOURCE_DIR}/src/llm-client.cpp)
link_ggml_backends(test-llm-client)
target_link_libraries(test-llm-client PRIVATE httplib yyjson)

# test-server: one conversation over the Realtime WebSocket.
add_executable(test-server test-server.cpp)
link_ggml_backends(test-server)
target_link_libraries(test-server PRIVATE httplib yyjson)
