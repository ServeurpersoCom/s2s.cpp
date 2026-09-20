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

# test-server: a scripted client over the Realtime WebSocket, with a mock
# endpoint in process: the behavior of the whole server.
add_executable(test-server test-server.cpp)
link_ggml_backends(test-server)
target_link_libraries(test-server PRIVATE httplib yyjson)
