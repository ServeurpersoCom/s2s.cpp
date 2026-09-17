#pragma once
// pipeline-asr.h: pcm -> text orchestration. Owns the backend pair, the loaded
// weights, the piece table and the decoder graphs, and drives mel -> conv2d
// stem -> FastConformer encoder -> projection -> TDT greedy loop ->
// detokenize. The public ABI in parakeet.h wraps this; the debug path calls it
// directly to dump the stage tensors.

#include "parakeet.h"
#include "s2s-error.h"

#include <string>
#include <vector>

struct pipeline_asr;

struct pipeline_asr_params {
    std::string model_path;
    int         n_threads = 0;
    bool        use_gpu   = true;
    std::string dump_dir;  // when set, write the stage tensors there as f32 dumps
};

pipeline_asr * pipeline_asr_load(const pipeline_asr_params & params);
void           pipeline_asr_free(pipeline_asr * p);

int pipeline_asr_sample_rate(const pipeline_asr * p);

// Full recognition. params carries the streaming callback, text receives the
// decoded UTF-8 string.
pk_status pipeline_asr_run(pipeline_asr *               p,
                           const float *                pcm,
                           size_t                       n_samples,
                           int                          sample_rate,
                           const pk_transcribe_params * params,
                           std::string &                text);
