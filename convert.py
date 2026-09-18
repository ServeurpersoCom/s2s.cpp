#!/usr/bin/env python3
# convert.py: upstream checkpoints to GGUF
#
# Usage: ./convert.py [model ...]
#   silero      checkpoints/silero-vad/onnx/model.onnx -> models/silero-vad-F32.gguf
#   smart-turn  checkpoints/smart-turn/smart-turn-v3.2-gpu.onnx -> models/smart-turn-v3.2-F32.gguf
#   parakeet    checkpoints/parakeet/model.safetensors -> models/parakeet-tdt-0.6b-v3-F32.gguf
#   localvqe    checkpoints/localvqe/localvqe-v1.3-4.8M.pt -> models/localvqe-v1.3-F32.gguf
#
# Every model of the pipeline gets a subcommand here so the whole project
# converts from one entry point. Outputs that already exist are skipped.

import argparse
import os
import sys

import numpy as np

import gguf

CHECKPOINTS = "checkpoints"
MODELS = "models"


# Silero VAD ships as a single ONNX graph holding both sample rates behind an
# If node keyed on the sr input. The 16 kHz branch is the only one this project
# runs, so the converter walks the subgraphs, keeps the tensors prefixed
# If_0_then_branch, and drops the 8 kHz twin.
#
# The branch computes:
#   reflect pad 64 on the right of the 576 sample input (64 context + 512 new)
#   conv1d(basis, k=256, stride=128) -> 258 channels over 4 frames
#   magnitude = sqrt(real^2 + imag^2) over the 129 bin split
#   4 conv1d + relu layers, strides 1, 2, 2, 1 -> 128 channels, 1 frame
#   LSTM 128 wide, gate order i, o, f, c
#   relu -> conv1d k=1 -> sigmoid -> speech probability
#
# The ONNX bias pair Wb and Rb is summed at convert time: both apply to the
# same gate preactivation, so one vector is enough at runtime.
SILERO_PREFIX = "If_0_then_branch__Inline_0__"


def onnx_initializers(graph, out):
    from onnx import numpy_helper

    for tensor in graph.initializer:
        out[tensor.name] = numpy_helper.to_array(tensor)
    for node in graph.node:
        for attr in node.attribute:
            if attr.g.ByteSize():
                onnx_initializers(attr.g, out)


def onnx_find_node(graph, op_type, prefix):
    for node in graph.node:
        if node.op_type == op_type and node.name.startswith(prefix):
            return node
        for attr in node.attribute:
            if attr.g.ByteSize():
                found = onnx_find_node(attr.g, op_type, prefix)
                if found is not None:
                    return found
    return None


def convert_silero(src, dst):
    import onnx

    model = onnx.load(src)
    tensors = {}
    onnx_initializers(model.graph, tensors)

    def get(name, shape):
        array = tensors[SILERO_PREFIX + name]
        if list(array.shape) != shape:
            raise SystemExit(f"[Silero] {name}: expected {shape}, found {list(array.shape)}")
        return array.astype(np.float32)

    basis = get("stft.forward_basis_buffer", [258, 1, 256])
    dec_w = get("decoder.decoder.2.weight", [1, 128, 1])
    dec_b = get("decoder.decoder.2.bias", [1])

    encoder = []
    for layer, shape in enumerate([[128, 129, 3], [64, 128, 3], [64, 64, 3], [128, 64, 3]]):
        weight = get(f"encoder.{layer}.reparam_conv.weight", shape)
        bias = get(f"encoder.{layer}.reparam_conv.bias", [shape[0]])
        encoder.append((weight, bias))

    # The LSTM weights sit in unnamed Unsqueeze outputs, so they are resolved
    # through the LSTM node inputs: X, W, R, B.
    lstm = onnx_find_node(model.graph, "LSTM", SILERO_PREFIX)
    lstm_w = tensors[lstm.input[1]].reshape(512, 128)
    lstm_r = tensors[lstm.input[2]].reshape(512, 128)
    lstm_b = tensors[lstm.input[3]].reshape(1024)

    writer = gguf.GGUFWriter(dst, "silero-vad")
    writer.add_uint32("vad.sample_rate", 16000)
    writer.add_uint32("vad.window", 512)
    writer.add_uint32("vad.context", 64)
    writer.add_uint32("vad.n_fft", 256)
    writer.add_uint32("vad.hop", 128)
    writer.add_uint32("vad.n_bins", 129)
    writer.add_uint32("vad.hidden", 128)

    writer.add_tensor("stft.basis", basis)
    for layer, (weight, bias) in enumerate(encoder):
        writer.add_tensor(f"enc.{layer}.weight", weight)
        writer.add_tensor(f"enc.{layer}.bias", bias)
    writer.add_tensor("lstm.w", lstm_w.astype(np.float32))
    writer.add_tensor("lstm.r", lstm_r.astype(np.float32))
    writer.add_tensor("lstm.b", (lstm_b[:512] + lstm_b[512:]).astype(np.float32))
    writer.add_tensor("dec.weight", dec_w.reshape(128).astype(np.float32))
    writer.add_tensor("dec.bias", dec_b.astype(np.float32))

    writer.write_header_to_file()
    writer.write_kv_data_to_file()
    writer.write_tensors_to_file()
    writer.close()


# Smart Turn v3.2 is a Whisper tiny encoder with an attention pooling head.
# The ONNX export keeps the biases under their torch names but stores every
# projection matrix as an anonymous val_* initializer, so the converter walks
# the graph in execution order and takes the matrices as they come:
#
#   conv1, conv2, positions, then per layer q, k, v, out, fc1, fc2,
#   then the two pooling matrices and the three classifier matrices
#
# Shapes are asserted at every step, which is what makes the positional read
# safe. ONNX stores the projections as x @ W with W in [in, out], while GGML
# multiplies with the weight first, so the matrices are transposed here and
# land as [in, out] in ne terms.
SMART_TURN_LAYERS = 4
SMART_TURN_DIM = 384
SMART_TURN_FFN = 1536


def convert_smart_turn(src, dst):
    import onnx
    from onnx import numpy_helper

    model = onnx.load(src)
    graph = model.graph
    inits = {t.name: numpy_helper.to_array(t) for t in graph.initializer}

    matrices = []
    for node in graph.node:
        if node.op_type not in ("MatMul", "Gemm"):
            continue
        for name in node.input[1:]:
            array = inits.get(name)
            if array is not None and array.ndim == 2:
                matrices.append(array.astype(np.float32))

    expected = SMART_TURN_LAYERS * 6 + 5
    if len(matrices) != expected:
        raise SystemExit(f"[SmartTurn] expected {expected} matrices, walked {len(matrices)}")

    def named(name, shape):
        array = inits[name]
        if list(array.shape) != shape:
            raise SystemExit(f"[SmartTurn] {name}: expected {shape}, found {list(array.shape)}")
        return array.astype(np.float32)

    def take(index, shape):
        array = matrices[index]
        if list(array.shape) != shape:
            raise SystemExit(f"[SmartTurn] matrix {index}: expected {shape}, found {list(array.shape)}")
        return array

    writer = gguf.GGUFWriter(dst, "smart-turn")
    writer.add_uint32("turn.sample_rate", 16000)
    writer.add_uint32("turn.window", 128000)
    writer.add_uint32("turn.n_fft", 400)
    writer.add_uint32("turn.hop", 160)
    writer.add_uint32("turn.n_mels", 80)
    writer.add_uint32("turn.n_frames", 800)
    writer.add_uint32("turn.n_layers", SMART_TURN_LAYERS)
    writer.add_uint32("turn.n_heads", 6)
    writer.add_uint32("turn.d_model", SMART_TURN_DIM)
    writer.add_uint32("turn.d_ffn", SMART_TURN_FFN)

    prefix = "inner.encoder."
    writer.add_tensor("enc.conv1.weight", named(prefix + "conv1.weight", [384, 80, 3]))
    writer.add_tensor("enc.conv1.bias", named(prefix + "conv1.bias", [384]))
    writer.add_tensor("enc.conv2.weight", named(prefix + "conv2.weight", [384, 384, 3]))
    writer.add_tensor("enc.conv2.bias", named(prefix + "conv2.bias", [384]))
    writer.add_tensor("enc.pos", named(prefix + "embed_positions.weight", [400, 384]))

    cursor = 0
    for layer in range(SMART_TURN_LAYERS):
        lp = f"{prefix}layers.{layer}."
        out = f"enc.{layer}."
        writer.add_tensor(out + "attn_norm.weight", named(lp + "self_attn_layer_norm.weight", [384]))
        writer.add_tensor(out + "attn_norm.bias", named(lp + "self_attn_layer_norm.bias", [384]))
        writer.add_tensor(out + "q.weight", take(cursor + 0, [384, 384]).T)
        writer.add_tensor(out + "q.bias", named(lp + "self_attn.q_proj.bias", [384]))
        writer.add_tensor(out + "k.weight", take(cursor + 1, [384, 384]).T)
        writer.add_tensor(out + "v.weight", take(cursor + 2, [384, 384]).T)
        writer.add_tensor(out + "v.bias", named(lp + "self_attn.v_proj.bias", [384]))
        writer.add_tensor(out + "o.weight", take(cursor + 3, [384, 384]).T)
        writer.add_tensor(out + "o.bias", named(lp + "self_attn.out_proj.bias", [384]))
        writer.add_tensor(out + "ffn_norm.weight", named(lp + "final_layer_norm.weight", [384]))
        writer.add_tensor(out + "ffn_norm.bias", named(lp + "final_layer_norm.bias", [384]))
        writer.add_tensor(out + "fc1.weight", take(cursor + 4, [384, 1536]).T)
        writer.add_tensor(out + "fc1.bias", named(lp + "fc1.bias", [1536]))
        writer.add_tensor(out + "fc2.weight", take(cursor + 5, [1536, 384]).T)
        writer.add_tensor(out + "fc2.bias", named(lp + "fc2.bias", [384]))
        cursor += 6

    writer.add_tensor("enc.norm.weight", named(prefix + "layer_norm.weight", [384]))
    writer.add_tensor("enc.norm.bias", named(prefix + "layer_norm.bias", [384]))

    writer.add_tensor("pool.hidden.weight", take(cursor + 0, [384, 256]).T)
    writer.add_tensor("pool.hidden.bias", named("inner.pool_attention.0.bias", [256]))
    writer.add_tensor("pool.score.weight", take(cursor + 1, [256, 1]).T)
    writer.add_tensor("pool.score.bias", named("inner.pool_attention.2.bias", [1]))

    writer.add_tensor("cls.0.weight", named("inner.classifier.0.weight", [256, 384]))
    writer.add_tensor("cls.0.bias", named("inner.classifier.0.bias", [256]))
    writer.add_tensor("cls.norm.weight", named("inner.classifier.1.weight", [256]))
    writer.add_tensor("cls.norm.bias", named("inner.classifier.1.bias", [256]))
    writer.add_tensor("cls.1.weight", named("inner.classifier.4.weight", [64, 256]))
    writer.add_tensor("cls.1.bias", named("inner.classifier.4.bias", [64]))
    writer.add_tensor("cls.2.weight", named("inner.classifier.6.weight", [1, 64]))
    writer.add_tensor("cls.2.bias", named("inner.classifier.6.bias", [1]))

    writer.write_header_to_file()
    writer.write_kv_data_to_file()
    writer.write_tensors_to_file()
    writer.close()


# Parakeet TDT 0.6B v3: FastConformer encoder, LSTM prediction network and
# the TDT joint. Two shape conventions have to be reconciled:
#
#   torch linear weights are [out, in] and land as [in, out] in ne terms,
#   which is what ggml_mul_mat wants, so they pass through untouched
#
#   the conv module BatchNorm is folded into the depthwise kernel at
#   convert time: scale = weight / sqrt(var + eps), the kernel is scaled
#   row by row and the shift becomes the bias the runtime adds
#
# The joint head covers vocab + blank + durations in one matrix, split at
# vocab_size by the runtime.
PARAKEET_BN_EPS = 1e-5


def convert_parakeet(src, dst):
    import json

    from safetensors import safe_open

    root = os.path.dirname(src)
    config = json.load(open(os.path.join(root, "config.json")))
    enc = config["encoder_config"]

    tensors = {}
    with safe_open(src, framework="np") as f:
        for key in f.keys():
            tensors[key] = f.get_tensor(key)

    def get(name, shape=None):
        array = tensors[name].astype(np.float32)
        if shape is not None and list(array.shape) != shape:
            raise SystemExit(f"[Parakeet] {name}: expected {shape}, found {list(array.shape)}")
        return array

    n_layers = enc["num_hidden_layers"]
    d_model = enc["hidden_size"]
    n_heads = enc["num_attention_heads"]

    writer = gguf.GGUFWriter(dst, "parakeet-tdt")
    writer.add_uint32("asr.sample_rate", 16000)
    writer.add_uint32("asr.n_fft", 512)
    writer.add_uint32("asr.win_length", 400)
    writer.add_uint32("asr.hop", 160)
    writer.add_uint32("asr.n_mels", enc["num_mel_bins"])
    writer.add_float32("asr.preemphasis", 0.97)
    writer.add_uint32("asr.n_layers", n_layers)
    writer.add_uint32("asr.n_heads", n_heads)
    writer.add_uint32("asr.d_model", d_model)
    writer.add_uint32("asr.d_ffn", enc["intermediate_size"])
    writer.add_uint32("asr.conv_kernel", enc["conv_kernel_size"])
    writer.add_uint32("asr.subsampling_channels", enc["subsampling_conv_channels"])
    writer.add_uint32("asr.subsampling_factor", enc["subsampling_factor"])
    writer.add_uint32("asr.d_decoder", config["decoder_hidden_size"])
    writer.add_uint32("asr.decoder_layers", config["num_decoder_layers"])
    writer.add_uint32("asr.vocab_size", config["vocab_size"])
    writer.add_uint32("asr.blank_id", config["blank_token_id"])
    writer.add_uint32("asr.max_symbols_per_step", config["max_symbols_per_step"])
    writer.add_array("asr.durations", config["durations"])

    # Subsampling: conv2d 1 -> 256 stride 2, then two depthwise plus pointwise
    # pairs, then the linear that flattens channels times frequency.
    channels = enc["subsampling_conv_channels"]
    writer.add_tensor("sub.0.weight", get("encoder.subsampling.layers.0.weight", [channels, 1, 3, 3]))
    writer.add_tensor("sub.0.bias", get("encoder.subsampling.layers.0.bias", [channels]))
    for index, (depth, point) in enumerate([(2, 3), (5, 6)]):
        writer.add_tensor(f"sub.{index + 1}.dw.weight", get(f"encoder.subsampling.layers.{depth}.weight", [channels, 1, 3, 3]))
        writer.add_tensor(f"sub.{index + 1}.dw.bias", get(f"encoder.subsampling.layers.{depth}.bias", [channels]))
        writer.add_tensor(f"sub.{index + 1}.pw.weight", get(f"encoder.subsampling.layers.{point}.weight", [channels, channels, 1, 1]).reshape(channels, channels))
        writer.add_tensor(f"sub.{index + 1}.pw.bias", get(f"encoder.subsampling.layers.{point}.bias", [channels]))
    writer.add_tensor("sub.linear.weight", get("encoder.subsampling.linear.weight"))
    writer.add_tensor("sub.linear.bias", get("encoder.subsampling.linear.bias", [d_model]))

    for layer in range(n_layers):
        src_prefix = f"encoder.layers.{layer}."
        out = f"enc.{layer}."

        for torch_name, gguf_name in [
            ("norm_feed_forward1", "norm_ff1"),
            ("norm_self_att", "norm_attn"),
            ("norm_conv", "norm_conv"),
            ("norm_feed_forward2", "norm_ff2"),
            ("norm_out", "norm_out"),
        ]:
            writer.add_tensor(out + gguf_name + ".weight", get(src_prefix + torch_name + ".weight", [d_model]))
            writer.add_tensor(out + gguf_name + ".bias", get(src_prefix + torch_name + ".bias", [d_model]))

        for index in (1, 2):
            writer.add_tensor(out + f"ff{index}.linear1.weight", get(src_prefix + f"feed_forward{index}.linear1.weight"))
            writer.add_tensor(out + f"ff{index}.linear2.weight", get(src_prefix + f"feed_forward{index}.linear2.weight"))

        writer.add_tensor(out + "attn.q.weight", get(src_prefix + "self_attn.q_proj.weight", [d_model, d_model]))
        writer.add_tensor(out + "attn.k.weight", get(src_prefix + "self_attn.k_proj.weight", [d_model, d_model]))
        writer.add_tensor(out + "attn.v.weight", get(src_prefix + "self_attn.v_proj.weight", [d_model, d_model]))
        writer.add_tensor(out + "attn.o.weight", get(src_prefix + "self_attn.o_proj.weight", [d_model, d_model]))
        writer.add_tensor(out + "attn.rel_k.weight", get(src_prefix + "self_attn.relative_k_proj.weight", [d_model, d_model]))
        writer.add_tensor(out + "attn.bias_u", get(src_prefix + "self_attn.bias_u", [n_heads, d_model // n_heads]))
        writer.add_tensor(out + "attn.bias_v", get(src_prefix + "self_attn.bias_v", [n_heads, d_model // n_heads]))

        writer.add_tensor(out + "conv.pw1.weight", get(src_prefix + "conv.pointwise_conv1.weight", [2 * d_model, d_model, 1]).reshape(2 * d_model, d_model))
        writer.add_tensor(out + "conv.pw2.weight", get(src_prefix + "conv.pointwise_conv2.weight", [d_model, d_model, 1]).reshape(d_model, d_model))

        # BatchNorm folded into the depthwise kernel.
        depthwise = get(src_prefix + "conv.depthwise_conv.weight", [d_model, 1, enc["conv_kernel_size"]])
        gamma = get(src_prefix + "conv.norm.weight", [d_model])
        beta = get(src_prefix + "conv.norm.bias", [d_model])
        mean = get(src_prefix + "conv.norm.running_mean", [d_model])
        var = get(src_prefix + "conv.norm.running_var", [d_model])
        scale = gamma / np.sqrt(var + PARAKEET_BN_EPS)
        writer.add_tensor(out + "conv.dw.weight", depthwise * scale[:, None, None])
        writer.add_tensor(out + "conv.dw.bias", beta - mean * scale)

    writer.add_tensor("enc.proj.weight", get("encoder_projector.weight", [640, d_model]))
    writer.add_tensor("enc.proj.bias", get("encoder_projector.bias", [640]))

    d_dec = config["decoder_hidden_size"]
    writer.add_tensor("dec.embedding.weight", get("decoder.embedding.weight", [config["vocab_size"], d_dec]))
    for layer in range(config["num_decoder_layers"]):
        # torch keeps the input and hidden gates apart, and both biases apply
        # to the same preactivation, so they are summed here.
        writer.add_tensor(f"dec.lstm.{layer}.w", get(f"decoder.lstm.weight_ih_l{layer}", [4 * d_dec, d_dec]))
        writer.add_tensor(f"dec.lstm.{layer}.r", get(f"decoder.lstm.weight_hh_l{layer}", [4 * d_dec, d_dec]))
        bias = get(f"decoder.lstm.bias_ih_l{layer}", [4 * d_dec]) + get(f"decoder.lstm.bias_hh_l{layer}", [4 * d_dec])
        writer.add_tensor(f"dec.lstm.{layer}.b", bias)
    writer.add_tensor("dec.proj.weight", get("decoder.decoder_projector.weight", [d_dec, d_dec]))
    writer.add_tensor("dec.proj.bias", get("decoder.decoder_projector.bias", [d_dec]))

    # SentencePiece pieces in id order, the runtime only ever decodes.
    vocab = json.load(open(os.path.join(root, "tokenizer.json")))["model"]["vocab"]
    pieces = [""] * config["vocab_size"]
    for piece, index in vocab.items():
        pieces[index] = piece
    writer.add_array("asr.vocab", pieces)

    writer.add_tensor("joint.weight", get("joint.head.weight"))
    writer.add_tensor("joint.bias", get("joint.head.bias"))

    writer.write_header_to_file()
    writer.write_kv_data_to_file()
    writer.write_tensors_to_file()
    writer.close()


# LocalVQE v1.3 is a DeepVQE derivative: a 512 point analysis with a sqrt-Hann
# window folded into the weights, hop 256 at 16 kHz, mic and far end conv
# encoders, a soft delay cross-attention over dmax frames, a diagonal state
# space bottleneck, a subpixel decoder and a 3x3 complex convolving mask. The
# channel widths are pruned and uneven, so the runtime reads every one of them
# from the tensor shapes; only the fixed signal geometry goes to metadata.
#
# Two things are baked here so the runtime has no knob for them: the softmax
# temperature of the alignment is folded into its smoothing conv, and the
# polar state matrix of the bottleneck is turned into its cartesian a_real,
# a_imag pair. Torch conv weights [OC, IC, KH, KW] land as [KW, KH, IC, OC] in
# ne terms, the layout the im2col matmul takes; linear and 1x1 conv weights
# land as [in, out].
def convert_localvqe(src, dst):
    import torch

    checkpoint = torch.load(src, map_location="cpu", weights_only=False)
    state = {k: v.float().numpy() for k, v in checkpoint["model_state_dict"].items()}

    writer = gguf.GGUFWriter(dst, "localvqe")
    writer.add_uint32("lv.sample_rate", 16000)
    writer.add_uint32("lv.n_fft", 512)
    writer.add_uint32("lv.hop", 256)
    writer.add_uint32("lv.dmax", 64)
    writer.add_float32("lv.power_law_c", 0.3)
    writer.add_float32("lv.norm_eps", 1e-5)

    temperature = float(state.pop("align.temperature"))
    state["align.conv.1.weight"] /= temperature
    state["align.conv.1.bias"] /= temperature

    rate = np.log1p(np.exp(state.pop("bottleneck.A_log_rate")))
    radius = np.exp(-np.maximum(rate, 0.01))
    theta = state.pop("bottleneck.A_theta")
    state["bottleneck.a_real"] = radius * np.cos(theta)
    state["bottleneck.a_imag"] = radius * np.sin(theta)

    state["encoder.conv.weight"] = state["encoder.conv.weight"].reshape(512, 512)
    for name, value in state.items():
        if value.ndim == 4 and value.shape[2:] == (1, 1):
            value = value.reshape(value.shape[0], value.shape[1])
        writer.add_tensor(name, np.ascontiguousarray(value.astype(np.float32)))

    writer.write_header_to_file()
    writer.write_kv_data_to_file()
    writer.write_tensors_to_file()
    writer.close()


MODEL_TABLE = {
    "silero": (
        os.path.join(CHECKPOINTS, "silero-vad", "onnx", "model.onnx"),
        os.path.join(MODELS, "silero-vad-F32.gguf"),
        convert_silero,
    ),
    "smart-turn": (
        os.path.join(CHECKPOINTS, "smart-turn", "smart-turn-v3.2-gpu.onnx"),
        os.path.join(MODELS, "smart-turn-v3.2-F32.gguf"),
        convert_smart_turn,
    ),
    "parakeet": (
        os.path.join(CHECKPOINTS, "parakeet", "model.safetensors"),
        os.path.join(MODELS, "parakeet-tdt-0.6b-v3-F32.gguf"),
        convert_parakeet,
    ),
    "localvqe": (
        os.path.join(CHECKPOINTS, "localvqe", "localvqe-v1.3-4.8M.pt"),
        os.path.join(MODELS, "localvqe-v1.3-F32.gguf"),
        convert_localvqe,
    ),
}


def main():
    parser = argparse.ArgumentParser(description="Convert upstream checkpoints to GGUF")
    parser.add_argument("models", nargs="*", help=f"one or more of: {', '.join(MODEL_TABLE)}")
    args = parser.parse_args()

    selection = args.models or list(MODEL_TABLE)
    for name in selection:
        if name not in MODEL_TABLE:
            print(f"[Miss] unknown model '{name}', known: {', '.join(MODEL_TABLE)}", file=sys.stderr)
            return 1

    os.makedirs(MODELS, exist_ok=True)
    for name in selection:
        src, dst, fn = MODEL_TABLE[name]
        if os.path.exists(dst):
            print(f"[Skip] {dst}")
            continue
        if not os.path.exists(src):
            print(f"[Miss] {src}, run ./checkpoints.sh first", file=sys.stderr)
            return 1
        print(f"[Convert] {src} -> {dst}")
        fn(src, dst)
        print(f"[Done] {dst} ({os.path.getsize(dst) / 1e6:.1f} MB)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
