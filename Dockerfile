# syntax=docker/dockerfile:1
# Ternary Bonsai 27B on a DGX Spark (GB10, aarch64, CUDA 13).
#
# The weights are GGUF in quantisation types only PrismML's llama.cpp fork
# reads, so the runtime is llama.cpp rather than vLLM and the decision server
# scores autoregressive logprobs rather than a diffusion canvas.

ARG CUDA_IMAGE=nvidia/cuda:13.0.0-devel-ubuntu24.04
ARG CUDA_RUNTIME=nvidia/cuda:13.0.0-runtime-ubuntu24.04

FROM ${CUDA_IMAGE} AS build
ARG LLAMA_REPO=https://github.com/PrismML-Eng/llama.cpp
ARG LLAMA_REF=9a9394a895b96003ca842a6041cb28ac49a108f7
# GB10 is sm_121; building for anything else produces a binary this box cannot run.
ARG CUDA_ARCH=121
RUN apt-get update && apt-get install -y --no-install-recommends \
      git cmake ninja-build build-essential ca-certificates \
    && rm -rf /var/lib/apt/lists/*

# libcuda is the driver library: the container runtime injects the real one at
# run time and the image carries only a stub, whose directory is named after the
# target (sbsa-linux on arm64), so find it rather than assuming the path. The
# stub stays in this stage; shipping it would shadow the injected driver.
RUN stub=$(find /usr/local/cuda/targets -path "*/stubs/libcuda.so" | head -1) \
    && test -n "$stub" \
    && ln -sf "$stub" "/usr/lib/$(uname -m)-linux-gnu/libcuda.so" \
    && ln -sf "$stub" "/usr/lib/$(uname -m)-linux-gnu/libcuda.so.1" \
    && ldconfig
RUN git clone --filter=blob:none "${LLAMA_REPO}" /src \
    && cd /src \
    && git checkout --quiet "${LLAMA_REF}"
RUN cmake -S /src -B /src/build -G Ninja \
      -DCMAKE_BUILD_TYPE=Release \
      -DGGML_CUDA=ON \
      -DGGML_CUDA_FA=ON \
      -DGGML_CUDA_COMPRESSION_MODE=size \
      -DCMAKE_CUDA_ARCHITECTURES=${CUDA_ARCH} \
      -DLLAMA_CURL=OFF \
      -DLLAMA_BUILD_TESTS=OFF \
    && cmake --build /src/build --target llama-server llama-bench -j"$(nproc)"

FROM ${CUDA_RUNTIME}
RUN apt-get update && apt-get install -y --no-install-recommends \
      python3 libgomp1 ca-certificates curl \
    && rm -rf /var/lib/apt/lists/*
COPY --from=build /src/build/bin/llama-server /usr/local/bin/llama-server
COPY --from=build /src/build/bin/llama-bench /usr/local/bin/llama-bench
COPY --from=build /src/build/bin/*.so* /usr/local/lib/
RUN ldconfig
COPY server/ /opt/bjev/
COPY entrypoint.sh /entrypoint.sh
RUN chmod +x /entrypoint.sh && python3 -m py_compile /opt/bjev/bjev_server.py

# 8010 llama.cpp, 8011 decisions
EXPOSE 8010 8011
ENTRYPOINT ["/entrypoint.sh"]
