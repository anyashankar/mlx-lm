# moe418_ext: MLX C++/Metal extension skeleton

Copied from MLX v0.32.4 `examples/cmake_extension` (matches the installed MLX 0.32.4),
module renamed to `moe418_ext`. It still contains MLX's sample `axpby` primitive
(CPU + Metal); the fused MoE verify kernel will replace it.

    pip install -r requirements.txt
    python setup.py build_ext -j8 --inplace
    python test.py      # prints c_cpu correct: True / c_gpu correct: True
