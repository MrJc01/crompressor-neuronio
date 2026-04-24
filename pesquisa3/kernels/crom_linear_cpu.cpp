#include <torch/extension.h>
#include <vector>
#include <omp.h>

// Kernel CPU C++ ultra otimizado para CromLinear
// Faz a operação Y = X @ W_quantized sem inflar os tensores.
// Utiliza OpenMP para multi-threading e #pragma omp simd para vetorização AVX.

torch::Tensor crom_gemv_cpu(
    torch::Tensor x,          // [B, in_features] ou [in_features] (FP32)
    torch::Tensor codebook,   // [K, D] (FP32 - convertido no Python para evitar gargalo FP16 na CPU)
    torch::Tensor indices,    // [n_blocks] -> out_features * (in_features/D) (INT16)
    int out_features,
    int in_features,
    int D
) {
    // Validações
    TORCH_CHECK(x.is_contiguous(), "x must be contiguous");
    TORCH_CHECK(codebook.is_contiguous(), "codebook must be contiguous");
    TORCH_CHECK(indices.is_contiguous(), "indices must be contiguous");
    TORCH_CHECK(x.dtype() == torch::kFloat32, "x must be float32");
    TORCH_CHECK(codebook.dtype() == torch::kFloat32, "codebook must be float32 for fast CPU math");
    TORCH_CHECK(indices.dtype() == torch::kInt16, "indices must be int16");

    // Lida com shapes flexíveis (1D, 2D, 3D)
    auto x_flat = x.view({-1, in_features}).contiguous(); 
    int B = x_flat.size(0);
    
    // Output tensor (sempre alocado em FP32)
    auto y = torch::zeros({B, out_features}, x.options().dtype(torch::kFloat32));
    
    // Pointers diretos para a memória crua
    const float* x_ptr = x_flat.data_ptr<float>();
    float* y_ptr = y.data_ptr<float>();
    const int16_t* idx_ptr = indices.data_ptr<int16_t>();
    const float* cb_ptr = codebook.data_ptr<float>();
    
    #pragma omp parallel for collapse(2)
    for (int b = 0; b < B; b++) {
        for (int r = 0; r < out_features; r++) {
            float sum = 0.0f;
            
            // Pointer para o início dos índices desta linha
            const int16_t* row_indices = idx_ptr + (r * num_blocks_per_row);
            const float* current_x = x_ptr + (b * in_features);
            
            // Loop principal (Blocos)
            for (int c_block = 0; c_block < num_blocks_per_row; c_block++) {
                int16_t code_idx = row_indices[c_block];
                const float* code = cb_ptr + (code_idx * D);
                const float* x_block = current_x + (c_block * D);
                
                // Otimização extrema: Unroll manual para D=8 (Valor fixo da Pesquisa 3)
                // Isso evita overhead de branches dinâmicos e força a vetorização.
                if (D == 8) {
                    sum += code[0]*x_block[0] + code[1]*x_block[1] + 
                           code[2]*x_block[2] + code[3]*x_block[3] +
                           code[4]*x_block[4] + code[5]*x_block[5] + 
                           code[6]*x_block[6] + code[7]*x_block[7];
                } else {
                    for (int d = 0; d < D; d++) {
                        sum += code[d] * x_block[d];
                    }
                }
            }
            
            y_ptr[b * out_features + r] = sum;
        }
    }
    
    // Restaura o shape original do PyTorch
    if (x.dim() == 1) {
        return y.view({out_features});
    } else if (x.dim() == 3) {
        return y.view({x.size(0), x.size(1), out_features});
    }
    return y;
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("gemv", &crom_gemv_cpu, "CromLinear GEMV (CPU OpenMP/SIMD)");
}
