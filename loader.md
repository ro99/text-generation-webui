{
  "model_name": {
    "loader": "loader_type",
    "args": {
      "key1": "value1",
      "key2": "value2"
    },
    "settings": {
      "setting1": "value1",
      "setting2": "value2"
    }
  }
}

Detailed explanation of each field:

1. "model_name": The name of your model. This should match the name of the model folder in your `models` directory.

2. "loader": The type of loader to use. Options include:
   - "Transformers"
   - "llama.cpp"
   - "llamacpp_HF"
   - "ExLlamav2_HF"
   - "ExLlamav2"
   - "AutoGPTQ"
   - "HQQ"
   - "TensorRT-LLM"

3. "args": Loader-specific arguments. The available options depend on the chosen loader:

```
loaders_and_params = OrderedDict({
    'Transformers': [
        'cpu_memory',
        'gpu_memory',
        'load_in_8bit',
        'bf16',
        'cpu',
        'disk',
        'auto_devices',
        'load_in_4bit',
        'use_double_quant',
        'quant_type',
        'compute_dtype',
        'trust_remote_code',
        'no_use_fast',
        'use_flash_attention_2',
        'use_eager_attention',
        'alpha_value',
        'compress_pos_emb',
        'disable_exllama',
        'disable_exllamav2',
        'transformers_info',
    ],
    'llama.cpp': [
        'n_ctx',
        'n_gpu_layers',
        'cache_8bit',
        'cache_4bit',
        'tensor_split',
        'n_batch',
        'threads',
        'threads_batch',
        'no_mmap',
        'mlock',
        'no_mul_mat_q',
        'rope_freq_base',
        'compress_pos_emb',
        'cpu',
        'numa',
        'no_offload_kqv',
        'row_split',
        'tensorcores',
        'flash_attn',
        'streaming_llm',
        'attention_sink_size',
    ],
    'llamacpp_HF': [
        'n_ctx',
        'n_gpu_layers',
        'cache_8bit',
        'cache_4bit',
        'tensor_split',
        'n_batch',
        'threads',
        'threads_batch',
        'no_mmap',
        'mlock',
        'no_mul_mat_q',
        'rope_freq_base',
        'compress_pos_emb',
        'cpu',
        'numa',
        'cfg_cache',
        'trust_remote_code',
        'no_use_fast',
        'logits_all',
        'no_offload_kqv',
        'row_split',
        'tensorcores',
        'flash_attn',
        'streaming_llm',
        'attention_sink_size',
        'llamacpp_HF_info',
    ],
    'ExLlamav2_HF': [
        'gpu_split',
        'max_seq_len',
        'cfg_cache',
        'no_flash_attn',
        'no_xformers',
        'no_sdpa',
        'num_experts_per_token',
        'cache_8bit',
        'cache_4bit',
        'autosplit',
        'alpha_value',
        'compress_pos_emb',
        'trust_remote_code',
        'no_use_fast',
    ],
    'ExLlamav2': [
        'gpu_split',
        'max_seq_len',
        'no_flash_attn',
        'no_xformers',
        'no_sdpa',
        'num_experts_per_token',
        'cache_8bit',
        'cache_4bit',
        'autosplit',
        'alpha_value',
        'compress_pos_emb',
        'exllamav2_info',
    ],
    'AutoGPTQ': [
        'triton',
        'no_inject_fused_mlp',
        'no_use_cuda_fp16',
        'wbits',
        'groupsize',
        'desc_act',
        'disable_exllama',
        'disable_exllamav2',
        'gpu_memory',
        'cpu_memory',
        'cpu',
        'disk',
        'auto_devices',
        'trust_remote_code',
        'no_use_fast',
        'autogptq_info',
    ],
    'HQQ': [
        'hqq_backend',
        'trust_remote_code',
        'no_use_fast',
    ],
    'TensorRT-LLM': [
        'max_seq_len',
        'cpp_runner',
        'tensorrt_llm_info',
    ]
})
```

4. "settings": Additional settings for the model. Common options include:
   - "instruction_template": Specifies the instruction template to use. Options can be found in the `instruction-templates` folder. Examples include "Alpaca", "Vicuna", "OpenAssistant", etc.
   - "truncation_length": Maximum number of tokens to process.
   - "custom_stopping_strings": List of strings that will cause the model to stop generating.
   - "chat_template": Custom chat template for formatting messages.

   See the settings-template.yaml file for all available settings.

Example JSON configuration:

{
  "llama-7b": {
    "loader": "Transformers",
    "args": {
      "load_in_4bit": true,
      "use_double_quant": true,
      "gpu_memory": null,
      "cpu_memory": null
    },
    "settings": {
      "instruction_template": "Alpaca",
      "truncation_length": 2048,
      "custom_stopping_strings": ["Human:", "User:"]
    }
  },
  "mistral-7b-instruct-v0.1.Q4_K_M": {
    "loader": "llama.cpp",
    "args": {
      "n_gpu_layers": 35,
      "n_ctx": 4096,
      "n_batch": 512,
      "threads": 8
    },
    "settings": {
      "instruction_template": "Mistral",
      "truncation_length": 4096
    }
  },
  "gpt-j-6b-gptq": {
    "loader": "AutoGPTQ",
    "args": {
      "wbits": 4,
      "groupsize": 128,
      "model_type": "gptj"
    },
    "settings": {
      "instruction_template": "None",
      "truncation_length": 2048
    }
  }
}

Notes:
1. The exact arguments and settings available may vary depending on the specific version of the oobabooga project you're using.
2. Some arguments are boolean flags (e.g., "load_in_4bit", "use_double_quant"). Set these to true to enable them.
3. For numerical values (e.g., "n_gpu_layers", "n_ctx"), provide the appropriate integer or float.
4. For null values, use null without quotes.
5. The "instruction_template" setting should match one of the templates in the `instruction-templates` folder.
6. Always refer to the latest documentation or source code for the most up-to-date options and their correct usage.