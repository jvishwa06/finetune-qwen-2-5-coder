# Fine-tune Qwen2.5-Coder

A Google Colab notebook that fine-tunes `unsloth/Qwen2.5-Coder-3B` on the `Vezora/Tested-143k-Python-Alpaca` dataset with LoRA, using [Unsloth](https://github.com/unslothai/unsloth) and TRL.

## What the notebook does

1. **Load the model**: `FastLanguageModel.from_pretrained("unsloth/Qwen2.5-Coder-3B")` with a maximum sequence length of 2048.
2. **Add LoRA adapters**: rank 16, alpha 16, no dropout, on the attention and MLP projection layers (`q_proj`, `k_proj`, `v_proj`, `o_proj`, `gate_proj`, `up_proj`, `down_proj`), with Unsloth gradient checkpointing.
3. **Prepare data**: loads the dataset from Hugging Face and formats every row into an Alpaca-style prompt (instruction, input, response).
4. **Train**: `SFTTrainer` with batch size 2 per device, 4 gradient accumulation steps, learning rate 2e-4, the `adamw_8bit` optimiser and `max_steps = 60`.
5. **Test**: generates code from a prompt with a streamer.
6. **Save**: saves the adapter and tokenizer locally as `Qwen2.5-Coder-3B-143k-Python-Alpaca_model` and pushes them to the Hugging Face Hub repo `Vishu06/Qwen2.5-Coder-3B-143k-Python-Alpaca_model`.

## Run

Open `finetuneQwen2.5Coder.ipynb` in Google Colab with a GPU runtime and run the cells in order. The first cell installs Unsloth and the other dependencies. Pushing to the Hub needs your own Hugging Face token in the `push_to_hub` calls.
