For pure architecture learning — no, you don't need `dataset.py` at all yet. You can feed dummy tensors directly into your model modules.

---

## Suggested order

**1. Finish the architecture in your notebook (where you are now)**
- Token embeddings
- Positional encoding
- Feed forward block
- Layer normalization
- Residual connection
- Encoder block → full Encoder
- Decoder block → full Decoder
- Projection layer
- Wire everything into a Transformer

Test each module with dummy tensors as you go.

**2. Then move to data pipeline**
- `datasets` library to load opus_books
- Build the tokenizer
- Understand how SOS/EOS/PAD are added
- Understand the masks

**3. Then wire it all together in training**
- DataLoader
- Loss function
- Training loop

---

The key insight is that `model.py` is completely self-contained — it only needs tensors of the right shape. So you can build and validate the entire architecture with `torch.randn(...)` before writing a single line of data pipeline code.