import torch
from smt_model.configuration_smt import SMTConfig
from smt_model.architectures.builder import build_model

w2i = {'<pad>': 0, '<bos>': 1, '<eos>': 2, 'a': 3, 'b': 4}
i2w = {v: k for k, v in w2i.items()}

config = SMTConfig(maxh=256, maxw=256, maxlen=10, out_categories=len(w2i), padding_token=0, w2i=w2i, i2w=i2w)

model = build_model(config, arch_type="qwen")

print("Created model")
x = torch.randn(2, 1, 128, 128)
di = torch.tensor([[1, 3, 4, 0, 0], [1, 4, 3, 2, 0]])
labels = torch.tensor([[3, 4, 2, 0, 0], [4, 3, 2, 0, 0]])

output = model(encoder_input=x, decoder_input=di, labels=labels)
print("Forward success. Loss:", output.loss)

preds, pred_out = model.predict(x, convert_to_str=True)
print("Predict success. Preds:", preds)
