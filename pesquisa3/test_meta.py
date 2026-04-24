import torch
import torch.nn as nn

class MyModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.linear = nn.Linear(10, 10)
        self.emb = nn.Embedding(10, 10)

with torch.device('meta'):
    model = MyModel()

state_dict = {
    'linear.weight': torch.randn(10, 10),
    'linear.bias': torch.randn(10),
    'emb.weight': torch.randn(10, 10)
}

try:
    model.load_state_dict(state_dict, assign=True)
    print("assign=True WORKS")
except Exception as e:
    print("FAILED:", e)
