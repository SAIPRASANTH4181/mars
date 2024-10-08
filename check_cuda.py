import torch
device = None
if device is None:
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
device_a = torch.device(device)
print(device_a)