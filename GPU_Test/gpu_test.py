import torch 

print("The number of cuda device: ", torch.cuda.device_count())

devices = [torch.device(f'cuda:{i}') for i in range(torch.cuda.device_count())]

