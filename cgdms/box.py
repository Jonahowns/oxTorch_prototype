import torch
import torch.nn.functional as F



class Box():
    def __init__(self, box_size, device='cpu'):
        self.shape = torch.tensor([box_size, box_size, box_size], device=device)

    def center_system(self, coords, masses):
        center = self.shape/2
        current_com = torch.mean(self.coords*F.normalize(self.massses))
        self.coords.add_(center - current_com)