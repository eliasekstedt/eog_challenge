
import torch
import torch.nn as nn

class RMSELoss(nn.Module):
    def __init__(self, eps=1e-6):
        super().__init__()
        self.mse = nn.MSELoss()
        self.eps = eps
        
    def forward(self,yhat,y):
        loss = torch.sqrt(self.mse(yhat,y) + self.eps)
        return loss
    
class CustomLoss(nn.Module):
    def __init__(self, eps=1e-6):
        super().__init__()
        self.mse = nn.MSELoss()
        self.eps = eps
        
    def forward(self,preds,y):
        #loss = torch.sqrt(torch.mean(y * (y - preds)**2 + self.eps))
        #loss = torch.sqrt(self.mse(preds,y) + self.eps)*(1+(y>0))
        squared_diff = (preds - y) ** 2
        weights = torch.where(y > 0, 
                            torch.tensor(2).to(y.to('cuda:0')), 
                            torch.tensor(1.0).to(y.device))
        weighted_mse = torch.mean(weights * squared_diff)
        return torch.sqrt(weighted_mse)


