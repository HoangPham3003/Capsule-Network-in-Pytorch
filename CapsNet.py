import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable


class DigitCapsLayer(nn.Module):
    def __init__(self, n_caps=10, n_dims=16, n_routing_iterations=3, device=None):
        super(DigitCapsLayer, self).__init__()

        self.device = device

        self.n_caps = n_caps
        self.n_dims = n_dims
        self.n_routing_iterations = n_routing_iterations

        # init weight matrix W
        self.routing_weights = nn.Parameter(torch.randn(1, 32*6*6, n_caps, n_dims, 8))


    def squash(self, tensor=None, dim=-1, epsilon=1e-7):
        squared_norm = (tensor ** 2).sum(dim=dim, keepdim=True)
        safe_norm = torch.sqrt(squared_norm + epsilon)
        squash_factor = squared_norm / (1. + squared_norm)
        unit_vector = tensor / safe_norm
        return squash_factor * unit_vector


    def forward(self, x):
        # x: [Batch_size, 1152, 8]

        # create W and x
        batch_size = x.size(dim=0)

        W = self.routing_weights.repeat(batch_size, 1, 1, 1, 1) # [BS, 1152, 10, 16, 8]
        
        x = torch.unsqueeze(x, dim=-1)
        x = torch.unsqueeze(x, dim=2)
        x = x.repeat(1, 1, 10, 1, 1) # [BS, 1152, 10, 8, 1]

        # Compute the Predicted Output Vectors (u_hat[j|i])
        prediction_vectors = W @ x # [BS, 1152, 10, 16, 1]

        # Routing by agreement
       
        # Initial logits b[i, j]
        initial_logits = Variable(torch.zeros(batch_size, 32*6*6, self.n_caps, 1, 1)).to(self.device) # [BS, 1152, 10, 1, 1]
        
        for i in range(self.n_routing_iterations):
            # The coupling coefficients c[i] = softmax(b[i])
            coupling_coefs = F.softmax(initial_logits, dim=2)
            
            outputs = (coupling_coefs * prediction_vectors).sum(dim=1, keepdims=True) # [BS, 1, 10, 16, 1]
            outputs = self.squash(tensor=outputs, dim=-2)

            if i != self.n_routing_iterations - 1:
                outputs = outputs.repeat(1, 32*6*6, 1, 1, 1)
                agreement = torch.transpose(prediction_vectors, -2, -1) @ outputs # [BS, 1152, 10, 1, 1]
                initial_logits += agreement 
        
        return outputs
    

class CapsNet(nn.Module):
    def __init__(self, n_routing_iterations=3, device=None):
        super(CapsNet, self).__init__()

        self.device = device

        # Layer 1: Convolution
        self.Conv1 = nn.Conv2d(in_channels=1, out_channels=256, kernel_size=9, stride=1, padding=0)

        # Layer 2: Primary Capsules
        self.Conv2 = nn.Conv2d(in_channels=256, out_channels=256, kernel_size=9, stride=2, padding=0)

        # Layer 3: Digit Capsules
        self.digitcaps = DigitCapsLayer(n_routing_iterations=n_routing_iterations, device=device)

        # Reconstruction
        self.decoder = nn.Sequential(
            nn.Linear(16*10, 512),
            nn.ReLU(),
            nn.Linear(512, 1024),
            nn.ReLU(),
            nn.Linear(1024, 28*28),
            nn.Sigmoid()
        )

    def squash(self, tensor=None, dim=-1, epsilon=1e-7):
        squared_norm = (tensor ** 2).sum(dim=dim, keepdim=True)
        safe_norm = torch.sqrt(squared_norm + epsilon)
        squash_factor = squared_norm / (1. + squared_norm)
        unit_vector = tensor / safe_norm
        return squash_factor * unit_vector


    def safe_norm(self, tensor=None, dim=-1, epsilon=1e-7, keep_dims=False):
        squared_norm = (tensor ** 2).sum(dim=dim, keepdim=keep_dims)
        return torch.sqrt(squared_norm + epsilon)


    def forward(self, x, y=None):
        # 1st layer
        x = self.Conv1(x)
        x = F.relu(x, inplace=True)

        # primary capsules
        x = self.Conv2(x)
        x = F.relu(x, inplace=True)
        x = torch.reshape(x, (-1, 32*6*6, 8))
        x = self.squash(tensor=x)
        
        # digit capsules
        caps_2_output = self.digitcaps(x) # [BS, 1, 10, 16, 1]

        # reconstruction

        # find longest vector
        y_proba = self.safe_norm(caps_2_output, dim=-2)
        y_proba_argmax = torch.argmax(y_proba, dim=2)
        y_pred = torch.squeeze(y_proba_argmax, dim=2)
        y_pred = torch.squeeze(y_pred, dim=1)
        
        mask = None
        if y is None:
            mask = torch.eye(10).to(self.device).index_select(dim=0, index=y_pred)
        else:
            mask = torch.eye(10).to(self.device).index_select(dim=0, index=y)
        

        reconstruction_mask_reshaped = torch.reshape(mask, [-1, 1, 10, 1, 1])
        
        caps2_output_masked = torch.multiply(caps_2_output, reconstruction_mask_reshaped)

        decoder_input = torch.reshape(caps2_output_masked, [-1, 10 * 16])

        reconstructions = self.decoder(decoder_input)

        return caps_2_output, y_pred, reconstructions