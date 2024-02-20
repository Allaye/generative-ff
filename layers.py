import torch
import torch.nn as nn


class FFLinearLayer(nn.Linear):
    def __init__(self, in_features, out_features, num_epoch, device="cpu", bias=True):
        super().__init__(in_features, out_features, bias, device)
        self.relu = nn.ReLU()
        self.opti = torch.optim.Adam(self.parameters(), lr=0.001)
        self.loss = nn.CrossEntropyLoss()
        self.num_epoch = num_epoch
        self.threshold = 2.0

    def forward(self, x):
        # perform a layer-wise batch normalization and then apply the ReLU activation function for the forward pass
        x_ = x / (x.norm(2, 1, keepdim=True) + 1e-4)
        return self.relu(torch.mm(x_, self.weight.T) + self.bias.unsqueeze(0))

    # @staticmethod
    def goodness_score(self, x_positive, x_negative):
        """
            compute the goodness score.
            Math: \sum_{y}^2
        """
        positive_goodness = self.forward(x_positive).pow(2).mean(1)
        negative_goodness = self.forward(x_negative).pow(2).mean(1)
        return positive_goodness, negative_goodness

    def goodness_loss(self, positive_goodness, negative_goodness, sigmoid=True):
        errors = torch.cat([-positive_goodness + self.threshold, negative_goodness - self.threshold])
        loss = torch.sigmoid(errors).mean() if sigmoid else torch.log(1 + torch.exp(errors)).mean()
        return loss

    def train(self, x_positive, x_negative):
        # the forward forward paradigm happens here
        for epoch in range(self.num_epoch):
            # perform a forward pass and compute the goodness score
            positive_goodness, negative_goodness = self.goodness_score(x_positive, x_negative)
            # compute the goodness loss with respect to the goodness score and the threshold
            loss = self.goodness_loss(positive_goodness, negative_goodness)
            # empty the gradient perform a backward pass(local descent) and update the weights and biases
            self.opti.zero_grad()
            loss.backward()
            self.opti.step()
        return self.forward(x_positive).detach(), self.forward(x_negative).detach()


class FFConvLayer(nn.Conv2d):
    def __init__(self):
        super().__init__()
        pass

    def forward(self, x):
        pass


ff = FFLinearLayer(10, 10, 10)
print(ff)
data_p = torch.randn(10, 10)
data_n = torch.randn(10, 10)
output_p = ff(data_p)
output_n = ff(data_n)
print(output_p)
print(output_n)
print(ff.train(data_p, data_n))
# <class '__main__.FFLinearLayer'>
