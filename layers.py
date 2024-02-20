import torch
import torch.nn as nn


class FFLinearLayer(nn.Linear):
    """
    A simple class for the forward forward layer .
    """

    def __init__(self, in_features, out_features, num_epoch, threshold=6.5, device="cpu", bias=True):
        super().__init__(in_features, out_features, bias, device)
        self.relu = nn.ReLU()
        self.opti = torch.optim.Adam(self.parameters(), lr=0.001)
        self.loss = nn.CrossEntropyLoss()
        self.num_epoch = num_epoch
        self.threshold = threshold

    def forward(self, x):
        """
            Perform a forward pass on the input data.
            perform a layer-wise batch normalization and then apply the ReLU activation function for the forward pass
            Math: y = ReLU(xW + b)
        :param x: the input data
        :return: the output data
        """
        # print(x.shape, self.weight.shape, self.bias.shape)
        x_ = x / (x.norm(2, 1, keepdim=True) + 1e-4)
        # print(x_.shape, self.weight.shape, self.weight.T.shape, self.bias.shape, self.bias.unsqueeze(0).shape,
        #       self.bias.unsqueeze(0).T.shape)

        return self.relu(torch.mm(x_, self.weight.T) + self.bias.unsqueeze(0))
        # return self.relu(torch.mm(x_, self.weight) + self.bias.view(1, -1))

    # @staticmethod
    def goodness_score(self, x_positive, x_negative):
        """
            compute the goodness score, meaning square up the activation of each neuron in the layer and square them up.
            Math: sum_{activations}^2
        :arg x_positive: the positive sample
        :arg x_negative: the negative sample
        :return: the positive and negative goodness score
        """
        positive_goodness = self.forward(x_positive).pow(2).mean(1)
        negative_goodness = self.forward(x_negative).pow(2).mean(1)
        return positive_goodness, negative_goodness

    def goodness_loss(self, positive_goodness, negative_goodness, sigmoid=True):
        """
            Compute the goodness loss, this is the loss function for the goodness score.
            Math: L = sigmoid(-goodness_positive + threshold) + sigmoid(goodness_negative - threshold)
        :param positive_goodness: the positive goodness score
        :param negative_goodness: the negative goodness score
        :param sigmoid: whether to use the sigmoid function or not
        :return: the goodness loss
        """
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


# Instantiate the FFLinearLayer
layer = FFLinearLayer(in_features=1000, out_features=2, num_epoch=10)

# Generate random input data (adjusted for the input feature as needed)
input_data_p = torch.randn(10, 1000)  # Example: 10 samples, 1000 features
input_data_n = torch.randn(10, 1000)  # Example: 10 samples, 1000 features


# Forward pass, for the positive and negative samples
output_p = layer(input_data_p)
output_n = layer(input_data_n)

print("Output_p shape:", output_p.shape)  # Shape Should be (data_sample, output_features)
print("Output_n shape:", output_n.shape)  # Shape Should be (data_sample, output_features)

print("Training the layer...", layer.train(input_data_p, input_data_n))

