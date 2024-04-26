"""
Code for research work for the layers of the forward-forward neural networks "
Date 19/02/2024.
Author: Kolade Gideon *Allaye*
"""
import torch
import torch.nn as nn


class FFLinearLayer(nn.Linear):
    """
    A simple class for the forward-forward layer .
    :param in_features: the number of input features
    :param out_features: the number of output features
    :param num_epoch: the number of epochs to train the layer
    :param threshold: the threshold for the goodness score
    :param device: the device to run the layer on
    :param bias: whether to use the bias or not
    """

    def __init__(self, in_features, out_features, num_epoch=1000, threshold=2.0, device="cuda", bias=True):
        super(FFLinearLayer, self).__init__(in_features, out_features, bias, device)
        self.relu = nn.ReLU()
        self.opti = torch.optim.Adam(self.parameters(), lr=0.03)
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

    def goodness_loss(self, positive_goodness, negative_goodness, sigmoid=False):
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

    def forward_forward(self, x_positive, x_negative):
        # the forward-forward paradigm happens here
        for epoch in range(self.num_epoch):
            # perform a forward pass and compute the goodness score
            positive_goodness, negative_goodness = self.goodness_score(x_positive, x_negative)
            # compute the goodness loss with respect to the goodness score and the threshold
            loss = self.goodness_loss(positive_goodness, negative_goodness)
            # empty the gradient perform a backward pass(local descent) and update the weights and biases
            self.opti.zero_grad()
            loss.backward()
            self.opti.step()
            print(f"Epoch {epoch + 1}: Training Loss {loss.item()}")
        return loss.item(), self.forward(x_positive).detach(), self.forward(x_negative).detach()

    def forward_forward_trad(self, x_pos, x_neg):
        g_pos = self.forward(x_pos).pow(2).mean(1)
        g_neg = self.forward(x_neg).pow(2).mean(1)
        # The following loss pushes pos (neg) samples to
        # values larger (smaller) than the self.threshold.
        loss = torch.log(1 + torch.exp(torch.cat([
            -g_pos + self.threshold,
            g_neg - self.threshold]))).mean()
        self.opti.zero_grad()
        # this backward just compute the derivative and hence
        # is not considered backpropagation.
        loss.backward()
        self.opti.step()
        print(f"Training Loss {loss.item()}")
        return loss.item(), self.forward(x_pos).detach(), self.forward(x_neg).detach()


class FFConvTransLayer(nn.ConvTranspose2d):
    def __init__(self, in_channels, out_channels, kernel_size, stride=1, padding=0, num_epoch=100, threshold=2.0,
                 drop=False, droprate=0.5, bias=False, padding_mode="reflect", act="relu", norm=True
                 ):
        super(FFConvTransLayer, self).__init__(in_channels, out_channels, kernel_size, stride, padding, bias=bias)
        # Initialize weights using Xavier/Glorot initialization
        nn.init.xavier_uniform_(self.weight)
        # Initialize biases to zeros
        # nn.init.zeros_(self.bias)
        self.drop = drop
        self.norm = norm
        self.dropout = nn.Dropout2d(droprate)
        self.threshold = threshold
        self.num_epoch = num_epoch
        self.opti = torch.optim.Adam(self.parameters(), lr=0.03)
        self.batch_norm = nn.BatchNorm2d(out_channels)
        self.acti = nn.ReLU() if act == "relu" else nn.Tanh()

        # self.pool = nn.MaxPool2d(2, 2)

    def forward(self, input):
        """
            Perform a forward pass on the input data.
            perform a layer-wise-batch normalization and then apply the ReLU activation function for the forward pass

        :param input: the input data
        :return: the output data
        """
        # print(x.shape, self.weight.shape, self.bias.shape)
        input_ = input / (input.norm(2, 1, keepdim=True) + 1e-4)
        #
        # Perform the convolution
        # print(input_.shape, self.weight.shape, self.bias.shape)
        print('shape inside transpose', input_.shape)
        output = super(FFConvTransLayer, self).forward(input_)
        # Perform batch normalization
        if self.norm:
            output = self.batch_norm(output)
        # Perform the ReLU activation
        output = self.acti(output)
        if self.drop:
            output = self.dropout(output)
        # Perform the max pooling
        # output = self.pool(output)
        return output
        # return self.relu(self._conv_forward(input_, self.weight.cuda(), self.bias.cuda()))
        # return self._conv_forward(input_, self.weight, self.bias)

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
        # errors = torch.cat([-positive_goodness + self.threshold, negative_goodness - self.threshold])
        # loss = torch.sigmoid(errors).mean() if sigmoid else torch.log(1 + torch.exp(errors)).mean()
        loss = torch.log(1 + torch.exp(torch.cat([-positive_goodness + self.threshold,
                                                  negative_goodness - self.threshold], 0)))
        return loss

    # def forward_forward(self, x_positive, x_negative):
    #     # the forward-forward paradigm happens here
    #     for epoch in range(self.num_epoch):
    #         # perform a forward pass and compute the goodness score
    #         # print('....', x_positive.shape, x_negative.shape, '....')
    #         positive_goodness, negative_goodness = self.goodness_score(x_positive, x_negative)
    #         # print('goodness', positive_goodness.shape, negative_goodness.shape, 'goddness')
    #         # compute the goodness loss with respect to the goodness score and the threshold
    #         loss = self.goodness_loss(positive_goodness, negative_goodness)
    #         # print('loss', loss.shape, 'loss', loss.mean(), loss.sum(), loss)
    #         # empty the gradient perform a backward pass(local descent) and update the weights and biases
    #         self.opti.zero_grad()
    #         print('loss:', loss.mean())
    #         # loss.backward() expects a scalar loss value, this implementation uses 2 losses so we need to sum or
    #         # compute the mean of the loss
    #         loss.mean().backward()
    #         self.opti.step()
    #         print('epoch:', epoch, 'loss:', loss.mean())
    #     return self.forward(x_positive).detach(), self.forward(x_negative).detach()

    def forward_forward_trad(self, x_pos, x_neg):
        g_pos = self.forward(x_pos).pow(2).mean(1)
        g_neg = self.forward(x_neg).pow(2).mean(1)
        # The following loss pushes pos (neg) samples to
        # values larger (smaller) than the self.threshold.
        loss = torch.log(1 + torch.exp(torch.cat([
            -g_pos + self.threshold,
            g_neg - self.threshold]))).mean()
        self.opti.zero_grad()
        # this backward just compute the derivative and hence
        # is not considered backpropagation.
        loss.backward()
        self.opti.step()
        print(f"Training Loss {loss.item()}")
        return loss.item(), self.forward(x_pos).detach(), self.forward(x_neg).detach()


class FFConvLayer(nn.Conv2d):
    def __init__(self, in_channels, out_channels, kernel_size, stride=1, padding=0, num_epoch=100, threshold=2.0,
                 drop=False, droprate=0.5, bias=False, padding_mode="reflect", init=False, act="relu", norm=True
                 ):
        super(FFConvLayer, self).__init__(in_channels, out_channels, kernel_size, stride, padding, bias=bias,
                                          padding_mode=padding_mode)
        # Initialize weights using Xavier/Glorot initialization
        nn.init.xavier_uniform_(self.weight)
        # Initialize biases to zeros
        # nn.init.zeros_(self.bias)
        self.drop = drop
        self.norm = norm
        self.threshold = threshold
        self.num_epoch = num_epoch
        self.opti = torch.optim.Adam(self.parameters(), lr=0.03)
        self.init = init
        if self.init:
            self.relu = nn.LeakyReLU(0.2)
        else:
            self.relu = nn.ReLU() if act == "relu" else nn.LeakyReLU(0.2)
            self.batch_norm = nn.BatchNorm2d(out_channels)
        self.dropout = nn.Dropout2d(droprate)
        # self.pool = nn.MaxPool2d(2, 2)

    def forward(self, input):
        """
            Perform a forward pass on the input data.
            perform a layer-wise-batch normalization and then apply the ReLU activation function for the forward pass

        :param input: the input data
        :return: the output data
        """
        # print(x.shape, self.weight.shape, self.bias.shape)
        input_ = input / (input.norm(2, 1, keepdim=True) + 1e-4)
        #

        # Perform the convolution
        # print(input_.shape, self.weight.shape, self.bias.shape)
        # pass the input data through the convolutional layer
        output = super(FFConvLayer, self).forward(input_)
        # Perform batch normalization
        if self.norm is True:
            output = self.batch_norm(output)
        # Perform the ReLU activation
        output = self.relu(output)
        # Calculate mean and variance of the kernel
        if self.drop:
            output = self.dropout(output)
        # Perform the max pooling
        # output = self.pool(output)
        return output
        # return self.relu(self._conv_forward(input_, self.weight.cuda(), self.bias.cuda()))
        # return self._conv_forward(input_, self.weight, self.bias)

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
        # errors = torch.cat([-positive_goodness + self.threshold, negative_goodness - self.threshold])
        # loss = torch.sigmoid(errors).mean() if sigmoid else torch.log(1 + torch.exp(errors)).mean()
        loss = torch.log(1 + torch.exp(torch.cat([-positive_goodness + self.threshold,
                                                  negative_goodness - self.threshold], 0)))
        return loss

    # def forward_forward(self, x_positive, x_negative):
    #     # the forward-forward paradigm happens here
    #     for epoch in range(self.num_epoch):
    #         # perform a forward pass and compute the goodness score
    #         # print('....', x_positive.shape, x_negative.shape, '....')
    #         positive_goodness, negative_goodness = self.goodness_score(x_positive, x_negative)
    #         # print('goodness', positive_goodness.shape, negative_goodness.shape, 'goddness')
    #         # compute the goodness loss with respect to the goodness score and the threshold
    #         loss = self.goodness_loss(positive_goodness, negative_goodness)
    #         # print('loss', loss.shape, 'loss', loss.mean(), loss.sum(), loss)
    #         # empty the gradient perform a backward pass(local descent) and update the weights and biases
    #         self.opti.zero_grad()
    #         print('loss:', loss.mean())
    #         # loss.backward() expects a scalar loss value, this implementation uses 2 losses so we need to sum or
    #         # compute the mean of the loss
    #         loss.mean().backward()
    #         self.opti.step()
    #         print('epoch:', epoch, 'loss:', loss.mean())
    #     return self.forward(x_positive).detach(), self.forward(x_negative).detach()

    def forward_forward_trad(self, x_pos, x_neg):
        g_pos = self.forward(x_pos).pow(2).mean(1)
        g_neg = self.forward(x_neg).pow(2).mean(1)
        # The following loss pushes pos (neg) samples to
        # values larger (smaller) than the self.threshold.
        loss = torch.log(1 + torch.exp(torch.cat([
            -g_pos + self.threshold,
            g_neg - self.threshold]))).mean()
        self.opti.zero_grad()
        # this backward just compute the derivative and hence
        # is not considered backpropagation.
        loss.backward()
        self.opti.step()
        print(f"Training Loss {loss.item()}")
        return loss.item(), self.forward(x_pos).detach(), self.forward(x_neg).detach()

# Instantiate the FFLinearLayer

# ffc = FFConvLayer(3, 64, 3, 5, True, 0.5, 1, 1)
# ffl = FFLinearLayer(1000, 2, 10)
# data = torch.randn(10, 3, 32, 32)
# print(ffc, ffl)
#
# layer = FFLinearLayer(in_features=1000, out_features=2, num_epoch=10)
#
# # Generate random input data (adjusted for the input feature as needed)
# input_data_p = torch.randn(10, 1000)  # Example: 10 samples, 1000 features
# input_data_n = torch.randn(10, 1000)  # Example: 10 samples, 1000 features
#
#
# # Forward pass, for the positive and negative samples
# output_p = layer(input_data_p)
# output_n = layer(input_data_n)
#
# print("Output_p shape:", output_p.shape)  # Shape Should be (data_sample, output_features)
# print("Output_n shape:", output_n.shape)  # Shape Should be (data_sample, output_features)
#
# print("Training the layer...", layer.forward_forward(input_data_p, input_data_n))

# x = torch.randn(10, 3, 32, 32)
# x_ = x / (x.norm(2, 1, keepdim=True) + 1e-4)
# print(x_, '.....')
# print(x)

# Instantiate the FFConvLayer
# layer = FFConvLayer(in_channels=3, out_channels=64, kernel_size=3, num_epoch=10, drop=True, droprate=0.5, threshold=1)
# data_p = torch.randn(10, 3, 32, 32)
# data_n = torch.randn(10, 3, 32, 32)
# # output_p = layer(data_p)
# # output_n = layer(data_n)
# print(layer)
#
# print("Output_p shape:", data_p.shape)  # Shape Should be (data_sample, output_features)
# print("Output_n shape:", data_n.shape)  # Shape Should be (data_sample, output_features)
# print("Training the layer...", layer.forward_forward(data_p, data_n))
