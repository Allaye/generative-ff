import cv2, random
import torch
import wandb
from discriminators_ff import *
from utils import *
from dataloader import CustomDataset
from generator_ff import *
import matplotlib.pyplot as plt

#
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# def MNIST_loaders(train_batch_size=50000, test_batch_size=10000):
#     transform = Compose([
#         ToTensor(),
#         Normalize((0.1307,), (0.3081,)),
#         Lambda(lambda x: torch.flatten(x))])
#
#     train_loader = DataLoader(
#         MNIST('./data/', train=True,
#               download=True,
#               transform=transform),
#         batch_size=train_batch_size, shuffle=True)
#
#     test_loader = DataLoader(
#         MNIST('./data/', train=False,
#               download=True,
#               transform=transform),
#         batch_size=test_batch_size, shuffle=False)
#
#     return train_loader, test_loader
#

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

if __name__ == "__main__":
    torch.manual_seed(1234)
    dataset = CustomDataset('data/facades/test/')
    dataloader = dataset.load_data(dataset, batch_size=1, shuffle=True)
    generator = FFConvGenerator(in_channels=3, out_channels=64)
    num_epochs = 20
    for i, (grand_truth_img, gen_img) in enumerate(dataloader):
        # print(data.shape, label.shape)
        # plt.imshow(data[0].reshape(256, 256, 3))
        # plt.show(block=True)
        if i == 5:
            break
        generator.train(grand_truth_img, gen_img, 30)
        generated_image = generator(grand_truth_img)
        print(generated_image.shape)
        image_array = generated_image.squeeze().permute(1, 2, 0).detach().numpy()

        # Display the image
        # plt.imshow(image_array)
        # plt.axis('off')  # Hide axes
        # plt.show()
    # x = torch.randn((1, 3, 256, 256))
    # generated_image = generator(x)



    # dataa = generator.initial_down(x)
    # print(generator.initial_down[0].forward_forward_trad)
    # print(dataa)
    # for idx, (negative, positive) in enumerate(dataloader):
    #     pass

#     train_loader, test_loader = MNIST_loaders()
#
#     net1 = FFDenseDiscriminator([784, 784, 500, 500, 500, 500], 1000, 10)
#     # net = FFConvDiscriminator([1, 32, 32]).cuda()
#     # xx, yy = next(iter(XX))
#     # print(xx[0].shape, xx[1].shape)
#
#     x, y = next(iter(train_loader))
#     # # print(x.shape, y[0])
#     x, y = x.cuda(), y.cuda()
#     # print('positive input data shape:', x.shape, y.shape)
#     x_pos = overlay_y_on_x(x, y)
#     rnd = torch.randperm(x.size(0))
#     # print('rnd:', rnd.shape)
#     x_neg = overlay_y_on_x(x, y[rnd])
#     # for data, name in zip([x, x_pos, x_neg], ['orig', 'pos', 'neg']):
#     #     visualize_sample(data, name)
#     # x_pos = x_pos.reshape(-1, 1, 28, 28)
#     # x_neg = x_neg.reshape(-1, 1, 28, 28)
#     # net1.train(x_pos, x_neg)
#     # x = x.reshape(-1, 1, 28, 28)
#     # print('data shape:', x.shape, y.shape)
#     net1.train(x_pos, x_neg)
#     print('predicted train labels:', net1.predict(x))
#     print('train error log :', 1.0 - net1.predict(x).eq(y).float().mean().item())
#     print('true train labels:', y)
#
#     # print('train error:', 1.0 - net.predict(x).eq(y).float().mean().item())
#
#     x_te, y_te = next(iter(test_loader))
#     x_te, y_te = x_te.cuda(), y_te.cuda()
#     # x_te = x_te.reshape(-1, 1, 28, 28)
#
#     # select first 10 samples
#     # x_te = x_te
#     # y_te = y_te
#     print('predicted test labels:', net1.predict(x_te))
#     print('test error log :', 1 - net1.predict(x_te).eq(y_te).float().mean().item())
#     print('true test labels:', y_te)
# print('test error:', 1.0 - net.predict(x_te).eq(y_te).float().mean().item())
# loop over the samples
# for i in range(10):
#     visualize_sample(x_te[i], 'test sample')
# for data, name in zip([x, x_pos, x_neg], ['orig', 'pos', 'neg']):
#     visualize_sample(data, name)

#
#########################################################################
# if __name__ == '__main__':
#     # start a new wandb run to track this script
#     wandb.init(
#         entity="shu-three-stars",
#         # set the wandb project where this run will be logged
#         project="ff-research",
#         # track hyperparameters and run metadata
#         config={
#             "learning_rate": 0.03,
#             "threshold": 2.0,
#             "architecture": "forward-forward dense discriminator with traditional training loop",
#             "dataset": " MNIST",
#             "epochs": 1000,
#         }
#     )
#     train_loader, test_loader = MNIST_loaders()
#
#     net1 = FFDenseDiscriminator([28*28, 784, 500, 500, 500, 500], 1000, 10)
#     # net = FFConvDiscriminator([1, 32, 32]).cuda()
#     # xx, yy = next(iter(XX))
#     # print(xx[0].shape, xx[1].shape)
#
#     x, y = next(iter(train_loader))
#     # # print(x.shape, y[0])
#     x, y = x.cuda(), y.cuda()
#     # print('positive input data shape:', x.shape, y.shape)
#     x_pos = overlay_y_on_x(x, y)
#     rnd = torch.randperm(x.size(0))
#     # print('rnd:', rnd.shape)
#     x_neg = overlay_y_on_x(x, y[rnd])
#
#     net1.train(x_pos, x_neg, wandb, net1, x, y)
#     print('predicted train labels:', net1.predict(x))
#     print('train error log :', 1.0 - net1.predict(x).eq(y).float().mean().item())
#     print('true train labels:', y)
#     wandb.log({"test error": 1 - net1.predict(x).eq(y).float().mean().item()})
#
#     # print('train error:', 1.0 - net.predict(x).eq(y).float().mean().item())
#
#     x_te, y_te = next(iter(test_loader))
#     x_te, y_te = x_te.cuda(), y_te.cuda()
#     # x_te = x_te.reshape(-1, 1, 28, 28)
#
#     # select first 10 samples
#     # x_te = x_te
#     # y_te = y_te
#     print('predicted test labels:', net1.predict(x_te))
#     print('test error log :', 1 - net1.predict(x_te).eq(y_te).float().mean().item())
#     print('true test labels:', y_te)
#     wandb.log({"test error": 1 - net1.predict(x_te).eq(y_te).float().mean().item()})
# wandb.finish()

#
#     torch.manual_seed(1234)
#     dataset = CustomDisDataset('data/facades/test/')
#     data = dataset.load_data(dataset, batch_size=32, shuffle=True)
#     #     train_loader, test_loader = MNIST_loaders()
#     #     # net = FFConvDiscriminator([1, 6, 16, 120]).cuda()
#     real, fake = next(iter(data))
#     #     X, Y = next(iter(train_loader))
#     #     x, y = X.cuda(), Y.cuda()
#     #     x_pos = overlay_y_on_x(x, y)
#     real_label, fake_label = generate_label(real, 1), generate_label(fake, 0)
#     #     # real, fake = real.cuda(), fake.cuda()
#     #     # real_label, fake_label = real_label.cuda(), fake_label.cuda()
#     positive_real = overlay_y_on_x1(real, real_label)
#     positive_fake = overlay_y_on_x1(fake, fake_label)
#     #     # merge the positive_real and positive_fake into one tensor called positive data
#     #     # print(positive_real.shape, positive_fake.shape)
#     #     # positive_data = torch.concat([positive_real, positive_fake], 0)
#     data = positive_real[0].permute(1, 2, 0)
#     reshaped = data.cpu().reshape(256, 256, 3).numpy()
#     reshaped = cv.cvtColor(reshaped, cv2.COLOR_BGR2GRAY)
#     #     #
#     #     plt.figure(figsize=(4, 4))
#     #     plt.subplot(1, 2, 1)
#     plt.title('name')
#     plt.imshow(reshaped, cmap="gray")
#     plt.show()
# #     # print(x_pos[0].shape, x_pos.shape)
# #     reshaped = x_pos[0].cpu().reshape(28, 28)
# #     plt.figure(figsize=(4, 4))
# #     plt.title('name')
# #     plt.imshow(reshaped, cmap="gray")
# #     plt.show()
# #     # for data, name in zip([fake, real, positive_real, positive_fake], ['fake', 'real', 'positive', 'negative']):
# #     #     # visualize_sample(data, name)
# #     #     data = data[10].permute(1, 2, 0)
# #     #     reshaped = data.cpu().reshape(256, 256, 3).numpy()
# #     #     #
# #     #     # # plt.figure(figsize=(5, 5))
# #     #     # plt.subplot(1, 2, 1)
# #     #     plt.title(name)
# #     #     plt.imshow(cv.cvtColor(reshaped, cv.COLOR_BGR2RGB))
# #     #     plt.show()
