import os

import torch
from torch import nn, optim
from torch.utils.data import DataLoader
from tqdm import tqdm
import os.path as osp
from SuperResolutionData import SRData
from Model.SRCNN import SRCNN
from torch.utils.tensorboard import SummaryWriter


# from transform import TrainTransform,TestTransfrom


def train():
    net = SRCNN()
    net = net.to(device)
    trainset = SRData(subset="train", transform=None)
    testset = SRData(subset="test", transform=None)
    trainloader = DataLoader(trainset, batch_size=batch_size, shuffle=True, num_workers=4)
    testloader = DataLoader(testset, batch_size=batch_size, shuffle=True, num_workers=4)
    criteron = nn.MSELoss()
    best_loss = 1e9

    # if osp.exists(sr_checkpoint):
    #     ckpt = torch.load(sr_checkpoint)
    #     best_loss = ckpt["loss"]
    #     # TODO: miss加载参数说明
    #     net.load_state_dict(ckpt["params"])
    #     print("checkpoint load")

    writer = SummaryWriter("super_log")
    for n, (num_epochs, lr) in enumerate(epoch_lr):
        optimizer = optim.Adam(net.parameters(), lr=lr, weight_decay=5e-3)
        for epoch in range(num_epochs):
            net.train()
            pbar = tqdm(enumerate(trainloader), total=len(trainloader))
            epoch_loss = 0.0
            for i, (img, mask) in pbar:
                img = img.to(device)
                mask = mask.to(device)
                out = net(img)

                # 残差学习
                # loss = criteron(out + img, mask)
                loss = criteron(out, mask)
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
                if i % 10 == 0:
                    pbar.set_description("loss:{}".format(loss))
                epoch_loss += loss.item()
            print("Epoch_loss:", epoch_loss)
            writer.add_scalar("SR_epoch_loss", epoch_loss,
                              sum(e[0] for e in epoch_lr[:n]) + epoch)

            with torch.no_grad():
                net.eval()
                test_loss = 0.0
                for i, (img, mask) in tqdm(enumerate(testloader), total=len(testloader)):
                    img = img.to(device)
                    mask = mask.to(device)
                    out = net(img)
                    loss = criteron(out + img, mask)
                    test_loss += loss.item()
                print("Test_loss:", test_loss)
                writer.add_scalar("SR_test_loss", test_loss,
                                  sum(e[0] for e in epoch_lr[:n]) + epoch)

                if (test_loss < best_loss):
                    best_loss = test_loss
                    torch.save(
                        {"params": net.state_dict(), "loss": test_loss},
                        sr_checkpoint
                    )
    writer.close()


if __name__ == "__main__":
    device = "cuda"
    batch_size = 4
    sr_checkpoint = "checkpoint/ckpt.pt"
    epoch_lr = [(20, 0.01)]
    train()
