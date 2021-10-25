import numpy as np
import torch
from reprod_log import ReprodLogger

from model.AlexNet_torch import alexnet


if __name__ == '__main__':
    reprod_logger = ReprodLogger()

    model = alexnet(True)
    model.eval()

    # read or gen fake data
    fake_data = np.load("pipeline/fake_data/fake_data.npy")
    fake_data = torch.from_numpy(fake_data)
    # forward
    out = model(fake_data)
    #
    reprod_logger.add("out", fake_data.cpu().detach().numpy())
    reprod_logger.save("pipeline/Step1/forward_torch.npy")

