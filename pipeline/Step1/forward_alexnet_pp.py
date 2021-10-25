import numpy as np
import paddle
from reprod_log import ReprodLogger

from model.AlexNet import alexnet


if __name__ == '__main__':
    reprod_logger = ReprodLogger()

    model = alexnet(True)
    model.eval()

    # read or gen fake data
    fake_data = np.load("pipeline/fake_data/fake_data.npy")
    fake_data = paddle.to_tensor(fake_data)
    # forward
    out = model(fake_data)
    #
    reprod_logger.add("out", fake_data.cpu().detach().numpy())
    reprod_logger.save("pipeline/fake_data/forward_paddle.npy")

