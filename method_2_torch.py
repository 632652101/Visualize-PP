if __name__ == '__main__':
    import matplotlib.pyplot as plt
    import torch

    from model.AlexNet_torch import alexnet

    model = alexnet(True)

    from saliency.ClassSpecificImageGenerationTorch import TorchVersion as ClassSpecific_torch

    class_specific_method = ClassSpecific_torch(model, target_class=130)

    out_list = class_specific_method.generate()

    # log
    from reprod_log import ReprodLogger
    reprod_logger = ReprodLogger()
    for idx, image in enumerate(out_list):
        reprod_logger.add(f"out_{idx}", image.cpu().detach().numpy())

    reprod_logger.save(f"npy/method_2/result_torch.npy")
