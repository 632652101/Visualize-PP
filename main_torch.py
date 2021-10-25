if __name__ == '__main__':
    import matplotlib.pyplot as plt
    import torch
    from flashtorch_copy.utils import apply_transforms, load_image

    image = load_image('content/images/great_grey_owl.jpg')

    from model.AlexNet_torch import alexnet

    model = alexnet(True)

    from flashtorch_copy.saliency.backprop import Backprop

    backprop = Backprop(model)

    owl = apply_transforms(image)

    target_class = 24
    # Ready to roll!
    backprop.visualize(owl, None, guided=False, use_gpu=False)
