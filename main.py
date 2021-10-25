if __name__ == '__main__':
    import matplotlib.pyplot as plt
    import paddle
    from utils import apply_transforms, load_image

    image = load_image('content/images/great_grey_owl.jpg')

    from model.AlexNet import alexnet

    model = alexnet(True)

    from saliency.backprop import Backprop

    backprop = Backprop(model)

    owl = apply_transforms(image)

    target_class = 24
    # Ready to roll!
    backprop.visualize(owl, target_class, guided=False, use_gpu=True)

