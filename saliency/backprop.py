import warnings

import matplotlib.pyplot as plt

import paddle
import paddle.nn as nn
from .utils import (denormalize,
                    format_for_plotting,
                    standardize_and_clip)


class Backprop:
    ####################
    # Public interface #
    ####################

    def __init__(self, model):
        # device
        device = paddle.get_device()
        paddle.set_device(device)

        self.model = model
        self.gradients = None
        # self._register_conv_hook()

    def calculate_gradients(self,
                            input_,
                            target_class=None,
                            take_max=False,
                            guided=False,
                            use_gpu=False):

        if 'inception' in self.model.__class__.__name__.lower():
            if input_.size()[1:] != (3, 299, 299):
                raise ValueError('Image must be 299x299 for Inception models.')

        if guided:
            self.relu_outputs = []
            self._register_relu_hooks()

        # self.model.zero_grad()
        # just for clear_grad
        opt = paddle.optimizer.SGD(parameters=self.model.parameters())
        opt.clear_grad()

        self.gradients = paddle.zeros(input_.shape)

        # Get a raw prediction value (logit) from the last linear layer

        output = self.model(input_)

        # Don't set the gradient target if the model is a binary classifier
        # i.e. has one class prediction

        if len(output.shape) == 1:
            target = None
        else:
            _, top_class = output.topk(1, axis=1)

            # Create a 2D tensor with shape (1, num_classes) and
            # set all element to zero

            target = paddle.zeros(shape=[1, output.shape[-1]])

            if (target_class is not None) and (top_class != target_class):
                warnings.warn(UserWarning(
                    f'The predicted class index {top_class.item()} does not' +
                    f'equal the target class index {target_class}. ' +
                    'Calculating the gradient w.r.t. the predicted class.'
                ))

            # Set the element at top class index to be 1
            target_np = target.cpu().detach().numpy()
            target_np[0][top_class] = 1
            target = paddle.to_tensor(target_np)

        # Calculate gradients of the target class output w.r.t. input_

        output.backward(grad_tensor=target)

        # Detach the gradients from the graph and move to cpu

        gradients = self.model.input_grad[0]
        if gradients is None:
            raise Exception("Value erro gradients is None!")

        if take_max:
            # Take the maximum across colour channels

            gradients = gradients.max(axis=0, keepdims=True)[0]

        return gradients

    def visualize(self, input_, target_class, guided=False, use_gpu=False,
                  figsize=(16, 4), cmap='viridis', alpha=.5,
                  return_output=False):
        # Calculate gradients

        gradients = self.calculate_gradients(input_,
                                             target_class,
                                             guided=guided,
                                             use_gpu=use_gpu)
        max_gradients = self.calculate_gradients(input_,
                                                 target_class,
                                                 guided=guided,
                                                 take_max=True,
                                                 use_gpu=use_gpu)

        # Setup subplots
        gradients = paddle.to_tensor(gradients)
        max_gradients = paddle.to_tensor(max_gradients)

        subplots = [
            # (title, [(image1, cmap, alpha), (image2, cmap, alpha)])
            ('Input image',
             [(format_for_plotting(denormalize(input_)), None, None)]),
            ('Gradients across RGB channels',
             [(format_for_plotting(standardize_and_clip(gradients)),
               None,
               None)]),
            ('Max gradients',
             [(format_for_plotting(standardize_and_clip(max_gradients)),
               cmap,
               None)]),
            ('Overlay',
             [(format_for_plotting(denormalize(input_)), None, None),
              (format_for_plotting(standardize_and_clip(max_gradients)),
               cmap,
               alpha)])
        ]

        fig = plt.figure(figsize=figsize)

        for i, (title, images) in enumerate(subplots):
            ax = fig.add_subplot(1, len(subplots), i + 1)
            ax.set_axis_off()
            ax.set_title(title)

            for image, cmap, alpha in images:
                ax.imshow(image, cmap=cmap, alpha=alpha)

        if return_output:
            return gradients, max_gradients

    #####################
    # Private interface #
    #####################

    def _register_conv_hook(self):
        def _record_gradients(module, grad_in, grad_out):
            if self.gradients.shape == grad_in[0].shape:
                self.gradients = grad_in[0]

        # for _, module in self.model.named_modules():
        #     if isinstance(module, nn.layer.conv.Conv2D):
        #         module.register_backward_hook(_record_gradients)
        #         break

        for _, module in self.model.named_modules():
            # if isinstance(module, nn.modules.conv.Conv2d):
            if isinstance(module, nn.layer.conv.Conv2D):
                module.register_backward_hook(_record_gradients)
                break

    def _register_relu_hooks(self):
        def _record_output(module, input_, output):
            self.relu_outputs.append(output)

        def _clip_gradients(module, grad_in, grad_out):
            relu_output = self.relu_outputs.pop()
            clippled_grad_out = grad_out[0].clamp(0.0)

            return (clippled_grad_out.mul(relu_output),)

        for _, module in self.model.named_modules():
            if isinstance(module, nn.ReLU):
                module.register_forward_hook(_record_output)
                module.register_backward_hook(_clip_gradients)
