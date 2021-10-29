if __name__ == '__main__':

    images_name = [
        "airdale_terrier",
        "basset_hound",
        "dalmatian",
        "great_grey_owl",
        "jay",
        "king_penguin",
        "oystercatcher",
        "peacock",
        "shih_tzu",
        "toad_lilly",
        "toucan"
    ]

    for image_name in images_name:

        from flashtorch.utils import apply_transforms, load_image

        image = load_image(f'content/images/{image_name}.jpg')

        from model.AlexNet_torch import alexnet

        model = alexnet(True)
        model.eval()

        from flashtorch.saliency.backprop import Backprop

        backprop = Backprop(model)

        owl = apply_transforms(image)
        out = model(owl)
        #
        target_class = 24
        # Ready to roll!
        visual_1, visual_2, images = backprop.visualize(owl, None, guided=False, use_gpu=False, return_output=True,
                                                        save_path=f"images/method_1/{image_name}_torch.jpg")
        visual_1 = visual_1.cpu().detach().numpy()
        visual_2 = visual_2.cpu().detach().numpy()
        #
        from reprod_log import ReprodLogger
        reprod_logger = ReprodLogger()
        reprod_logger.add("input", owl.cpu().detach().numpy())
        reprod_logger.add("out", out.cpu().detach().numpy())
        reprod_logger.add("visual_1", visual_1)
        reprod_logger.add("visual_2", visual_2)
        for idx, image in enumerate(images):
            if idx >= 3:
                break
            reprod_logger.add(f"images_{idx}", image.cpu().detach().numpy())
        reprod_logger.save(f"npy/method_1/result_torch_{image_name}.npy")
