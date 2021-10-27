
if __name__ == "__main__":
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
        from reprod_log import ReprodDiffHelper
        diff_helper = ReprodDiffHelper()
        torch_info = diff_helper.load_info(f"npy/result_torch_{image_name}.npy")
        paddle_info = diff_helper.load_info(f"npy/result_paddle_{image_name}.npy")

        diff_helper.compare_info(torch_info, paddle_info)

        diff_helper.report(diff_method="mean", path=f"logs/result_diff_{image_name}.log")
