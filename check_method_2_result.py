
if __name__ == "__main__":

    from reprod_log import ReprodDiffHelper
    diff_helper = ReprodDiffHelper()
    torch_info = diff_helper.load_info(f"npy/method_2/result_torch.npy")
    paddle_info = diff_helper.load_info(f"npy/method_2/result_pp.npy")

    diff_helper.compare_info(torch_info, paddle_info)

    diff_helper.report(diff_method="mean", path=f"logs/method_2/result_diff.log")
