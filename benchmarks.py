from utils import make_preload_model, summery_model, benchmark_model

if __name__ == '__main__':
    model = make_preload_model(r"checkpoints/bunny_coarse/latest.pth")

    summery_model(model)
    benchmark_model(model, 1024 * 16, is_cuda=True)

