import torch
import torchdynamo
import torchinductor.config
from torchdynamo.testing import same
from torchdynamo.testing import rand_strided

# torchinductor.config.debug = True
torchinductor.config.triton.dense_indexing = True
torch.manual_seed(0)

# The flag below controls whether to allow TF32 on matmul.
torch.backends.cuda.matmul.allow_tf32 = True

class Func(object):
    @torchdynamo.optimize("inductor")
    def mm(x, w, bias):
        y =  torch.mm(x, w)
        return y

    # mm+bias
    @torchdynamo.optimize("inductor")
    def mm_add(x, w, bias):
        y =  torch.mm(x, w)
        return y + bias

    # relu(mm)
    @torchdynamo.optimize("inductor")
    def mm_relu(x, w, bias):
        y =  torch.mm(x, w)
        return torch.relu(y)

    # relu(mm+bias)
    @torchdynamo.optimize("inductor")
    def mm_add_relu(x, w, bias):
        y =  torch.mm(x, w)
        y += bias
        return torch.relu(y)


# def test(shape, fusion_type="add"):
#     dtype=torch.float16
#     torch.manual_seed(0)
#     # allocate inputs
#     a = torch.randn(shape[0], dtype=dtype, device='cuda')
#     b = torch.randn(shape[1], dtype=dtype, device='cuda')
#     bias = torch.randn((shape[0][0],shape[1][1]), dtype=dtype, device='cuda')
#
#     if fusion_type == "":
#         mm_fusion = getattr(Func, f"mm")
#     else:
#         mm_fusion = getattr(Func, f"mm_{fusion_type}")
#     # torchinductor from template
#     torchinductor.config.triton.mm = "triton"
#     torchinductor.metrics.reset()
#     y = mm_fusion(a, b, bias)
#     assert torchinductor.metrics.generated_kernel_count == 1, f"codegen #kernel != 1"
#     # baseline
#     # reset to force code gen new python code
#     torchdynamo.reset()
#     torchinductor.config.triton.mm = "aten"
#     y_correct = mm_fusion(a, b, bias)
#     assert(same(y, y_correct, cos_similarity=True))

def test_cpu():
    shapes = [
        # alexnet
        ([128, 9216], [9216, 4096]),
        ([128, 4096], [4096, 4096]),
        ([128, 4096], [4096, 1000]),
        # BERT
        ([2048, 768], [768, 768]),
        ([2048, 768], [768, 3072]),
        ([2048, 3072], [3072, 768]),
        # hf_GPT2
        ([1024, 768], [768, 768]),
        ([1024, 768], [768, 3072]),
        ([1024, 3072], [3072, 768]),
        ([1024, 768], [768, 2304]),
    ]
    for shape in shapes:
        fusion_type = "add"
        dtype=torch.float16
        torch.manual_seed(0)
        # allocate inputs
        a = torch.randn(shape[0], dtype=dtype)
        b = torch.randn(shape[1], dtype=dtype)
        bias = torch.randn((shape[0][0],shape[1][1]), dtype=dtype)

        if fusion_type == "":
            mm_fusion = getattr(Func, f"mm")
        else:
            mm_fusion = getattr(Func, f"mm_{fusion_type}")
        # torchinductor from template
        torchinductor.config.triton.mm = "triton"
        torchinductor.metrics.reset()
        y = mm_fusion(a, b, bias)
        assert torchinductor.metrics.generated_kernel_count == 1, f"codegen #kernel != 1"
        # baseline
        # reset to force code gen new python code
        torchdynamo.reset()
        torchinductor.config.triton.mm = "aten"
        y_correct = mm_fusion(a, b, bias)
        assert(same(y, y_correct, cos_similarity=True))

# fusion_types = ["", "add", "relu", "add_relu"]
# shapes = [
#     # alexnet
#     ([128, 9216], [9216, 4096]),
#     ([128, 4096], [4096, 4096]),
#     ([128, 4096], [4096, 1000]),
#     # BERT
#     ([2048, 768], [768, 768]),
#     ([2048, 768], [768, 3072]),
#     ([2048, 3072], [3072, 768]),
#     # hf_GPT2
#     ([1024, 768], [768, 768]),
#     ([1024, 768], [768, 3072]),
#     ([1024, 3072], [3072, 768]),
#     ([1024, 768], [768, 2304]),
# ]
# for fusion_type in fusion_types:
#     print(f"testing correctness of mm+{fusion_type}")
#     for id, shape in enumerate(shapes):
#         test(shape, fusion_type)
#     print("passed")
