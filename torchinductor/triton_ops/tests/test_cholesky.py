import torch

from benchmarks.microbenchmarks.model import resnet50_layers, alexnet_layers
import torchdynamo
import math
from typing import List

def cholesky(x:torch.Tensor) -> torch.Tensor:
    L = torch.zeros(x.size())
    size = x.size()
    assert(size.__len__() == 2)
    assert(size[0] == size[1])
    N = size[0]
    for k in range(0,N):
        row_pred_squares_sum = 0.0
        for s in range(0, k):
            row_pred_squares_sum += L[k][s] * L[k][s]
        positive_value = x[k][k] - row_pred_squares_sum
        L[k][k] = math.sqrt(positive_value)
        for i in range(k+1, N):
            sum = 0.0
            for s in range(0,k):
                sum += L[i][s] * L[k][s]
            L[i][k] = (x[i][k] - sum)/L[k][k]
    return L

def cholesky_sliced(x:torch.Tensor) -> torch.Tensor:
    L = torch.zeros(x.size())
    size = x.size()
    assert(size.__len__() == 2)
    assert(size[0] == size[1])
    N = size[0]
    for k in range(0,N):
        row_pred_squares_sum = 0.0
        #take row
        #row_to_reduce = L.index_select[]
        for s in range(0, k):
            row_pred_squares_sum += L[k][s] * L[k][s]
        positive_value = x[k][k] - row_pred_squares_sum
        L[k][k] = math.sqrt(positive_value)
        for i in range(k+1, N):
            sum = 0.0
            for s in range(0,k):
                sum += L[i][s] * L[k][s]
            L[i][k] = (x[i][k] - sum)/L[k][k]
    return L

def my_compiler(gm: torch.fx.GraphModule, example_inputs: List[torch.Tensor]):
    print("my_compiler() called with FX graph:")
    gm.graph.print_tabular()
    return gm.forward



def test_python():
    a = torch.tensor([[4,3,2,1], [3,3,2,1], [2,2,2,1], [1,1,1,1]])
    L = cholesky(a)
    expectation = torch.tensor([[2.0, 0.0, 0.0, 0.0],
                   [1.5, 0.866, 0.0, 0.0],
                   [1.0, 0.5774, 0.8165, 0.0],
                   [0.5, 0.2887, 0.4082, 0.7071]])
    for i in range(0,4):
        for j in range(0,4):
            diff = abs(L[i][j] - expectation[i][j])
            diff_is_neglible = diff < 0.0001
            assert  diff_is_neglible

@torchdynamo.optimize('inductor')
def cholesky_opt(x: torch.Tensor) -> torch.Tensor:
    L = torch.zeros(x.size())
    size = x.size()
    assert (size.__len__() == 2)
    assert (size[0] == size[1])
    N = size[0]
    for k in range(0, N):
        row_pred_squares_sum = 0.0
        for s in range(0, k):
            row_pred_squares_sum += L[k][s].item() * L[k][s].item()
        positive_value = x[k][k].item() - row_pred_squares_sum
        L[k][k] = math.sqrt(positive_value)
        for i in range(k + 1, N):
            sum = 0.0
            for s in range(0, k):
                sum += L[i][s] * L[k][s]
            L[i][k] = (x[i][k] - sum) / L[k][k]
    return L

#@torchdynamo.optimize('inductor')
def cholesky_opt2(x: torch.Tensor) -> torch.Tensor:
    L = torch.zeros(x.size())
    size = x.size()
    assert (size.__len__() == 2)
    assert (size[0] == size[1])
    N = size[0]
    for k in range(0, N):
        row_pred_squares_sum = 0.0
        for s in range(0, k):
            row_pred_squares_sum += L[k][s] * L[k][s]
        positive_value = x[k][k] - row_pred_squares_sum
        L[k][k] = torch.sqrt(positive_value)
        for i in range(k + 1, N):
            sum = 0.0
            for s in range(0, k):
                sum += L[i][s] * L[k][s]
            L[i][k] = (x[i][k] - sum) / L[k][k]
    return L

#@torchdynamo.optimize('inductor')
def fuse_read(a:torch.Tensor, b:torch.Tensor) -> (torch.Tensor, torch.Tensor):
    pointwise_op1 = torch.sin(a)
    pointwise_op2 = torch.cos(b)
    pointwise_op3 = torch.exp(pointwise_op2)
    return (pointwise_op1, pointwise_op3)

def my_mm(a:torch.Tensor, b:torch.Tensor) -> torch.Tensor:
    assert (a.size().__len__() == 3)
    assert (a.size().__len__() == 3)
    assert (a.size()[2] == b.size()[1])
    matrices = []
    for left in a:
        for right in b:
            # create matrix
            matrix = torch.zeros([a.size()[1], b.size()[2]])
            for i in range(0, a.size()[1]):
                row = torch.select(left, 0, i)
                for j in range(0, b.size()[2]):
                    column = torch.select(right, 1, j)
                    element = torch.dot(row, column)
                    matrix[i][j] = element
            matrices.append(matrix)
    return torch.stack(matrices)



#@torchdynamo.optimize('inductor')
def fuse_conv2d(x, w, bias, stride, padding, dilation, groups) -> torch.Tensor:
    y = torch.conv2d(x, w, bias, stride, padding, dilation, groups)
    return y

def test_opt():
    a = torch.tensor([[4, 3, 2, 1], [3, 3, 2, 1], [2, 2, 2, 1], [1, 1, 1, 1]])
    L = cholesky_opt(a)
    expectation = torch.tensor([[2.0, 0.0, 0.0, 0.0],
                   [1.5, 0.866, 0.0, 0.0],
                   [1.0, 0.5774, 0.8165, 0.0],
                   [0.5, 0.2887, 0.4082, 0.7071]])
    for i in range(0,4):
        for j in range(0,4):
            diff = abs(L[i][j] - expectation[i][j])
            diff_is_neglible = diff < 0.0001
            if not diff_is_neglible:
                print(diff)
            assert  diff_is_neglible

def test_simplified():
    a = torch.randn([32, 512, 1024])
    b = torch.randn([32, 1024, 2048])
    c = fuse_read(a,b)

def test_vertical_fusion():
    @torchdynamo.optimize('inductor')
    def other(a:torch.Tensor) -> torch.Tensor:
        return torch.exp(a)

    @torchdynamo.optimize('inductor')
    def fuse_vert(a: torch.Tensor, b: torch.Tensor) -> (torch.Tensor, torch.Tensor, torch.Tensor):
        assert (a.size().__len__() == 3)
        assert (a.size().__len__() == 3)
        assert (a.size()[2] == b.size()[1])
        products = torch.matmul(a, b)
        pointwise_op1 = torch.sqrt(products)
        pointwise_op2 = torch.sin(other(pointwise_op1))
        return (products, pointwise_op1, pointwise_op2)
    a = torch.randn([2, 512, 1024])
    b = torch.randn([2, 1024, 2048])
    c = fuse_vert(a,b)
    print(c[0].data)

def test_vertical_fusion2():
    @torchdynamo.optimize('inductor')
    def other(a:torch.Tensor) -> torch.Tensor:
        return torch.mm(a,torch.t(a))

    @torchdynamo.optimize('inductor')
    def fuse_vert(a: torch.Tensor, b: torch.Tensor) -> (torch.Tensor, torch.Tensor, torch.Tensor):
        assert (a.size().__len__() == 3)
        assert (a.size().__len__() == 3)
        assert (a.size()[2] == b.size()[1])
        products = torch.matmul(a, b)
        single_matrix = torch.sum(products, 0)
        pointwise_op1 = torch.sqrt(single_matrix)
        pointwise_op2 = torch.sin(other(single_matrix))
        return (products, pointwise_op1, pointwise_op2)
    a = torch.randn([2, 512, 1024])
    b = torch.randn([2, 1024, 2048])
    c = fuse_vert(a,b)
    print(c[0].data)
def test_conv2d(layer_params):
    BATCH = 32
    IN_H, IN_W, IN_C, KERNEL_H, KERNEL_W, KERNEL_N, stride, padding = layer_params
    dilation, groups = (1,1), 1
    dtype=torch.float32

    torch.manual_seed(0)
    # allocate inputs, nchw
    x = torch.randn((BATCH, IN_C, IN_H, IN_W), dtype=dtype) #.to(memory_format=torch.channels_last)
    w = torch.randn((KERNEL_N, IN_C // groups, KERNEL_H, KERNEL_W),
                    dtype=dtype) #.to(memory_format=torch.channels_last)
    # bias = torch.randn((1, KERNEL_N, 1, 1), dtype=dtype, device='cuda') #.to(memory_format=torch.channels_last)
    bias = torch.randn((KERNEL_N), dtype=dtype)
    fuse_conv2d(x, w, bias, stride, padding, dilation, groups)

def test_con2d_layers():
    for id, layer in enumerate(alexnet_layers):
        test_conv2d(layer)
    print("passed")

