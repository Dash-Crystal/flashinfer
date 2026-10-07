import pytest
import torch
import torch.nn.functional as F

from flashinfer import autotune, bmm_fp8
from flashinfer.utils import get_compute_capability
from tests.utils_fp8 import to_float8


@pytest.mark.parametrize(
    "transpose_a,transpose_b",
    [(False, False), (False, True), (True, False), (True, True)],
)
@pytest.mark.parametrize("m,n", [(64, 32), (512, 4096)])
def test_tf32x3_fp32_preserves_batched_layouts_epilogue_and_graph(
    transpose_a, transpose_b, m, n
):
    """The shared FP32 adapter must preserve views, beta*C, and live graph inputs."""
    from flashinfer.gemm.fp32 import bmm_fp32_tf32x3

    torch.manual_seed(91)
    a = torch.randn(3, m, 128, device="cuda")
    b = torch.randn(3, 128, n, device="cuda")
    if transpose_a:
        a = a.mT.contiguous().mT
    if transpose_b:
        b = b.mT.contiguous().mT
    c = torch.randn(3, m, n, device="cuda")

    def reference():
        return (0.75 * (a.double() @ b.double()) - 0.5 * c.double()).float()

    def check(value):
        assert bool(torch.allclose(value, reference(), atol=7e-5, rtol=5e-5))

    check(bmm_fp32_tf32x3(a, b, c, -0.5, 0.75))
    bmm_fp32_tf32x3(a, b, c, -0.5, 0.75)
    stream = torch.cuda.Stream()
    stream.wait_stream(torch.cuda.current_stream())
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph, stream=stream):
        actual = bmm_fp32_tf32x3(a, b, c, -0.5, 0.75)
    torch.cuda.current_stream().wait_stream(stream)
    a.add_(0.125)
    graph.replay()
    check(actual)
    c.fill_(float("nan"))
    actual = bmm_fp32_tf32x3(a[0], b[0], c[0], 0.0)
    expected = (a[0].double() @ b[0].double()).float()
    assert bool(torch.allclose(actual, expected, atol=7e-5, rtol=5e-5))


@pytest.mark.parametrize("b", [1, 16])
@pytest.mark.parametrize("m", [1, 48, 128])
@pytest.mark.parametrize("n", [64, 80, 10304])
@pytest.mark.parametrize("k", [64, 256, 2688])
@pytest.mark.parametrize("input_dtype", [torch.float8_e4m3fn, torch.float8_e5m2])
@pytest.mark.parametrize("mat2_dtype", [torch.float8_e4m3fn, torch.float8_e5m2])
@pytest.mark.parametrize("res_dtype", [torch.bfloat16, torch.float16])
@pytest.mark.parametrize("backend", ["cudnn", "cublas", "cutlass", "auto"])
@pytest.mark.parametrize("auto_tuning", [True, False])
def test_bmm_fp8(b, m, n, k, input_dtype, mat2_dtype, res_dtype, backend, auto_tuning):
    compute_capability = get_compute_capability(torch.device("cuda"))
    if backend == "cutlass" and compute_capability[0] not in [10, 11, 12]:
        pytest.skip(
            "bmm_fp8 with cutlass backend is only supported on SM100, SM110, and SM120/121 GPUs."
        )
    if input_dtype == torch.float8_e5m2 and mat2_dtype == torch.float8_e5m2:
        pytest.skip("Invalid combination: both input and mat2 are e5m2")
    if input_dtype == torch.float8_e5m2 or mat2_dtype == torch.float8_e5m2:
        if backend == "cutlass":
            pytest.skip("Invalid combination: cutlass does not support e5m2")
    if auto_tuning and backend not in ["cutlass", "cudnn", "cublas"]:
        pytest.skip(
            "Invalid combination: auto_tuning only supported for cutlass, cudnn, and cublas"
        )
    if compute_capability[0] == 11 and (
        input_dtype == torch.float8_e5m2 or mat2_dtype == torch.float8_e5m2
    ):
        pytest.skip(
            "Invalid combination: only cutlass supports SM110 which does not support e5m2"
        )
    input = torch.randn([b, m, k], device="cuda", dtype=torch.bfloat16)
    input_fp8, input_inv_s = to_float8(input, dtype=input_dtype)

    # mat2 row  major -> column major
    mat2 = torch.randn([b, n, k], device="cuda", dtype=torch.bfloat16).transpose(-2, -1)
    mat2_fp8, mat2_inv_s = to_float8(mat2, dtype=mat2_dtype)
    reference = torch.bmm(input, mat2)

    res = torch.empty([b, m, n], device="cuda", dtype=res_dtype)

    with autotune(auto_tuning):
        bmm_fp8(
            input_fp8,
            mat2_fp8,
            input_inv_s,
            mat2_inv_s,
            res_dtype,
            res,
            backend=backend,
        )

    cos_sim = F.cosine_similarity(
        reference.reshape(-1).float(), res.reshape(-1).float(), dim=0
    )
    assert cos_sim > 0.99


if __name__ == "__main__":
    pytest.main([__file__])
