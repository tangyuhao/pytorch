# Owner(s): ["module: ProxyTensor"]
# ruff: noqa: F841

"""
Tests for make_fx with C++ FakeTensor mode.

All tests run make_fx(tracing_mode="real") inside cpp_fake_tensor_mode().
The C++ Fake dispatch key handles ops with Meta kernels. Ops without Meta
kernels fall back to CppFakeFallbackMode, which looks up the specific
Python handler (decomposition, fake_impl, etc.) and calls it. Sub-ops
re-enter C++ Fake dispatch, so all results remain C++ fake tensors.
"""

from torch.testing import make_tensor
from torch.testing._internal.common_utils import TestCase, run_tests
import torch
import torch._dynamo
import torch._library.simple_registry
import torch._library.utils
import unittest
import warnings
import operator
import contextlib
from collections.abc import Iterable
from torch.nn.utils import stateless
from torch._subclasses.fake_tensor import (
    DynamicOutputShapeException,
    DataDependentOutputException,
    FakeTensorConverter,
    FakeTensorMode,
)
from torch._subclasses.functional_tensor import FunctionalTensor, FunctionalTensorMode
from torch._decomp import decomposition_table
from torch.fx.experimental.symbolic_shapes import (
    ShapeEnv,
    eval_guards,
    bind_symbols,
    fx_placeholder_vals,
    fx_placeholder_targets,
    guard_int,
    GuardOnDataDependentSymNode,
)
from torch.testing._internal.common_device_type import (
    ops,
    instantiate_device_type_tests,
)
from torch.testing._internal.common_methods_invocations import (
    op_db,
    skip,
    xfail,
    skipOps,
)
from torch.testing._internal.custom_op_db import custom_op_db
from torch.testing._internal.hop_db import hop_db
import torch.testing._internal.optests as optests
from torch._dispatch.python import enable_python_dispatcher
from torch.fx.experimental.proxy_tensor import (
    make_fx,
    DecompositionInterpreter,
    get_isolated_graphmodule,
)
from torch.utils._pytree import tree_map, tree_map_only
from torch.fx.passes.runtime_assert import insert_deferred_runtime_asserts
from torch import nn
from torch.utils._python_dispatch import TorchDispatchMode

import functools
import itertools
import torch._functorch.config
import torch.fx.experimental._config

aten = torch.ops.aten

HAS_CUDA = torch.cuda.is_available()

USE_TORCHVISION = False
try:
    import torchvision

    USE_TORCHVISION = True
except ImportError:
    warnings.warn(
        "Couldn't import torchvision. Some of our tests use it, try "
        "to install it with commands from pytorch.org, post-fixed with "
        "`--no-deps` to avoid overwriting the pytorch installation",
        UserWarning,
    )


@contextlib.contextmanager
def cpp_fake_tensor_mode(*, shape_env=None):
    """Activate C++ FakeTensor mode with a Python fallback for unhandled ops.

    The C++ Fake dispatch key handles ops that have Meta kernels.
    Ops without Meta kernels are forwarded to CppFakeFallbackMode which
    looks up the specific Python handler (decomposition, fake_impl, etc.)
    and calls it. Sub-ops re-enter C++ Fake dispatch, so all tensors
    remain C++ fake tensors — no Python FakeTensors are created.
    """
    if shape_env is None:
        shape_env = ShapeEnv()
    converter = FakeTensorConverter()
    # fallback = CppFakeFallbackMode()
    # torch._C._create_and_enter_fake_tensor_mode(converter, shape_env, fallback)
    torch._C._create_and_enter_fake_tensor_mode(converter, shape_env)
    try:
        yield shape_env
    finally:
        torch._C._exit_fake_tensor_mode()


def _create_new_input(x):
    if not isinstance(x, torch.Tensor):
        return x
    if x.dtype != torch.float:
        return x + 1
    if x.is_leaf:
        return torch.rand_like(x, requires_grad=x.requires_grad)
    else:
        return torch.rand_like(x)


class TestCppFakeProxyTensor(TestCase):
    """Tests for make_fx under C++ FakeTensor mode.

    Each test wraps the make_fx call in cpp_fake_tensor_mode() and uses
    tracing_mode="real" so that the C++ Fake dispatch key provides the
    fake tensor semantics.
    """

    def _test(self, f, inps, compare_graph=False):
        # Trace under C++ fake mode
        with cpp_fake_tensor_mode():
            cpp_gm = make_fx(f, tracing_mode="real")(*inps)

        if compare_graph:
            # Trace under Python fake mode and compare graph structure
            py_gm = make_fx(f, tracing_mode="fake")(*inps)
            cpp_ops = [n.target for n in cpp_gm.graph.nodes if n.op == "call_function"]
            py_ops = [n.target for n in py_gm.graph.nodes if n.op == "call_function"]
            self.assertEqual(cpp_ops, py_ops)

        # Verify correctness with real inputs
        new_inps = tree_map(_create_new_input, inps)
        r1 = cpp_gm(*new_inps)
        r2 = f(*new_inps)
        self.assertEqual(r1, r2)

    def test_make_fx_simple(self):
        def f(x):
            return torch.sin(x)

        self._test(f, (torch.randn(3),))

    def test_scalar_device(self, device="cpu"):
        def f(a, b):
            return a + b

        self._test(f, [torch.randn(3, device=device), torch.tensor(5)])

    def test_empty_like_doesnt_burn_in_defaults(self):
        def f(x):
            return torch.empty_like(x)

        with cpp_fake_tensor_mode():
            out = make_fx(f, tracing_mode="real")(torch.randn(3))
        self.assertExpectedInline(
            out.code.strip(),
            """\
def forward(self, x_1):
    empty_like = torch.ops.aten.empty_like.default(x_1, pin_memory = False);  x_1 = None
    return empty_like""",
        )

    def test_proxy_tensor_mode_with_decomp_table_preserves_proxy(self):
        def f(x):
            y = x.new_zeros(x.size())
            y.copy_(x)
            return y

        def _new_zeros_decomp(
            inp, size, dtype=None, layout=None, device=None, pin_memory=None
        ):
            return torch.zeros(size, dtype=inp.dtype, device=inp.device)

        factory_func_decomp = {torch.ops.aten.new_zeros.default: _new_zeros_decomp}

        with cpp_fake_tensor_mode():
            out = make_fx(
                f, tracing_mode="real", decomposition_table=factory_func_decomp
            )(torch.ones(2))
        self.assertExpectedInline(
            out.code,
            """\



def forward(self, x_1):
    zeros = torch.ops.aten.zeros.default([2], dtype = torch.float32, device = device(type='cpu'), pin_memory = False)
    copy_ = torch.ops.aten.copy_.default(zeros, x_1);  zeros = x_1 = None
    return copy_
    """,
        )

    def test_make_fx_reentrant_dispatch(self):
        def f(x):
            return torch.ops.aten.norm.Scalar(x, 2.0)

        def norm_decomp(x, p=2.0):
            if p != 2.0:
                raise RuntimeError("can't handle with p != 2")
            return torch.sqrt(torch.sum(torch.square(x)))

        decomp = {torch.ops.aten.norm.Scalar: norm_decomp}

        with cpp_fake_tensor_mode():
            traced = make_fx(f, tracing_mode="real", decomposition_table=decomp)(
                torch.rand(3)
            )

        for n in traced.graph.nodes:
            self.assertTrue("square" not in str(n.target))
            self.assertTrue("norm" not in str(n.target))

    def test_varargs(self):
        def f(*args):
            return sum(args)

        self._test(f, [torch.randn(2), torch.randn(2)])

    def test_proxy_tensor(self):
        def f_grad(x):
            val = x.cos().cos().sum()
            return torch.autograd.grad(val, x)

        def f_backward(x):
            val = x.cos().cos().sum()
            val.backward()
            return x.grad

        for f in [f_grad, f_backward]:
            self._test(f, [torch.randn(3, requires_grad=True)])

    def test_inplace_metadata(self):
        def f(x):
            x = x.clone()
            x.unsqueeze_(-1)
            if x.shape[-1] != 1:
                raise AssertionError(f"expected x.shape[-1] == 1, got {x.shape[-1]}")
            return x

        self._test(f, [torch.randn(5)])

    def test_mode_tracing_factory_function(self):
        def f(x):
            return x + torch.randn(x.shape)

        with cpp_fake_tensor_mode():
            traced = make_fx(f, tracing_mode="real")(torch.randn(3))
        self.assertTrue(
            any(node.target == aten.randn.default for node in traced.graph.nodes)
        )

    def test_val_metadata_mutation(self):
        def f(x):
            y = x.clone()
            y.unsqueeze_(0)
            return y

        with cpp_fake_tensor_mode():
            traced = make_fx(f, tracing_mode="real")(torch.randn(3, requires_grad=True))
        self.assertEqual(
            [
                tuple(node.meta["val"].shape)
                for node in traced.graph.nodes
                if "val" in node.meta
            ],
            [(3,), (3,), (1, 3)],
        )

    def test_make_fx_overloads(self):
        def f(x):
            return x.cos() + torch.randn(x.shape)

        with cpp_fake_tensor_mode():
            traced = make_fx(f, tracing_mode="real")(torch.randn(3))

        self.assertTrue(
            all(
                isinstance(node.target, torch._ops.OpOverload)
                for node in traced.graph.nodes
                if node.op == "call_function"
            )
        )

    @unittest.skip("C++ fake mode has no constant propagation")
    def test_tensor_constants(self):
        def f():
            val = torch.tensor(float("inf"))
            return torch.full((100, 100), val)

        self._test(f, [])

    @unittest.skip("C++ fake mode has no constant propagation")
    def test_constant_proxy_tensor_mut(self):
        def f():
            val = torch.tensor(float(1))
            val.add_(2)
            return torch.full((100, 100), val)

        with cpp_fake_tensor_mode():
            g = make_fx(f, tracing_mode="real")()
        self.assertEqual(g(), f())
        self.assertEqual(g(), f())

    @unittest.skip("C++ fake mode has no constant propagation")
    def test_constant_unbind(self):
        def f():
            val = torch.tensor([2])
            (r,) = torch.unbind(val, 0)
            return r.item()

        with cpp_fake_tensor_mode():
            g = make_fx(f, tracing_mode="real")()
        self.assertEqual(g(), f())

    def test_decomposition_interpreter(self):
        def fn(x):
            return torch.nn.functional.silu(x)

        x = torch.rand((4, 4))
        with cpp_fake_tensor_mode():
            fx_module = make_fx(fn, tracing_mode="real", decomposition_table=None)(x)

        found_silu = False
        for n in fx_module.graph.nodes:
            if (
                n.target == torch.ops.aten.silu
                or n.target == torch.ops.aten.silu.default
            ):
                found_silu = True

        self.assertTrue(found_silu)

        new_graph = torch.fx.Graph()
        silu_decomp_table = {
            torch.ops.aten.silu.default: decomposition_table[
                torch.ops.aten.silu.default
            ]
        }
        DecompositionInterpreter(
            fx_module,
            new_graph=new_graph,
            decomposition_table=silu_decomp_table,
        ).run(x)

        decomposed_module = torch.fx.GraphModule(fx_module, new_graph)

        for n in decomposed_module.graph.nodes:
            self.assertTrue(n.target != torch.ops.aten.silu)
            self.assertTrue(n.target != torch.ops.aten.silu.default)

        self.assertEqual(fx_module(x), decomposed_module(x))

    def test_make_fx_model_fwd_bwd(self):
        class Foo(torch.nn.Module):
            def __init__(self) -> None:
                super().__init__()
                self.linear = torch.nn.Linear(5, 5)

            def forward(self, x):
                return self.linear(x).relu()

        model = Foo()

        def f(x, params):
            out = torch.func.functional_call(model, params, x).sum()
            out.backward()
            return list(params.values())

        input = torch.randn(3, 5, requires_grad=True)
        params = dict(model.named_parameters())
        with cpp_fake_tensor_mode():
            fx_f = make_fx(f, tracing_mode="real")(input, params)
        self.assertTrue(
            torch.allclose(fx_f(input, params)[0], f(input, params)[0])
            or torch.allclose(fx_f(input, params)[0], f(input, params)[1])
        )
        self.assertTrue(
            torch.allclose(fx_f(input, params)[1], f(input, params)[0])
            or torch.allclose(fx_f(input, params)[1], f(input, params)[1])
        )

    def test_make_fx_model_fwd_bwd_wgtupdate(self):
        class Foo(torch.nn.Module):
            def __init__(self) -> None:
                super().__init__()
                self.linear = torch.nn.Linear(5, 5)

            def forward(self, x):
                return self.linear(x).relu()

        model = Foo()

        def f(args, params, buffers):
            for p in params.values():
                p.grad = None
            if not isinstance(args, Iterable):
                args = [args]
            params_and_buffers = {**params, **buffers}
            out = torch.func.functional_call(model, params_and_buffers, args)
            out.sum().backward()
            return [p - 1e-4 * p.grad for p in params.values()]

        input = torch.randn(3, 5, requires_grad=True)
        params = dict(model.named_parameters())
        buffers = dict(model.named_buffers())
        with cpp_fake_tensor_mode():
            fx_f = make_fx(f, tracing_mode="real")(input, params, buffers)
        self.assertTrue(
            torch.allclose(
                fx_f(input, params, buffers)[0],
                f(input, params, buffers)[0],
                atol=1e-03,
            )
            or torch.allclose(
                fx_f(input, params, buffers)[0],
                f(input, params, buffers)[1],
                atol=1e-03,
            )
        )
        self.assertTrue(
            torch.allclose(
                fx_f(input, params, buffers)[1],
                f(input, params, buffers)[0],
                atol=1e-03,
            )
            or torch.allclose(
                fx_f(input, params, buffers)[1],
                f(input, params, buffers)[1],
                atol=1e-03,
            )
        )

    def test_make_fx_model_double_param(self):
        class Emformer(torch.nn.Module):
            def __init__(
                self,
                input_dim: int = 256,
            ) -> None:
                super().__init__()
                self.layer_norm = torch.nn.LayerNorm(input_dim)

            def forward(mod_self, x):  # noqa: B902
                self.assertTrue(isinstance(mod_self.layer_norm.weight, torch.Tensor))
                y = mod_self.layer_norm(x)
                self.assertTrue(isinstance(mod_self.layer_norm.weight, torch.Tensor))
                z = mod_self.layer_norm(y)
                return z

        with cpp_fake_tensor_mode():
            gm = make_fx(Emformer(), tracing_mode="real")(torch.randn(16, 1, 256))
        ops = {n.target for n in gm.graph.nodes if n.op == "call_function"}
        self.assertEqual(len(ops), 2)

    def test_partial_decomp(self):
        def f(a, b, c):
            x = torch.addmm(a, b, c)
            y = torch.addmm(a, b, c, beta=2, alpha=1)
            return x + y

        inps = [torch.randn(5, 5), torch.randn(5, 5), torch.randn(5, 5)]
        with cpp_fake_tensor_mode():
            fx_g = make_fx(f, tracing_mode="real")(*inps)

        def addmm(a, b, c, beta=1, alpha=1):
            if beta == 1 and alpha == 1:
                return NotImplemented
            return beta * a + alpha * (b @ c)

        with cpp_fake_tensor_mode():
            decomposed_fx = make_fx(
                f, tracing_mode="real", decomposition_table={aten.addmm.default: addmm}
            )(*inps)

        self.assertEqual(fx_g(*inps), decomposed_fx(*inps))
        self.assertEqual(
            len([n for n in fx_g.graph.nodes if n.target == aten.addmm.default]), 2
        )
        self.assertEqual(
            len(
                [n for n in decomposed_fx.graph.nodes if n.target == aten.addmm.default]
            ),
            1,
        )

    def test_decomp_of_capture(self):
        val = torch.randn(5)

        def f(x):
            return x.t() + val.t()

        def nop(x):
            return x.cos()

        with cpp_fake_tensor_mode():
            traced = make_fx(
                f,
                tracing_mode="real",
                decomposition_table={torch.ops.aten.t.default: nop},
            )(torch.randn(5))
        self.assertEqual(
            len(
                [n for n in traced.graph.nodes if n.target == torch.ops.aten.t.default]
            ),
            0,
        )

    @unittest.skipIf(not HAS_CUDA, "CUDA-only test")
    def test_amp_cache(self):
        layer = torch.nn.Conv2d(3, 3, 3).cuda()

        def f(x, w):
            return torch.nn.functional.conv2d(x, w, stride=layer.stride)

        inp = torch.randn(4, 3, 10, 10, device="cuda")
        with torch.autocast("cuda"):
            with cpp_fake_tensor_mode():
                out_graph = make_fx(f, tracing_mode="real")(inp, layer.weight).graph
                out_graph2 = make_fx(f, tracing_mode="real")(inp, layer.weight).graph

        self.assertEqual(len(out_graph.nodes), len(out_graph2.nodes))
        for a, b in zip(out_graph.nodes, out_graph2.nodes):
            self.assertEqual(a.op, b.op)

    def test_strides(self):
        def f(x):
            self.assertTrue(x.is_contiguous())
            self.assertFalse(x.is_contiguous(memory_format=torch.channels_last))
            x = x.permute(0, 3, 1, 2)
            self.assertFalse(x.is_contiguous())
            self.assertTrue(x.is_contiguous(memory_format=torch.channels_last))
            return x

        with cpp_fake_tensor_mode():
            make_fx(f, tracing_mode="real")(torch.randn(2, 3, 4, 5))

        def f(x):
            self.assertTrue(x.is_contiguous())
            y = x[:, 1]
            self.assertFalse(y.is_contiguous())
            y = x[:, ::2]
            self.assertFalse(y.is_contiguous())
            return x.cos()

        with cpp_fake_tensor_mode():
            make_fx(f, tracing_mode="real")(torch.randn(2, 3, 4, 5))

    def test_pr_86917(self):
        def f(a, b):
            return torch.ops.aten.nll_loss_forward(a, b, None, 1, 10)

        self._test(f, [torch.randn(1, 10), torch.zeros(1, dtype=torch.long)])

    def test_use_fake_and_tensor(self):
        def f(x, y):
            z = torch.tensor([2.0, 3.0])
            return x + y + z

        with cpp_fake_tensor_mode():
            g = make_fx(f, tracing_mode="real")(torch.randn(2), torch.randn(2))
        x, y = torch.randn(2), torch.randn(2)
        self.assertEqual(g(x, y), f(x, y))

    def test_fused_adam(self):
        params = [torch.randn(10, 10) for _ in range(10)]
        grads = [torch.randn(10, 10) for _ in range(10)]
        exp_avgs = [torch.randn(10, 10) for _ in range(10)]
        exp_avg_sqs = [torch.randn(10, 10) for _ in range(10)]
        max_exp_avg_sqs = [torch.randn(10, 10) for _ in range(10)]
        state_steps = [torch.tensor(0) for _ in range(10)]

        def fused_adam(
            params, grads, exp_avgs, exp_avg_sqs, max_exp_avg_sqs, state_steps
        ):
            (new_params, _, _, _, _) = aten._fused_adam.default(
                params,
                grads,
                exp_avgs,
                exp_avg_sqs,
                max_exp_avg_sqs,
                state_steps,
                lr=0.1,
                beta1=0.9,
                beta2=0.999,
                weight_decay=0.01,
                eps=1e-8,
                amsgrad=False,
                maximize=False,
            )

            for p, new_p in zip(params, new_params):
                p.copy_(new_p)

            return params

        with cpp_fake_tensor_mode():
            gm = make_fx(fused_adam, tracing_mode="real")(
                params,
                grads,
                exp_avgs,
                exp_avg_sqs,
                max_exp_avg_sqs,
                state_steps,
            )
        ensure_ops_have_val = [aten._fused_adam.default, operator.getitem]
        for n in gm.graph.nodes:
            if n.op == "call_function" and n.target in ensure_ops_have_val:
                self.assertIn("val", n.meta)

    def test_alias(self):
        def f(x):
            return torch.ops.aten.alias(x)

        with cpp_fake_tensor_mode():
            r = str(make_fx(f, tracing_mode="real")(torch.randn(2)).code).strip()
        self.assertExpectedInline(
            r,
            """\
def forward(self, x_1):
    alias = torch.ops.aten.alias.default(x_1);  x_1 = None
    return alias""",
        )

    def test_meta(self):
        def f(x):
            a = x.cos()
            b = torch.var_mean(a, dim=0)
            c = b * 2
            return c

        with cpp_fake_tensor_mode():
            out = make_fx(f, tracing_mode="real")(torch.randn(5, 5))
        for n in out.graph.nodes:
            if n.op == "output":
                continue
            self.assertTrue("val" in n.meta)

    def test_simple_add(self):
        def f(x, y):
            return x + y

        with cpp_fake_tensor_mode():
            gm = make_fx(f, tracing_mode="real")(torch.randn(3, 4), torch.randn(3, 4))

        # Verify the graph has the expected structure
        call_nodes = [n for n in gm.graph.nodes if n.op == "call_function"]
        self.assertTrue(len(call_nodes) >= 1)

        # Verify it runs correctly with real inputs
        x, y = torch.randn(3, 4), torch.randn(3, 4)
        self.assertEqual(gm(x, y), f(x, y))

    def test_matmul(self):
        def f(x, y):
            return torch.matmul(x, y)

        with cpp_fake_tensor_mode():
            gm = make_fx(f, tracing_mode="real")(torch.randn(3, 4), torch.randn(4, 5))
        x, y = torch.randn(3, 4), torch.randn(4, 5)
        self.assertEqual(gm(x, y), f(x, y))

    def test_multiple_outputs(self):
        def f(x):
            return torch.max(x, dim=0)

        with cpp_fake_tensor_mode():
            gm = make_fx(f, tracing_mode="real")(torch.randn(3, 4))
        x = torch.randn(3, 4)
        r1 = gm(x)
        r2 = f(x)
        self.assertEqual(r1[0], r2[0])
        self.assertEqual(r1[1], r2[1])

    def test_inplace_ops(self):
        def f(x):
            y = x.clone()
            y.add_(1.0)
            return y

        self._test(f, (torch.randn(3, 4),))

    def test_view_ops(self):
        def f(x):
            y = x.view(2, 6)
            z = y.t()
            return z

        with cpp_fake_tensor_mode():
            gm = make_fx(f, tracing_mode="real")(torch.randn(3, 4))
        x = torch.randn(3, 4)
        self.assertEqual(gm(x), f(x))

    def test_cat(self):
        def f(x, y):
            return torch.cat([x, y], dim=0)

        self._test(f, (torch.randn(3, 4), torch.randn(5, 4)))

    def test_nn_module(self):
        model = nn.Sequential(
            nn.Linear(10, 20),
            nn.ReLU(),
            nn.Linear(20, 5),
        )

        def f(x):
            return model(x)

        with cpp_fake_tensor_mode():
            gm = make_fx(f, tracing_mode="real")(torch.randn(3, 10))
        x = torch.randn(3, 10)
        self.assertEqual(gm(x), f(x))

    def test_comparison_with_python_fake(self):
        """Verify that C++ fake mode and Python fake mode produce the same graph structure."""

        def f(x):
            y = torch.sin(x)
            z = torch.cos(y)
            return z + x

        inp = torch.randn(4, 4)

        # Trace with Python fake mode
        py_gm = make_fx(f, tracing_mode="fake")(inp)

        # Trace with C++ fake mode
        with cpp_fake_tensor_mode():
            cpp_gm = make_fx(f, tracing_mode="real")(inp)

        # Both should produce identical graph structure
        py_ops = [n.target for n in py_gm.graph.nodes if n.op == "call_function"]
        cpp_ops = [n.target for n in cpp_gm.graph.nodes if n.op == "call_function"]
        self.assertEqual(py_ops, cpp_ops)

        # Both should produce correct results
        x = torch.randn(4, 4)
        self.assertEqual(py_gm(x), cpp_gm(x))

    def test_factory_ops_under_cpp_fake(self):
        """Factory ops like torch.zeros should work under C++ fake mode."""

        def f(x):
            z = torch.zeros(x.shape)
            return x + z

        with cpp_fake_tensor_mode():
            gm = make_fx(f, tracing_mode="real")(torch.randn(3, 4))
        x = torch.randn(3, 4)
        self.assertEqual(gm(x), f(x))

    def test_dtype_promotion(self):
        def f(x, y):
            return x + y

        with cpp_fake_tensor_mode():
            gm = make_fx(f, tracing_mode="real")(
                torch.randn(3, dtype=torch.float32),
                torch.randn(3, dtype=torch.float64),
            )
        x = torch.randn(3, dtype=torch.float32)
        y = torch.randn(3, dtype=torch.float64)
        self.assertEqual(gm(x, y), f(x, y))

    def test_broadcasting(self):
        def f(x, y):
            return x + y

        with cpp_fake_tensor_mode():
            gm = make_fx(f, tracing_mode="real")(torch.randn(3, 4), torch.randn(4))
        x, y = torch.randn(3, 4), torch.randn(4)
        self.assertEqual(gm(x, y), f(x, y))

    @unittest.skipIf(not HAS_CUDA, "CUDA-only test")
    def test_cuda_device(self):
        def f(x):
            return x.sin() + x.cos()

        with cpp_fake_tensor_mode():
            gm = make_fx(f, tracing_mode="real")(torch.randn(3, device="cuda"))
        x = torch.randn(3, device="cuda")
        self.assertEqual(gm(x), f(x))

    # --- Higher Order Op tests ---
    # These mirror the hop_db entries but call internal ops (cond_op, while_loop_op,
    # map_impl, scan_op) directly, bypassing the user-facing wrappers that route
    # through torch.compile/Dynamo.

    def _make_arg(self, *shape, low=0.1, high=2):
        return make_tensor(*shape, low=low, high=high, dtype=torch.float, device="cpu")

    def test_cond_simple(self):
        """Mirrors hop_db simple_cond."""
        from torch._higher_order_ops.cond import cond_op

        def f(x):
            return cond_op(x.sum() > 2, lambda x: (x.cos(),), lambda x: (x.sin(),), [x])

        self._test(f, (self._make_arg(2, 2, 2),), compare_graph=True)

    def test_while_loop_simple(self):
        """Mirrors hop_db simple_while_loop."""
        from torch._higher_order_ops.while_loop import while_loop_op

        def f(iter_t, x):
            def cond_fn(iter_t, x):
                return iter_t > 0

            def body_fn(iter_t, x):
                return iter_t - 1, x.cos()

            return while_loop_op(cond_fn, body_fn, (iter_t, x), ())

        self._test(f, (torch.tensor(3), self._make_arg(2, 3, 4)), compare_graph=True)

    def test_map_simple(self):
        """Mirrors hop_db simple_map."""
        from torch._higher_order_ops.map import map_impl

        def inner_f(x0, x1, y0, y1):
            return [x0.cos().add_(1.0) * y0, (x1 + y1.sin()).cos_().view(x1.size())]

        def f(x0, x1, y0, y1):
            return map_impl(inner_f, [x0, x1], (y0, y1))

        self._test(
            f,
            (
                self._make_arg(2, 2, 2),
                self._make_arg(2, 2, 2),
                self._make_arg(1),
                self._make_arg(1),
            ),
            compare_graph=True,
        )

    def test_scan_simple(self):
        """Mirrors hop_db simple_scan."""
        from torch._higher_order_ops.scan import scan_op

        def combine_fn(carry, x):
            result = carry @ x + x
            return result, carry.clone()

        def f(init, xs):
            return scan_op(combine_fn, [init], [xs], ())

        self._test(
            f, (self._make_arg(2, 2), self._make_arg(2, 2, 2)), compare_graph=True
        )


# --- OpInfo-based exhaustive tests for C++ FakeTensor mode ---

# Failures shared with the original make_fx tests (ops that don't work with
# proxy tensor tracing regardless of fake mode implementation).
cpp_fake_make_fx_failures = {
    # unknown
    xfail("allclose"),
    xfail("equal"),
    # empty
    skip("new_empty"),
    # skip('new_empty_strided'),
    skip("empty_like"),
    skip("empty"),
    skip("empty_permuted"),
    # flaky
    skip("linalg.lstsq", "grad_oriented"),
    skip("nn.functional.max_unpool1d", "", device_type="cpu"),
    skip("nn.functional.max_unpool2d", "", device_type="cpu"),
    skip("nn.functional.max_unpool3d", "", device_type="cpu"),
    skip("linalg.lstsq"),
    # data-dependent control flow
    skip("item"),
    xfail("cov"),
    xfail("nn.functional.gaussian_nll_loss"),
    xfail("corrcoef"),
    # sparse
    xfail("sparse.sampled_addmm"),
    xfail("sparse.mm", "reduce"),
    skip("to_sparse"),
    # segfaults
    skip("block_diag"),
    # AssertionError: Tensor-likes are not close!
    skip("empty_strided", "", device_type="cpu"),
}

cpp_fake_only_real_failures = {
    xfail("narrow"),
    xfail("tensor_split"),
}

cpp_fake_only_fake_failures = {
    xfail("tensor_split"),
}

# Failures specific to symbolic shapes under C++ fake mode.
# These mirror symbolic_tensor_failures from test_proxy_tensor.py.
cpp_fake_symbolic_failures = {
    xfail("combinations", ""),
    xfail("geqrf", ""),
    xfail("histogram", ""),
    xfail("histogramdd", ""),
    xfail("nn.functional.binary_cross_entropy", ""),
    xfail("nn.functional.cross_entropy", ""),
    xfail("nn.functional.ctc_loss"),
    xfail("max_pool2d_with_indices_backward", ""),
    skip("nn.functional.batch_norm"),
    skip(
        "tensor_split"
    ),  # reviSIT THIS !!!!!! this is xfail in python but in c++ it segfaults idk is it supposed to throw a real error
}


def _get_safe_inplace(inplace_variant):
    @functools.wraps(inplace_variant)
    def _fn(t, *args, **kwargs):
        return inplace_variant(t.clone(), *args, **kwargs)

    return _fn


def _test_make_fx_helper_cpp_fake(
    self, device, dtype, op, inplace=False, out=False, decomp_table=None
):
    """Like _test_make_fx_helper but wraps make_fx in cpp_fake_tensor_mode()."""
    fn = _get_safe_inplace(op.get_inplace()) if inplace else op.op
    sample_inputs_itr = op.sample_inputs(device, dtype, requires_grad=False)

    count = 100
    if out:
        count = 5
    for sample_input in itertools.islice(sample_inputs_itr, count):
        if inplace and sample_input.broadcasts_input:
            continue
        args = [sample_input.input] + list(sample_input.args)
        kwargs = sample_input.kwargs
        if out:
            expected = fn(*args, **kwargs)
            kwargs["out"] = expected

        try:
            _make_fx_check_cpp_fake(
                fn,
                args,
                kwargs,
                self.assertEqual,
                randomize_data=True,
                decomp_table=decomp_table,
            )
        except DynamicOutputShapeException:
            self.skipTest("Dynamic output shape operation in trace")


def _test_make_fx_helper_cpp_fake_symbolic(
    self, device, dtype, op, inplace=False, out=False
):
    """Like _test_make_fx_helper_cpp_fake but with symbolic shapes."""
    fn = _get_safe_inplace(op.get_inplace()) if inplace else op.op
    sample_inputs_itr = op.sample_inputs(device, dtype, requires_grad=False)

    count = 100
    if out:
        count = 5
    for sample_input in itertools.islice(sample_inputs_itr, count):
        if inplace and sample_input.broadcasts_input:
            continue
        args = [sample_input.input] + list(sample_input.args)
        kwargs = sample_input.kwargs
        if out:
            expected = fn(*args, **kwargs)
            kwargs["out"] = expected

        try:
            _make_fx_check_cpp_fake_symbolic(
                fn, args, kwargs, self.assertEqual, randomize_data=True
            )
        except DynamicOutputShapeException:
            self.skipTest("Dynamic output shape operation in trace")


_fake_counter = 0


def _to_cpp_fake(x):
    if isinstance(x, torch.Tensor):
        return torch._C._make_fake_tensor(x)
    return x


def _to_cpp_fake_symbolic(x):
    if not isinstance(x, torch.Tensor):
        return x
    global _fake_counter
    _fake_counter += 1
    from torch._dynamo.source import ConstantSource
    from torch.fx.experimental.symbolic_shapes import (
        DimDynamic,
        StatelessSymbolicContext,
    )

    source = ConstantSource(f"arg_{_fake_counter}")
    ctx = StatelessSymbolicContext(
        dynamic_sizes=[DimDynamic.DYNAMIC] * x.dim(),
    )
    result = torch._C._make_fake_tensor(x, source=source, symbolic_context=ctx)
    return result


def _make_fx_check_cpp_fake(
    func, args, kwargs, assert_close, randomize_data=False, decomp_table=None
):
    """Like optests.make_fx_check but traces under cpp_fake_tensor_mode()."""
    from torch.testing._internal.optests.make_fx import (
        handle_sizes_for_dynamic_shapes,
        randomize,
    )
    from torch.testing._utils import wrapper_set_seed

    f, *new_args = handle_sizes_for_dynamic_shapes(func, args, kwargs)

    def run(f, *args, **kwargs):
        return wrapper_set_seed(f, *args, **kwargs)

    with cpp_fake_tensor_mode():
        traced_f = make_fx(f, tracing_mode="real", decomposition_table=decomp_table)(
            *new_args
        )

    msg = (
        "op(*args, **kwargs) and make_fx(op)(*args, **kwargs) under "
        "cpp_fake_tensor_mode produced different values."
    )

    if randomize_data:
        new_args = randomize(new_args)
    try:
        expected = run(f, *new_args)
    except Exception:
        if randomize_data:
            return
        raise
    result = run(traced_f, *new_args)
    assert_close(result, expected, msg=msg)


def _make_fx_check_cpp_fake_symbolic(
    func, args, kwargs, assert_close, randomize_data=False, decomp_table=None
):
    """Like _make_fx_check_cpp_fake but with symbolic shapes."""
    from torch.testing._internal.optests.make_fx import (
        handle_sizes_for_dynamic_shapes,
        randomize,
    )
    from torch.testing._utils import wrapper_set_seed

    f, *new_args = handle_sizes_for_dynamic_shapes(func, args, kwargs)

    def run(f, *args, **kwargs):
        return wrapper_set_seed(f, *args, **kwargs)

    with cpp_fake_tensor_mode() as shape_env:
        from torch.fx.experimental.proxy_tensor import enable_python_dispatcher
        from torch.utils._pytree import tree_map_only

        symbolic_args = tree_map_only(torch.Tensor, _to_cpp_fake_symbolic, new_args)
        with enable_python_dispatcher():
            traced_f = make_fx(f, tracing_mode="real", decomposition_table=decomp_table)(
                *symbolic_args
            )

    msg = (
        "op(*args, **kwargs) and make_fx(op)(*args, **kwargs) under "
        "cpp_fake_tensor_mode (symbolic) produced different values."
    )

    if randomize_data:
        new_args = randomize(new_args)
    try:
        expected = run(f, *new_args)
    except Exception:
        if randomize_data:
            return
        raise
    result = run(traced_f, *new_args)
    assert_close(result, expected, msg=msg)


# HOPs whose user-facing wrappers (torch.cond, etc.) call torch.compile internally,
# creating a Python FakeTensorMode that conflicts with C++ fake mode.
# These are tested directly via internal ops in TestCppFakeProxyTensor.
_HOP_SKIP_USER_FACING = {
    "cond",
    "map",
    "scan",
    "while_loop",
    "while_loop_stack_output",
    "auto_functionalize",
}
filtered_hop_db = [op for op in hop_db if op.name not in _HOP_SKIP_USER_FACING]


cpp_fake_inplace_symbolic_failures = {
    xfail("float_power", ""),
}

cpp_fake_out_symbolic_failures = {
    xfail("_batch_norm_with_update", ""),
    xfail("_native_batch_norm_legit", ""),
    xfail("angle", ""),
    xfail("argmax", ""),
    xfail("argmin", ""),
    xfail("gather", ""),
    xfail("linalg.pinv", ""),
    xfail("linalg.pinv", "hermitian"),
    xfail("scatter_add", ""),
    xfail("scatter", ""),
    xfail("take_along_dim", ""),
    xfail("randn", ""),
    xfail("index_reduce", "prod"),
    xfail("index_reduce", "mean"),
    xfail("index_reduce", "amax"),
    xfail("index_reduce", "amin"),
    skip("nanmean", ""),
}


@unittest.skipIf(not torch._dynamo.is_dynamo_supported(), "Cond requires dynamo")
class TestCppFakeProxyTensorOpInfo(TestCase):
    """Exhaustive op tests under C++ FakeTensor mode."""

    @ops(op_db + filtered_hop_db + custom_op_db, allowed_dtypes=(torch.float,))
    @skipOps(
        "TestCppFakeProxyTensorOpInfo",
        "test_make_fx_exhaustive",
        cpp_fake_make_fx_failures | cpp_fake_only_real_failures,
    )
    def test_make_fx_exhaustive(self, device, dtype, op):
        _test_make_fx_helper_cpp_fake(self, device, dtype, op)

    @ops(op_db + filtered_hop_db + custom_op_db, allowed_dtypes=(torch.float,))
    @skipOps(
        "TestCppFakeProxyTensorOpInfo",
        "test_make_fx_fake_exhaustive",
        cpp_fake_make_fx_failures | cpp_fake_only_fake_failures,
    )
    def test_make_fx_fake_exhaustive(self, device, dtype, op):
        _test_make_fx_helper_cpp_fake(self, device, dtype, op)

    @ops(op_db + filtered_hop_db + custom_op_db, allowed_dtypes=(torch.float,))
    @skipOps(
        "TestCppFakeProxyTensorOpInfo",
        "test_make_fx_symbolic_exhaustive",
        cpp_fake_make_fx_failures | cpp_fake_symbolic_failures,
    )
    def test_make_fx_symbolic_exhaustive(self, device, dtype, op):
        _test_make_fx_helper_cpp_fake_symbolic(self, device, dtype, op)

    @ops(op_db + custom_op_db, allowed_dtypes=(torch.float,))
    @skipOps(
        "TestCppFakeProxyTensorOpInfo",
        "test_make_fx_symbolic_exhaustive_inplace",
        cpp_fake_make_fx_failures
        | cpp_fake_symbolic_failures
        | cpp_fake_inplace_symbolic_failures,
    )
    def test_make_fx_symbolic_exhaustive_inplace(self, device, dtype, op):
        if not op.get_inplace():
            self.skipTest("No inplace variable for this op")
        _test_make_fx_helper_cpp_fake_symbolic(
            self, device, dtype, op, inplace=True
        )

    @ops(op_db + custom_op_db, allowed_dtypes=(torch.float,))
    @skipOps(
        "TestCppFakeProxyTensorOpInfo",
        "test_make_fx_symbolic_exhaustive_out",
        cpp_fake_make_fx_failures
        | cpp_fake_symbolic_failures
        | cpp_fake_out_symbolic_failures,
    )
    def test_make_fx_symbolic_exhaustive_out(self, device, dtype, op):
        if not op.supports_out:
            self.skipTest("Op doesn't support out")
        _test_make_fx_helper_cpp_fake_symbolic(
            self, device, dtype, op, out=True
        )


only_for = ("cpu",)
instantiate_device_type_tests(
    TestCppFakeProxyTensorOpInfo, globals(), only_for=only_for
)


# --- Real-mode tests under C++ FakeTensor mode ---


class TestCppFakeRealProxyTensor(TestCase):
    def test_error_on_data_dependent_ops(self):
        def f():
            x = torch.randn([])
            y = torch.randn([])
            if not torch.allclose(x * y, y * x):
                raise AssertionError("x * y should equal y * x")
            z = float(x)
            z2 = float(y)

        with cpp_fake_tensor_mode():
            make_fx(f, tracing_mode="real", _error_on_data_dependent_ops=False)()

    def test_disable_torch_fn_metadata_mode(self):
        class MyModule(nn.Module):
            def forward(self, x):
                return torch.sin(x) + torch.cos(x)

        mod = MyModule()

        def fn(x):
            return mod(x)

        fn._orig_mod = mod

        with cpp_fake_tensor_mode():
            gm_with = make_fx(
                fn, tracing_mode="real", record_module_stack=True
            )(torch.randn(3))
        torch_fn_present = any(
            "torch_fn" in n.meta
            for n in gm_with.graph.nodes
            if n.op == "call_function"
        )
        self.assertTrue(
            torch_fn_present,
            "torch_fn metadata should be present by default",
        )

        with cpp_fake_tensor_mode():
            gm_without = make_fx(
                fn,
                tracing_mode="real",
                record_module_stack=True,
                _disable_torch_fn_metadata_mode=True,
            )(torch.randn(3))
        torch_fn_absent = all(
            "torch_fn" not in n.meta
            for n in gm_without.graph.nodes
            if n.op == "call_function"
        )
        self.assertTrue(
            torch_fn_absent,
            "torch_fn metadata should be absent when mode is disabled",
        )


# --- Symbolic Tracing Tests under C++ FakeTensor mode ---


def strip_end(s, suffix):
    if suffix and s.endswith(suffix):
        return s[: -len(suffix)]
    else:
        return s


def show_guards(gm):
    names = [strip_end(n, "_1") for n in fx_placeholder_targets(gm)]
    return "\n".join(
        gm.shape_env.produce_guards(
            fx_placeholder_vals(gm), names, _simplified=True, input_contexts=None
        )
    )


def _get_node(fx_g, cond):
    for n in fx_g.graph.nodes:
        if cond(n):
            return n
    raise AssertionError


def _get_free_symbols(shape_env):
    vars = tuple(shape_env.backed_var_to_val.keys())
    return len([var for var in vars if var not in shape_env.replacements])


def _cpp_fake_trace(f, *args):
    inps = [torch.randn(arg) for arg in args]
    with cpp_fake_tensor_mode() as shape_env:
        symbolic_inps = tree_map_only(
            torch.Tensor, _to_cpp_fake_symbolic, inps
        )
        with enable_python_dispatcher():
            gm = make_fx(f, tracing_mode="real")(*symbolic_inps)
    if not hasattr(gm, "shape_env") or gm.shape_env is None:
        gm.shape_env = shape_env
    return gm


class TestCppFakeSymbolicTracing(TestCase):
    def _make_fx_symbolic(self, f, *args, **make_fx_kwargs):
        """Trace f with symbolic shapes under C++ fake mode."""
        with cpp_fake_tensor_mode() as shape_env:
            symbolic_args = tree_map_only(
                torch.Tensor, _to_cpp_fake_symbolic, args
            )
            with enable_python_dispatcher():
                gm = make_fx(f, tracing_mode="real", **make_fx_kwargs)(*symbolic_args)
        if not hasattr(gm, "shape_env") or gm.shape_env is None:
            gm.shape_env = shape_env
        return gm

    def _test_dynamic(self, fn, trace_inputs, test_inputs, assert_eq=True):
        trace_inputs = [torch.randn(shape) for shape in trace_inputs]
        gm = self._make_fx_symbolic(fn, *trace_inputs)
        for input in test_inputs:
            input = [torch.randn(shape) for shape in input]
            rx, ry = gm(*input), fn(*input)
            if assert_eq:
                self.assertEqual(rx, ry)
        return gm

    def _assert_no_guards(self, fx_g, free_symbols):
        if _get_free_symbols(fx_g.shape_env) != free_symbols:
            raise AssertionError(
                f"free symbols mismatch: {fx_g.shape_env.backed_var_to_val}"
            )
        if len(fx_g.shape_env.get_nontrivial_guards()) != 0:
            raise AssertionError(
                f"expected no guards: {fx_g.shape_env.format_guards()}"
            )

    def test_debug_interpreter(self):
        from torch.library import _scoped_library

        with _scoped_library("foo", "DEF") as foo:
            foo.define("foo(Tensor self) -> Tensor")

            @torch.library.impl(foo, "foo", "CPU")
            def foo_cpu(x):
                return x.clone().T

            @torch.library.impl(foo, "foo", "Meta")
            def foo_meta(x):
                return x.clone()

            def f(x):
                return torch.ops.foo.foo.default(x)

            gm = self._make_fx_symbolic(f, torch.randn(2, 2))
            from torch._functorch.compilers import DebugInterpreter

            interp = DebugInterpreter(gm)

            self.assertRaisesRegex(
                AssertionError,
                r"3 != 1",
                lambda: interp.run(torch.randn(3, 3).T),
            )

            self.assertRaisesRegex(
                AssertionError,
                r"\(3, 1\) != \(1, 3\)",
                lambda: interp.run(torch.randn(3, 3)),
            )

    def test_int_input(self):
        def f(x, y):
            return x.view(y)

        r = str(self._make_fx_symbolic(f, torch.empty(3, 4), 12).code).strip()
        self.assertExpectedInline(
            r,
            """\
def forward(self, x_1, y_1):
    view = torch.ops.aten.view.default(x_1, [y_1]);  x_1 = y_1 = None
    return view""",
        )

    def test_resize_from_zero(self):
        def f(x, y):
            x.resize_(y.size(0))

        r = str(
            self._make_fx_symbolic(f, torch.empty(0), torch.empty(2)).code
        ).strip()
        self.assertExpectedInline(
            r,
            """\
def forward(self, x_1, y_1):
    sym_size_int = torch.ops.aten.sym_size.int(y_1, 0);  y_1 = None
    resize_ = torch.ops.aten.resize_.default(x_1, [sym_size_int]);  x_1 = sym_size_int = resize_ = None
    return None""",
        )

    def test_broadcast_shapes(self):
        def f(x, y):
            return torch.functional.broadcast_shapes(x.size(), y.size()[0])

        r = str(
            self._make_fx_symbolic(f, torch.empty(3, 1), torch.empty(5)).code
        ).strip()
        self.assertExpectedInline(
            r,
            """\
def forward(self, x_1, y_1):
    sym_size_int = torch.ops.aten.sym_size.int(x_1, 0);  x_1 = None
    sym_size_int_1 = torch.ops.aten.sym_size.int(y_1, 0);  y_1 = None
    return (sym_size_int, sym_size_int_1)""",
        )

    def test_deduped_shape(self):
        def f(s0, s1, x, y):
            return torch.functional.broadcast_shapes(
                x.size(), y.size()[0]
            ), torch.empty(x.shape[0])

        x = torch.empty(3, 1)
        y = torch.empty(5)

        with cpp_fake_tensor_mode() as shape_env:
            x_fake = _to_cpp_fake_symbolic(x)
            y_fake = _to_cpp_fake_symbolic(y)
            with enable_python_dispatcher():
                r = str(
                    make_fx(f, tracing_mode="real")(
                        x_fake.shape[0], y_fake.shape[0], x_fake, y_fake
                    ).code
                ).strip()
        self.assertExpectedInline(
            r,
            """\
def forward(self, s0_1, s1_1, x_1, y_1):
    empty = torch.ops.aten.empty.memory_format([s0_1], device = device(type='cpu'), pin_memory = False)
    return ((s0_1, s1_1), empty)""",
        )

    def test_non_deduped_shape(self):
        def f(x, y):
            return torch.functional.broadcast_shapes(
                x.size(), y.size()[0]
            ), torch.empty(x.shape[0])

        x = torch.empty(3, 1)
        y = torch.empty(5)

        with cpp_fake_tensor_mode() as shape_env:
            x_fake = _to_cpp_fake_symbolic(x)
            y_fake = _to_cpp_fake_symbolic(y)
            with enable_python_dispatcher():
                r = str(
                    make_fx(f, tracing_mode="real")(x_fake, y_fake).code
                ).strip()
        self.assertExpectedInline(
            r,
            """\
def forward(self, x_1, y_1):
    sym_size_int = torch.ops.aten.sym_size.int(x_1, 0);  x_1 = None
    sym_size_int_1 = torch.ops.aten.sym_size.int(y_1, 0);  y_1 = None
    empty = torch.ops.aten.empty.memory_format([sym_size_int], device = device(type='cpu'), pin_memory = False)
    return ((sym_size_int, sym_size_int_1), empty)""",
        )

    def test_unary(self):
        def f(x):
            if x.shape[0] >= 20:
                raise AssertionError(
                    f"expected x.shape[0] < 20, got {x.shape[0]}"
                )
            return x.cos()

        test_inputs = []
        test_inputs.append([(2, 5)])
        test_inputs.append([(6, 8)])
        gm = self._test_dynamic(f, [(3, 4)], test_inputs)
        self.assertTrue(eval_guards(gm, torch.randn(4, 5)))
        self.assertFalse(eval_guards(gm, torch.randn(25, 5)))
        self.assertExpectedInline(
            show_guards(gm), """L['x'].size()[0] <= 19"""
        )

    def test_repeat_interleave(self):
        def f(src_tokens, beam_size_src):
            return src_tokens.repeat_interleave(beam_size_src.size(0), 0)

        prompt_size = 64
        vocab_size = 64
        batch_size = 4
        src_tokens = torch.randint(1, vocab_size, (batch_size, prompt_size))
        gm = self._make_fx_symbolic(f, src_tokens, torch.randn(5))
        self.assertEqual(len(gm.shape_env.guards), 0)

    def test_non_symint_size_spec(self):
        def f(x):
            torch._C._non_sym_sizes(x)
            return x + 1

        x = torch.randn(2, 3)
        self._make_fx_symbolic(f, x)

    def test_symbolic_repeat_interleave(self):
        def f(y, x):
            return y.repeat_interleave(x, dim=1)

        y = torch.tensor([[1, 2], [3, 4]])
        x = torch.tensor([2, 3])
        r = str(self._make_fx_symbolic(f, y, x).code).strip()
        self.assertExpectedInline(
            r,
            """\
def forward(self, y_1, x_1):
    repeat_interleave = torch.ops.aten.repeat_interleave.Tensor(x_1);  x_1 = None
    index_select = torch.ops.aten.index_select.default(y_1, 1, repeat_interleave);  y_1 = repeat_interleave = None
    return index_select""",
        )

    def test_mod_gcd_unbacked(self):
        def f(_a, _b, _stride):
            a = _a.item()
            b = _b.item()
            stride = _stride.item()
            ta = torch.randn(a * stride)
            tb = torch.randn(b * stride)
            r = torch.cat([ta, tb])
            return r.view(a + b, stride)

        _a = torch.tensor(30)
        _b = torch.tensor(20)
        _stride = torch.tensor(10)
        r = str(self._make_fx_symbolic(f, _a, _b, _stride).code).strip()
        self.assertExpectedInline(
            r,
            """\
def forward(self, _a_1, _b_1, _stride_1):
    _local_scalar_dense = torch.ops.aten._local_scalar_dense.default(_a_1);  _a_1 = None
    _local_scalar_dense_1 = torch.ops.aten._local_scalar_dense.default(_b_1);  _b_1 = None
    _local_scalar_dense_2 = torch.ops.aten._local_scalar_dense.default(_stride_1);  _stride_1 = None
    mul = _local_scalar_dense * _local_scalar_dense_2
    randn = torch.ops.aten.randn.default([mul], device = device(type='cpu'), pin_memory = False);  mul = None
    mul_1 = _local_scalar_dense_1 * _local_scalar_dense_2
    randn_1 = torch.ops.aten.randn.default([mul_1], device = device(type='cpu'), pin_memory = False);  mul_1 = None
    cat = torch.ops.aten.cat.default([randn, randn_1]);  randn = randn_1 = None
    add = _local_scalar_dense + _local_scalar_dense_1;  _local_scalar_dense = _local_scalar_dense_1 = None
    view = torch.ops.aten.view.default(cat, [add, _local_scalar_dense_2]);  cat = add = _local_scalar_dense_2 = None
    return view""",
        )

    def test_cumsum_unbacked(self):
        def f(x):
            y = x.item()
            z = torch.randn((3, y, 3))
            return z.cumsum(0)

        r = str(
            self._make_fx_symbolic(f, torch.tensor([5])).code
        ).strip()
        self.assertExpectedInline(
            r,
            """\
def forward(self, x_1):
    _local_scalar_dense = torch.ops.aten._local_scalar_dense.default(x_1);  x_1 = None
    randn = torch.ops.aten.randn.default([3, _local_scalar_dense, 3], device = device(type='cpu'), pin_memory = False);  _local_scalar_dense = None
    cumsum = torch.ops.aten.cumsum.default(randn, 0);  randn = None
    return cumsum""",
        )

    def test_repeat_interleave_unbacked_output_size(self):
        def f(x, y):
            s = x.sum().item()
            return y.repeat_interleave(x, dim=0, output_size=s)

        r = str(
            self._make_fx_symbolic(
                f, torch.tensor([2, 3]), torch.randn(2)
            ).code
        ).strip()
        self.assertExpectedInline(
            r,
            """\
def forward(self, x_1, y_1):
    sum_1 = torch.ops.aten.sum.default(x_1)
    _local_scalar_dense = torch.ops.aten._local_scalar_dense.default(sum_1);  sum_1 = None
    repeat_interleave = torch.ops.aten.repeat_interleave.Tensor(x_1, output_size = _local_scalar_dense);  x_1 = _local_scalar_dense = None
    index_select = torch.ops.aten.index_select.default(y_1, 0, repeat_interleave);  y_1 = repeat_interleave = None
    return index_select""",
        )

    def test_arange_unbacked_output_size(self):
        def f(x):
            return torch.arange(0, x)

        r = str(
            self._make_fx_symbolic(f, torch.tensor(10)).code
        ).strip()
        self.assertExpectedInline(
            r,
            """\
def forward(self, x_1):
    _local_scalar_dense = torch.ops.aten._local_scalar_dense.default(x_1);  x_1 = None
    arange = torch.ops.aten.arange.start(0, _local_scalar_dense, device = device(type='cpu'), pin_memory = False);  _local_scalar_dense = None
    return arange""",
        )

    def test_adv_index_batch(self):
        def f(src_tokens):
            bsz, src_len = src_tokens.size()[:2]
            start_step = src_tokens.shape[1]
            beam_size = 1
            generate_size = 64
            max_len = src_len + generate_size
            tokens = (
                torch.zeros(bsz * beam_size, max_len).to(src_tokens).long().fill_(0)
            )
            tokens[:, :start_step] = src_tokens.repeat_interleave(beam_size, 0)
            return tokens

        prompt_size = 64
        vocab_size = 64
        batch_size = 4
        src_tokens = torch.randint(1, vocab_size, (batch_size, prompt_size))
        gm = self._make_fx_symbolic(f, src_tokens)
        self.assertEqual(len(gm.shape_env.guards), 0)

    @unittest.skipIf(not HAS_CUDA, "CUDA-only test")
    def test_cpu_scalar_cuda(self):
        def f(a, b):
            return (a * b) @ b

        r = str(
            self._make_fx_symbolic(
                f, torch.tensor(1.0), torch.randn(2, 2, device="cuda")
            ).code
        ).strip()
        self.assertExpectedInline(
            r,
            """\
def forward(self, a_1, b_1):
    mul = torch.ops.aten.mul.Tensor(a_1, b_1);  a_1 = None
    mm = torch.ops.aten.mm.default(mul, b_1);  mul = b_1 = None
    return mm""",
        )

    def test_binary_broadcast(self):
        def f(a, b):
            c = a * b
            return c

        test_inputs = []
        test_inputs.append([(1, 5), (3, 1)])
        test_inputs.append([(1, 4), (4, 1)])
        shape_env = self._test_dynamic(
            f, [(1, 2), (3, 1)], test_inputs
        ).shape_env
        if len(shape_env.guards) != 0:
            raise AssertionError(
                f"expected no guards, got {len(shape_env.guards)}"
            )

    def test_multiply_shape(self):
        def f(a):
            return torch.empty(a.shape[0] * 2)

        r = str(self._make_fx_symbolic(f, torch.empty(4)).code).strip()
        self.assertExpectedInline(
            r,
            """\
def forward(self, a_1):
    sym_size_int = torch.ops.aten.sym_size.int(a_1, 0);  a_1 = None
    mul = sym_size_int * 2;  sym_size_int = None
    empty = torch.ops.aten.empty.memory_format([mul], device = device(type='cpu'), pin_memory = False);  mul = None
    return empty""",
        )

    def test_item(self):
        def f(a):
            r = a.item()
            return r * a

        r = str(self._make_fx_symbolic(f, torch.randn(1)).code).strip()
        self.assertExpectedInline(
            r,
            """\
def forward(self, a_1):
    _local_scalar_dense = torch.ops.aten._local_scalar_dense.default(a_1)
    mul = torch.ops.aten.mul.Tensor(a_1, _local_scalar_dense);  a_1 = _local_scalar_dense = None
    return mul""",
        )

    def test_tensor_symfloat(self):
        def f(a):
            r = torch.tensor(a.size(0) ** 2.0)
            if r.dtype is not torch.float:
                raise AssertionError(f"expected dtype torch.float, got {r.dtype}")
            return r

        gm = self._make_fx_symbolic(f, torch.randn(2))
        r = str(gm.code).strip()
        self.assertExpectedInline(
            r,
            """\
def forward(self, a_1):
    _tensor_constant0 = self._tensor_constant0
    lift_fresh_copy = torch.ops.aten.lift_fresh_copy.default(_tensor_constant0);  _tensor_constant0 = None
    return lift_fresh_copy""",
        )
        self.assertEqual(gm._tensor_constant0, torch.tensor(4.0))

    def test_item_to_constructor(self):
        def f(a):
            r = a.item()
            return torch.empty(r)

        r = str(
            self._make_fx_symbolic(f, torch.randint(5, (1,))).code
        ).strip()
        self.assertExpectedInline(
            r,
            """\
def forward(self, a_1):
    _local_scalar_dense = torch.ops.aten._local_scalar_dense.default(a_1);  a_1 = None
    empty = torch.ops.aten.empty.memory_format([_local_scalar_dense], device = device(type='cpu'), pin_memory = False);  _local_scalar_dense = None
    return empty""",
        )

    def test_setitem_symint(self):
        def f(x):
            x[0] = x.size(0)
            return x

        r = str(self._make_fx_symbolic(f, torch.randn(10)).code).strip()
        self.assertExpectedInline(
            r,
            """\
def forward(self, x_1):
    sym_size_int = torch.ops.aten.sym_size.int(x_1, 0)
    scalar_tensor = torch.ops.aten.scalar_tensor.default(sym_size_int, dtype = torch.float32, layout = torch.strided, device = device(type='cpu'));  sym_size_int = None
    select = torch.ops.aten.select.int(x_1, 0, 0)
    copy_ = torch.ops.aten.copy_.default(select, scalar_tensor);  select = scalar_tensor = copy_ = None
    return x_1""",
        )

    def test_dynamic_pointwise_scalar(self):
        def f(gravity, mask):
            gravity[mask, 0] = gravity[mask, 0] * -1

        r = str(
            self._make_fx_symbolic(
                f,
                torch.randn((12, 4)),
                torch.randint(0, 2, (12,), dtype=torch.bool),
            ).code
        ).strip()
        self.assertExpectedInline(
            r,
            """\
def forward(self, gravity_1, mask_1):
    select = torch.ops.aten.select.int(gravity_1, 1, 0)
    index = torch.ops.aten.index.Tensor(select, [mask_1]);  select = None
    mul = torch.ops.aten.mul.Tensor(index, -1);  index = None
    select_1 = torch.ops.aten.select.int(gravity_1, 1, 0);  gravity_1 = None
    index_put_ = torch.ops.aten.index_put_.default(select_1, [mask_1], mul);  select_1 = mask_1 = mul = index_put_ = None
    return None""",
        )

    def test_reflect_r_over_x(self):
        def reflect_R_over_x(R):
            reflect = torch.eye(3, device=R.device)
            reflect[0, 0] = -1
            return reflect @ R @ reflect

        def f(crop_camera, mask):
            crop_camera[mask] = reflect_R_over_x(crop_camera[mask])

        r = str(
            self._make_fx_symbolic(
                f,
                torch.randn((12, 3, 3)),
                torch.randint(0, 2, (12,), dtype=torch.bool),
            ).code
        ).strip()
        self.assertExpectedInline(
            r,
            """\
def forward(self, crop_camera_1, mask_1):
    index = torch.ops.aten.index.Tensor(crop_camera_1, [mask_1])
    eye = torch.ops.aten.eye.default(3, device = device(type='cpu'), pin_memory = False)
    _tensor_constant0 = self._tensor_constant0
    lift_fresh_copy = torch.ops.aten.lift_fresh_copy.default(_tensor_constant0);  _tensor_constant0 = None
    select = torch.ops.aten.select.int(eye, 0, 0)
    select_1 = torch.ops.aten.select.int(select, 0, 0);  select = None
    copy_ = torch.ops.aten.copy_.default(select_1, lift_fresh_copy);  select_1 = lift_fresh_copy = copy_ = None
    sym_size_int = torch.ops.aten.sym_size.int(index, 0)
    expand = torch.ops.aten.expand.default(eye, [sym_size_int, 3, 3])
    view = torch.ops.aten.view.default(expand, [sym_size_int, 3, 3]);  expand = None
    sym_size_int_1 = torch.ops.aten.sym_size.int(crop_camera_1, 1)
    sym_size_int_2 = torch.ops.aten.sym_size.int(crop_camera_1, 2)
    expand_1 = torch.ops.aten.expand.default(index, [sym_size_int, sym_size_int_1, sym_size_int_2]);  index = None
    view_1 = torch.ops.aten.view.default(expand_1, [sym_size_int, sym_size_int_1, sym_size_int_2]);  expand_1 = sym_size_int_1 = sym_size_int_2 = None
    bmm = torch.ops.aten.bmm.default(view, view_1);  view = view_1 = None
    view_2 = torch.ops.aten.view.default(bmm, [sym_size_int, 3, 3]);  bmm = None
    mul_11 = sym_size_int * 3
    view_3 = torch.ops.aten.view.default(view_2, [mul_11, 3]);  view_2 = mul_11 = None
    mm = torch.ops.aten.mm.default(view_3, eye);  view_3 = eye = None
    _unsafe_view = torch.ops.aten._unsafe_view.default(mm, [sym_size_int, 3, 3]);  mm = sym_size_int = None
    index_put_ = torch.ops.aten.index_put_.default(crop_camera_1, [mask_1], _unsafe_view);  crop_camera_1 = mask_1 = _unsafe_view = index_put_ = None
    return None""",
        )

    def test_unbacked_slice(self):
        def f(x, m):
            x = x[m]
            return x[
                slice(None, None, None),
                slice(None, None, None),
                slice(None, 2, None),
            ]

        self._make_fx_symbolic(
            f,
            torch.randn((12, 3, 3)),
            torch.randint(0, 2, (12,), dtype=torch.bool),
        )

    @unittest.skipIf(not USE_TORCHVISION, "test requires torchvision")
    def test_unbacked_batch_resnet(self):
        mod = torchvision.models.resnet18()

        def f(x, mask, params, buffers):
            for p in itertools.chain(
                [x, mask], params.values(), buffers.values()
            ):
                for s in p.shape:
                    guard_int(s)
            x = x[mask]
            torch._check(x.shape[0] >= 1)
            for p in params.values():
                p.grad = None
            return torch.func.functional_call(
                mod, {**params, **buffers}, (x,)
            ).sum()

        self._make_fx_symbolic(
            f,
            torch.randn(3, 3, 250, 250),
            torch.randint(0, 2, (3,), dtype=torch.bool),
            dict(mod.named_parameters()),
            dict(mod.named_buffers()),
        )

    def test_boolean_index(self):
        def f(images, handedness, valid):
            images = images[valid]
            handedness = handedness[valid]
            right_hand_mask = handedness == 1
            images[right_hand_mask] = images[right_hand_mask].flip(-1)

        r = str(
            self._make_fx_symbolic(
                f,
                torch.randint(0, 256, (512, 1, 96, 96)),
                torch.randint(0, 1, (512,)),
                torch.randint(0, 2, (512,), dtype=torch.bool),
            ).code
        ).strip()
        self.assertExpectedInline(
            r,
            """\
def forward(self, images_1, handedness_1, valid_1):
    index = torch.ops.aten.index.Tensor(images_1, [valid_1]);  images_1 = None
    index_1 = torch.ops.aten.index.Tensor(handedness_1, [valid_1]);  handedness_1 = valid_1 = None
    eq = torch.ops.aten.eq.Scalar(index_1, 1);  index_1 = None
    index_2 = torch.ops.aten.index.Tensor(index, [eq])
    flip = torch.ops.aten.flip.default(index_2, [-1]);  index_2 = None
    index_put_ = torch.ops.aten.index_put_.default(index, [eq], flip);  index = eq = flip = index_put_ = None
    return None""",
        )

    def test_neg_shape(self):
        def f(a):
            return torch.empty(-a.shape[0] + 10)

        r = str(self._make_fx_symbolic(f, torch.empty(2)).code).strip()
        self.assertExpectedInline(
            r,
            """\
def forward(self, a_1):
    sym_size_int = torch.ops.aten.sym_size.int(a_1, 0);  a_1 = None
    neg = -sym_size_int;  sym_size_int = None
    add = neg + 10;  neg = None
    empty = torch.ops.aten.empty.memory_format([add], device = device(type='cpu'), pin_memory = False);  add = None
    return empty""",
        )

    def test_unbacked_unification(self):
        def f(x, y):
            z = torch.zeros(x.item())
            return z + y

        r = str(
            self._make_fx_symbolic(
                f, torch.tensor(10), torch.randn(10)
            ).code
        ).strip()
        self.assertExpectedInline(
            r,
            """\
def forward(self, x_1, y_1):
    _local_scalar_dense = torch.ops.aten._local_scalar_dense.default(x_1);  x_1 = None
    zeros = torch.ops.aten.zeros.default([_local_scalar_dense], device = device(type='cpu'), pin_memory = False);  _local_scalar_dense = None
    add = torch.ops.aten.add.Tensor(zeros, y_1);  zeros = y_1 = None
    return add""",
        )

    def test_reshape_divisibility_unbacked(self):
        def f(x):
            i0 = x.item()
            r = torch.zeros(i0, 4, 20)
            r = r.transpose(2, 1)
            return r.reshape(-1, 80)

        self._make_fx_symbolic(f, torch.tensor(24))

    def test_view_divisibility_unbacked(self):
        def f(x):
            i0 = x.item()
            r = torch.zeros(i0, 192)
            return r.view(12, -1, 192)

        self._make_fx_symbolic(f, torch.tensor(24))

    @unittest.skipIf(not HAS_CUDA, "CUDA-only test")
    def test_view_divisibility_unbacked_relatively_prime(self):
        def f(x):
            i0 = x.item()
            torch._check(i0 > 0)
            torch._check(i0 <= 448)
            return torch.zeros(256 * i0).view(-1, 447)

        self._make_fx_symbolic(
            f, torch.tensor(256 * 447, device="cuda")
        )

    def test_unbacked_unify_guard(self):
        def f(x, y):
            z = torch.zeros(x.item())
            torch._check(z.size(0) == y.size(0))
            if z.size(0) == 4:
                return y * 2
            else:
                return y + 2

        r = str(
            self._make_fx_symbolic(
                f, torch.tensor(10), torch.randn(10)
            ).code
        ).strip()
        self.assertExpectedInline(
            r,
            """\
def forward(self, x_1, y_1):
    _local_scalar_dense = torch.ops.aten._local_scalar_dense.default(x_1);  x_1 = None
    zeros = torch.ops.aten.zeros.default([_local_scalar_dense], device = device(type='cpu'), pin_memory = False);  _local_scalar_dense = zeros = None
    add = torch.ops.aten.add.Tensor(y_1, 2);  y_1 = None
    return add""",
        )

    @unittest.skipIf(not HAS_CUDA, "CUDA-only test")
    @unittest.expectedFailure
    def test_unbacked_unify_guard_transitivity(self):
        def f(x1, x2, y):
            z1 = torch.zeros(x1.item())
            z2 = torch.zeros(x2.item())
            torch._check(z1.size(0) == z2.size(0))
            torch._check(z2.size(0) == y.size(0))
            if z1.size(0) == 4:
                return y * 2
            else:
                return y + 2

        gm = self._make_fx_symbolic(
            f,
            torch.tensor(10, device="cuda"),
            torch.tensor(10, device="cuda"),
            torch.randn(10, device="cuda"),
        )
        insert_deferred_runtime_asserts(gm, gm.shape_env, "test")
        gm.recompile()
        r = str(gm.code).strip()

    @unittest.skipIf(not HAS_CUDA, "CUDA-only test")
    def test_unbacked_unify_dependency_violation(self):
        def f(x1, x2, x3, y):
            z1 = x1.item()
            torch._check(z1 // 9 == 1)
            z2 = x2.item()
            z3 = x3.item()
            torch._check(z1 == z2 + z3)
            return y * 2

        gm = self._make_fx_symbolic(
            f,
            torch.tensor(10, device="cuda"),
            torch.tensor(5, device="cuda"),
            torch.tensor(5, device="cuda"),
            torch.randn(1, device="cuda"),
        )
        insert_deferred_runtime_asserts(gm, gm.shape_env, "test")
        gm.recompile()
        self.assertEqual(
            gm(
                torch.tensor(12, device="cuda"),
                torch.tensor(6, device="cuda"),
                torch.tensor(6, device="cuda"),
                torch.tensor([1.0], device="cuda"),
            ),
            torch.tensor([2.0], device="cuda"),
        )
        with self.assertRaises(RuntimeError):
            gm(
                torch.tensor(20, device="cuda"),
                torch.tensor(10, device="cuda"),
                torch.tensor(10, device="cuda"),
                torch.tensor([1.0], device="cuda"),
            )

    def test_split_unbacked_sizes(self):
        def f(lengths, values):
            sizes = [lengths[i].item() for i in range(lengths.size(0))]
            return torch.split(values, sizes)

        r = str(
            self._make_fx_symbolic(
                f, torch.tensor([2, 3, 4]), torch.randn(9)
            ).code
        ).strip()
        self.assertExpectedInline(
            r,
            """\
def forward(self, lengths_1, values_1):
    select = torch.ops.aten.select.int(lengths_1, 0, 0)
    _local_scalar_dense = torch.ops.aten._local_scalar_dense.default(select);  select = None
    select_1 = torch.ops.aten.select.int(lengths_1, 0, 1)
    _local_scalar_dense_1 = torch.ops.aten._local_scalar_dense.default(select_1);  select_1 = None
    select_2 = torch.ops.aten.select.int(lengths_1, 0, 2);  lengths_1 = None
    _local_scalar_dense_2 = torch.ops.aten._local_scalar_dense.default(select_2);  select_2 = None
    split_with_sizes = torch.ops.aten.split_with_sizes.default(values_1, [_local_scalar_dense, _local_scalar_dense_1, _local_scalar_dense_2]);  values_1 = _local_scalar_dense = _local_scalar_dense_1 = _local_scalar_dense_2 = None
    getitem = split_with_sizes[0]
    getitem_1 = split_with_sizes[1]
    getitem_2 = split_with_sizes[2];  split_with_sizes = None
    return (getitem, getitem_1, getitem_2)""",
        )

    def test_invalidate_nonzero(self):
        ok = False

        def f(a):
            nonlocal ok
            b = a.clone()
            x = b.nonzero()
            x1 = b.nonzero()
            x2 = b.nonzero()
            if x1.shape[0] != x2.shape[0]:
                raise AssertionError("x1.shape[0] should equal x2.shape[0]")
            ok = True
            b.normal_()
            y = b.nonzero()
            try:
                bool(x1.shape[0] == y.shape[0])
                self.fail("didn't raise exception")
            except GuardOnDataDependentSymNode:
                pass

        self._make_fx_symbolic(f, torch.randn(4))

    @torch._functorch.config.patch(fake_tensor_propagate_real_tensors=True)
    def test_invalidate_nonzero_propagate_real_tensors(self):
        def f(a):
            b = a.clone()
            x = b.nonzero()
            x1 = b.nonzero()
            x2 = b.nonzero()
            if x1.shape[0] != x2.shape[0]:
                raise AssertionError("x1.shape[0] should equal x2.shape[0]")
            b.normal_()
            y = b.nonzero()
            if x1.shape[0] != y.shape[0]:
                raise AssertionError("x1.shape[0] should equal y.shape[0]")

        self._make_fx_symbolic(f, torch.randn(4))

    def test_sqrt_size(self):
        def f(a):
            return a / a.size(-1) ** 0.5

        r = str(self._make_fx_symbolic(f, torch.empty(4)).code).strip()
        self.assertExpectedInline(
            r,
            """\
def forward(self, a_1):
    sym_size_int = torch.ops.aten.sym_size.int(a_1, 0)
    sym_float = torch.sym_float(sym_size_int);  sym_size_int = None
    pow_1 = sym_float ** 0.5;  sym_float = None
    div = torch.ops.aten.div.Tensor(a_1, pow_1);  a_1 = pow_1 = None
    return div""",
        )

    def test_make_fx_with_custom_tracer_preserving_nn_module_stack(self):
        class Bar(torch.nn.Module):
            def __init__(self) -> None:
                super().__init__()

            def forward(self, x):
                return x + 1

        class Foo(torch.nn.Module):
            def __init__(self) -> None:
                super().__init__()
                self.bar = Bar()

            def forward(self, x):
                return x + self.bar(x)

        with cpp_fake_tensor_mode():
            gm = make_fx(Foo(), tracing_mode="real")(torch.randn(4, 4))
        for node in gm.graph.nodes:
            self.assertTrue("nn_module_stack" not in node.meta)

        foo = Foo()

        def functional_call(*args, **kwargs):
            with stateless._reparametrize_module(foo, {}):
                return foo(*args, **kwargs)

        functional_call._orig_mod = foo

        with cpp_fake_tensor_mode():
            gm_with_stack = make_fx(
                functional_call, tracing_mode="real", record_module_stack=True
            )(torch.randn(4, 4))
        found = False
        for node in gm_with_stack.graph.nodes:
            if "nn_module_stack" in node.meta:
                if len(node.meta["nn_module_stack"]) == 1:
                    self.assertTrue(
                        "custom_tracer_preserving_nn_module_stack.<locals>.Foo"
                        in str(node.meta["nn_module_stack"])
                    )
                    found = True
                elif len(node.meta["nn_module_stack"]) == 2:
                    self.assertTrue(
                        "preserving_nn_module_stack.<locals>.Bar"
                        in str(node.meta["nn_module_stack"])
                    )
                    found = True
                else:
                    self.assertTrue(False)
        self.assertTrue(found)

        with cpp_fake_tensor_mode():
            gm_without_stack = make_fx(
                functional_call, tracing_mode="real"
            )(torch.randn(4, 4))
        for node in gm_without_stack.graph.nodes:
            self.assertTrue("nn_module_stack" not in node.meta)

    def test_symint_to_tensor(self):
        def f(a):
            return a / a.shape[0]

        r = str(self._make_fx_symbolic(f, torch.empty(4)).code).strip()
        self.assertExpectedInline(
            r,
            """\
def forward(self, a_1):
    sym_size_int = torch.ops.aten.sym_size.int(a_1, 0)
    div = torch.ops.aten.div.Tensor(a_1, sym_size_int);  a_1 = sym_size_int = None
    return div""",
        )

        r = str(
            self._make_fx_symbolic(
                f, torch.empty(4), decomposition_table=decomposition_table
            ).code
        ).strip()
        self.assertExpectedInline(
            r,
            """\
def forward(self, a_1):
    sym_size_int = torch.ops.aten.sym_size.int(a_1, 0)
    sym_float = torch.sym_float(sym_size_int);  sym_size_int = None
    div = torch.ops.prims.div.default(a_1, sym_float);  a_1 = sym_float = None
    return div""",
        )

    def test_cat(self):
        def f(a, b):
            val = torch.mul(a, b)
            out = torch.cat([val, val])
            if out.shape[0] * out.shape[1] > 20:
                out = out.cos()
            return out

        test_inputs = []
        test_inputs.append([(1, 5), (6, 1)])
        test_inputs.append([(1, 4), (3, 1)])
        gm = self._test_dynamic(f, [(1, 6), (8, 1)], test_inputs)
        self.assertTrue(eval_guards(gm, torch.randn(1, 10), torch.randn(6, 1)))
        self.assertFalse(eval_guards(gm, torch.randn(1, 2), torch.randn(4, 1)))
        self.assertExpectedInline(
            show_guards(gm),
            """2*L['b'].size()[0]*L['a'].size()[1] > 20""",
        )

    def test_new_empty(self):
        def f(a, b):
            return a.new_empty(b.shape[0], b.shape[1] * 2)

        self._test_dynamic(
            f,
            [(2, 4), (4, 5)],
            [[(2, 3), (5, 7)], [(3, 7), (9, 3)]],
            assert_eq=False,
        ).shape_env

    def test_size_with_tensor(self):
        def f(tensor):
            max_size = torch.tensor([800, 1216], dtype=torch.int64)
            batch_shape = [2] + list(tensor.shape[:-2]) + list(max_size)
            return tensor.new_empty(batch_shape)

        a = torch.randn(3, 800, 1199)
        f(a)
        self._make_fx_symbolic(f, a)

    def test_fake_tensor_as_size(self):
        def f(x):
            r = torch.zeros([x])
            return r

        fx_g = self._make_fx_symbolic(f, torch.tensor(4))
        self.assertExpectedInline(
            fx_g.code.strip(),
            """\
def forward(self, x_1):
    _local_scalar_dense = torch.ops.aten._local_scalar_dense.default(x_1);  x_1 = None
    zeros = torch.ops.aten.zeros.default([_local_scalar_dense], device = device(type='cpu'), pin_memory = False);  _local_scalar_dense = None
    return zeros""",
        )

    def test_expand(self):
        def f(a):
            b = torch.mul(a, a)
            c = b.expand(a.shape)
            return c

        self._test_dynamic(f, [(3,)], [[(3,)], [(4,)], [(2,)]])
        self._test_dynamic(f, [(5, 1)], [[(4, 1)], [(3, 1)], [(6, 1)]])

    def test_metadata(self):
        def f(a, b):
            d = a.new_empty(a.shape[0] + b.shape[0])
            return d

        fx_g = self._make_fx_symbolic(f, torch.randn(5), torch.randn(4))
        meta_c = _get_node(
            fx_g, lambda x: x.target == aten.new_empty.default
        )
        meta_d = _get_node(fx_g, lambda x: x.target == operator.add)
        self.assertTrue(
            meta_c.meta["val"].shape[0].node.expr
            == meta_d.meta["val"].node.expr
        )

    def test_metadata_fresh(self):
        def f(x):
            if x.shape[0] != 3:
                raise AssertionError(
                    f"expected x.shape[0] == 3, got {x.shape[0]}"
                )
            return x.cos()

        fx_g = self._make_fx_symbolic(f, torch.randn(3))
        meta_cos = _get_node(
            fx_g, lambda x: x.target == aten.cos.default
        )
        meta_inp = _get_node(fx_g, lambda x: x.op == "placeholder")
        self.assertTrue(meta_cos.meta["val"].shape[0] == 3)
        self.assertTrue(meta_inp.meta["val"].shape[0] == 3)

    def test_elementwise_meta_with_sym_numbers(self):
        def f(x, offset, as_sym_float=False):
            x0 = x.size()[0]
            if as_sym_float:
                x0 = torch.sym_float(x0)
            return torch.add(x0, offset)

        fx_g = self._make_fx_symbolic(f, torch.rand(2, 3), 2.0, False)
        meta_add = _get_node(
            fx_g, lambda x: x.target == aten.add.Tensor
        )
        self.assertEqual(meta_add.meta["val"].shape, ())
        self.assertEqual(meta_add.meta["val"].dtype, torch.float32)

        fx_g = self._make_fx_symbolic(f, torch.rand(2, 3), 2, False)
        meta_add = _get_node(
            fx_g, lambda x: x.target == aten.add.Tensor
        )
        self.assertEqual(meta_add.meta["val"].shape, ())
        self.assertEqual(meta_add.meta["val"].dtype, torch.int64)

        fx_g = self._make_fx_symbolic(f, torch.rand(2, 3), 2, True)
        meta_add = _get_node(
            fx_g, lambda x: x.target == aten.add.Tensor
        )
        self.assertEqual(meta_add.meta["val"].shape, ())
        self.assertEqual(meta_add.meta["val"].dtype, torch.float32)

    def test_return_symint(self):
        def f(x):
            return x.shape[0], x.cos(), x.shape[0] / 5

        self._test_dynamic(f, [(5,)], [[(4,)], [(12,)]])

        def f(x):
            return x.shape

        self._test_dynamic(f, [(5, 3)], [[(4, 6)]])

    def test_rmethod(self):
        def f(x):
            return x.size(0) + x

        self._test_dynamic(f, [(5,)], [[(4,)], [(12,)]])

    def test_mega_guard(self):
        def f(a, b):
            if a.shape[0] != b.shape[0] * 2:
                raise AssertionError("a.shape[0] should equal b.shape[0] * 2")
            return a.cos()

        fx_g = self._make_fx_symbolic(f, torch.randn(16), torch.randn(8))
        from torch._dynamo.source import LocalSource

        self.assertExpectedInline(
            str(
                fx_g.shape_env.produce_guards(
                    fx_placeholder_vals(fx_g),
                    [LocalSource("a"), LocalSource("b")],
                    ignore_static=False,
                )
            ),
            """["L['a'].size()[0] == 2*L['b'].size()[0]", "L['a'].stride()[0] == 1", "L['a'].storage_offset() == 0", "L['b'].stride()[0] == 1", "L['b'].storage_offset() == 0", "2 <= L['b'].size()[0]"]""",  # noqa: B950
        )
        self.assertExpectedInline(
            str(
                fx_g.shape_env.produce_guards(
                    fx_placeholder_vals(fx_g),
                    [LocalSource("a"), LocalSource("b")],
                    ignore_static=True,
                )
            ),
            """["L['a'].size()[0] == 2*L['b'].size()[0]", "2 <= L['b'].size()[0]"]""",  # noqa: B950
        )

    def test_guard_upperbound_range_refinement(self):
        def f(a):
            if not (a.shape[0] > 5 and a.shape[0] > 12):
                raise AssertionError("a.shape[0] should be > 12")
            return a.cos()

        tensor = self._make_fx_symbolic(f, torch.randn(15))
        self.assertExpectedInline(
            show_guards(tensor), """13 <= L['a'].size()[0]"""
        )

    def test_guard_lowerbound_range_refinement(self):
        def f(a):
            if not (a.shape[0] < 20 and a.shape[0] < 30):
                raise AssertionError("a.shape[0] should be < 20")
            return a.cos()

        tensor = self._make_fx_symbolic(f, torch.randn(15))
        self.assertExpectedInline(
            show_guards(tensor), """L['a'].size()[0] <= 19"""
        )

    def test_guard_upperbound_range_refinement_multivariate(self):
        def f(a):
            if not (a.shape[0] > 5 and a.shape[0] > 12):
                raise AssertionError("a.shape[0] should be > 12")
            if not (a.shape[1] > 5 and a.shape[1] > a.shape[0]):
                raise AssertionError("a.shape[1] should be > a.shape[0]")
            return a.cos()

        tensor = self._make_fx_symbolic(f, torch.randn((15, 20)))
        self.assertExpectedInline(
            show_guards(tensor),
            """\
L['a'].size()[1] > L['a'].size()[0]
13 <= L['a'].size()[0]
14 <= L['a'].size()[1]""",
        )

    def test_guard_lowerbound_range_refinement_multivariate(self):
        def f(a):
            if not (a.shape[0] < 20 and a.shape[0] < 30):
                raise AssertionError("a.shape[0] should be < 20")
            if not (a.shape[1] < 30 and a.shape[1] < a.shape[0]):
                raise AssertionError("a.shape[1] should be < a.shape[0]")
            return a.cos()

        tensor = self._make_fx_symbolic(f, torch.randn((15, 5)))
        self.assertExpectedInline(
            show_guards(tensor),
            """\
L['a'].size()[1] < L['a'].size()[0]
3 <= L['a'].size()[0] and L['a'].size()[0] <= 19
L['a'].size()[1] <= 18""",
        )

    def test_sym_storage_offset(self):
        def f(x, y):
            return x + y

        inp = (torch.randn(8)[3:], torch.randn(5))
        fx_g = self._make_fx_symbolic(f, *inp)
        inp = (torch.randn(8)[3:], torch.randn(5))
        self.assertEqual(fx_g(*inp), f(*inp))

    def test_guards_equal(self):
        def f(a, b):
            return a * b

        fx_g = _cpp_fake_trace(f, (5, 6), (5, 6))
        self._assert_no_guards(fx_g, 2)

        fx_g = _cpp_fake_trace(f, (5, 6, 7), (5, 6, 7))
        self._assert_no_guards(fx_g, 3)

        fx_g = _cpp_fake_trace(f, (5, 1), (1, 6))
        self._assert_no_guards(fx_g, 2)

        def f(a, b, c, d):
            a = a + b
            cat = torch.cat([c, d])
            return a + cat

        fx_g = _cpp_fake_trace(f, 7, 7, 4, 3)
        self._assert_no_guards(fx_g, 2)

        def f(a, b, c, d, e):
            vals = [a, b, c, d, e]
            x = a
            for idx in range(len(vals) - 1):
                x = torch.cat([x, vals[idx]]) + vals[idx + 1]
            return x

        fx_g = _cpp_fake_trace(f, 2, 4, 8, 16, 32)
        self._assert_no_guards(fx_g, 1)

        def f(a, b):
            a = a.view(b.shape[0])
            return a + b.sum()

        fx_g = _cpp_fake_trace(f, (4, 2), 8)
        self._assert_no_guards(fx_g, 2)

        fx_g = _cpp_fake_trace(f, (4, 2), (8, 5))
        self._assert_no_guards(fx_g, 3)

        fx_g = _cpp_fake_trace(f, (2, 3, 4), 24)
        self._assert_no_guards(fx_g, 3)

    def test_nonidentity_transitive_guards(self):
        def f(a, b, c, d, e):
            vals = [a, b, c, d, e]
            cat_vals = []
            for idx in range(len(vals) - 1):
                cat_vals.append(torch.cat([vals[idx], vals[idx]]))
            final_vals = []
            for a, b in reversed(list(zip(cat_vals, vals[1:]))):
                final_vals.append(a + b)
            return final_vals

        fx_g = _cpp_fake_trace(f, 2, 4, 8, 16, 32)
        self.assertExpectedInline(show_guards(fx_g), """""")

    @torch.fx.experimental._config.patch(translation_validation=True)
    def test_constant_specialization(self):
        def f(t):
            if t.shape[0] != 10:
                raise AssertionError(
                    f"expected t.shape[0] == 10, got {t.shape[0]}"
                )
            return t

        tensor = self._make_fx_symbolic(f, torch.randn(10))
        self.assertExpectedInline(show_guards(tensor), """""")


if __name__ == "__main__":
    run_tests()
