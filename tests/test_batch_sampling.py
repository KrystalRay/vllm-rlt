import torch

from vllm_rlt import SamplingParams
from vllm_rlt.worker.sampler import Sampler


def test_batched_greedy_preserves_ties_and_ignores_padding():
    logits = torch.tensor([[1.0, 3.0, 3.0], [4.0, 1.0, 0.0], [9.0, 0.0, 1.0]])
    params = (SamplingParams(), SamplingParams())
    assert torch.equal(Sampler().sample_batch(logits, params, (None, None)), torch.tensor([1, 0]))


def test_mixed_batch_preserves_request_local_rng():
    logits = torch.tensor([[1.0, 2.0, 3.0], [3.0, 1.0, 0.0]])
    params = (SamplingParams(), SamplingParams(temperature=0.8))
    generators = [None, torch.Generator().manual_seed(17)]
    actual = Sampler().sample_batch(logits, params, generators)
    control = torch.Generator().manual_seed(17)
    expected = torch.stack((logits[0].argmax(), Sampler().sample(logits[1], params[1], control)))
    assert torch.equal(actual, expected)
    assert torch.equal(generators[1].get_state(), control.get_state())
