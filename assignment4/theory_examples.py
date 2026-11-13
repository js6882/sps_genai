"""Run the Assignment 4 calculations: python theory_examples.py."""
import math
import torch


def sinusoidal_embedding(t, dimension=8, max_period=10000):
    half = dimension // 2
    frequencies = torch.exp(-math.log(max_period) *
                            torch.arange(half, dtype=torch.float64) / max(half - 1, 1))
    angles = torch.tensor(t, dtype=torch.float64) * frequencies
    return torch.cat([angles.sin(), angles.cos()])


def main():
    print('Q2:', sinusoidal_embedding(1).tolist())
    x = torch.tensor([2.0], requires_grad=True)
    (x**2 + 3*x).backward()
    assert x.grad.item() == 7
    print('Q6a: x.grad =', x.grad)
    x = torch.tensor([2.0], requires_grad=False)
    try:
        (x**2 + 3*x).backward()
    except RuntimeError as error:
        print('Q6b:', error)
    else:
        raise AssertionError('Expected a graph-free backward error')
    print('Q6c / Q7c: default requires_grad =', torch.tensor([2.0]).requires_grad)
    x = torch.tensor([2.0], requires_grad=True)
    w = torch.tensor([1.0, 3.0])
    (w[0]*x**2 + w[1]*x).backward()
    assert w.grad is None
    print('Q7a: w.grad =', w.grad)
    x = torch.tensor([2.0])
    w = torch.tensor([1.0, 3.0], requires_grad=True)
    (w[0]*x**2 + w[1]*x).backward()
    assert torch.equal(w.grad, torch.tensor([4.0, 2.0]))
    print('Q7b: w.grad =', w.grad)
    x = torch.tensor([1.0], requires_grad=True)
    y = x * 3
    z = y.clone()
    w = z * 2
    w.backward()
    assert x.grad.item() == 6
    print('Q8 fixed: x.grad =', x.grad)
    x = torch.tensor([1.0], requires_grad=True)
    (x*2).backward()
    print('Q9 first: x.grad =', x.grad)
    (x*3).backward()
    assert x.grad.item() == 5
    print('Q9 second: x.grad =', x.grad)
    x.grad = None
    (x*3).backward()
    assert x.grad.item() == 3
    print('Q9 after reset: x.grad =', x.grad)


if __name__ == '__main__':
    main()
