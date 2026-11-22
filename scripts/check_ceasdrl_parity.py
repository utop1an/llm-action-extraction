"""Check the GPU port against stored execution results from the original Keras code."""
import argparse
import json
from pathlib import Path

import numpy as np
import torch
from ceasdrl_torch import CnnDqn, binary_rewards, dqn_loss


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--fixtures', type=Path, required=True)
    args = parser.parse_args()
    torch.set_num_threads(4)
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    root = Path(__file__).resolve().parents[1]
    report = {'device': torch.cuda.get_device_name(0), 'modes': {}, 'passed': False}
    for mode in ['act', 'arg']:
        f = np.load(args.fixtures / (mode + '.npz'))
        model = CnnDqn(mode).cuda()
        model.load_keras_h5(root / 'data/ceasdrl/weights' / ('cooking_' + mode + '_elmo.h5'))
        t = {k: torch.from_numpy(f[k]).cuda() for k in ['words', 'before', 'after', 'distance', 'actions', 'rewards', 'terminal']}
        q = model(t['words'], t['before'], t['distance'])
        loss = dqn_loss(model, model, t['words'], t['before'], t['after'], t['distance'], t['actions'], t['rewards'], t['terminal'])
        loss.backward()
        gradients = []
        for conv in model.convs:
            gradients.extend([conv.weight.grad.detach().cpu().numpy().transpose(2, 1, 0)[:, :, None, :], conv.bias.grad.detach().cpu().numpy()])
        for layer in [model.dense, model.output]:
            gradients.extend([layer.weight.grad.detach().cpu().numpy().T, layer.bias.grad.detach().cpu().numpy()])
        errors = [float(np.linalg.norm(g - f['gradient_' + str(i)]) / max(np.linalg.norm(f['gradient_' + str(i)]), 1e-8))
                  for i, g in enumerate(gradients)]
        q_error = float(np.max(np.abs(q.detach().cpu().numpy() - f['q'])))
        loss_error = abs(loss.item() - float(f['loss'])) / max(abs(float(f['loss'])), 1e-8)
        assert np.allclose(q.detach().cpu().numpy(), f['q'], rtol=1e-4, atol=1e-3), (mode, q_error)
        assert loss_error < 1e-4 and max(errors) < 1e-4, (mode, loss_error, errors)
        report['modes'][mode] = {'max_absolute_q_error': q_error, 'relative_loss_error': loss_error,
                                 'max_relative_gradient_l2_error': max(errors), 'passed': True}
    cases = json.loads((args.fixtures / 'reward_cases.json').read_text())
    error = 0.
    for n, action, positive, count, terminal, expected in cases:
        got = binary_rewards(np.array([action]), np.array([positive]), np.array([count]), n, np.array([terminal]))[0]
        error = max(error, abs(float(got) - expected))
    assert error < 1e-4, error
    report['reward_cases'], report['max_reward_error'], report['passed'] = len(cases), error, True
    (args.fixtures / 'parity_report.json').write_text(json.dumps(report, indent=2), encoding='utf-8')
    print(json.dumps(report, indent=2))


if __name__ == '__main__':
    main()
