"""Legacy py36: validate newly trained weights in the actual upstream Keras CNN."""
import argparse
import contextlib
import io
import json
import os
import sys
from pathlib import Path

from ceasdrl_local import VENDOR, args_for


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--run', type=Path, required=True)
    args = p.parse_args()
    os.environ['CUDA_VISIBLE_DEVICES'] = '-1'
    os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
    sys.path.insert(0, str(VENDOR))
    import numpy as np
    import tensorflow as tf
    from keras.backend.tensorflow_backend import set_session
    from EADQN import DeepQLearner
    config = json.loads((args.run / 'config.json').read_text())
    mode, n = config['mode'], config['num_words']
    set_session(tf.Session(config=tf.ConfigProto(intra_op_parallelism_threads=4, inter_op_parallelism_threads=1)))
    network_args = args_for(mode, n, 868)
    if config.get('span_labels'):
        network_args.num_actions = 3
    with contextlib.redirect_stdout(io.StringIO()):
        net = DeepQLearner(network_args, mode, 'channels_last')
        net.load_weights(str(args.run / 'best.keras.weights.h5'))
    fixture = np.load(str(args.run / 'trained_keras_probe.npz'))
    components = [fixture['words']]
    if mode == 'arg':
        components.append(np.repeat(fixture['distance'][:, :, None], 868, axis=2))
    components.append(np.repeat(fixture['tags'][:, :, None], 868, axis=2))
    q = net.model.predict_on_batch(np.concatenate(components, axis=2)[:, :, :, None])
    error = float(np.max(np.abs(q - fixture['q'])))
    passed = bool(np.allclose(q, fixture['q'], rtol=1e-4, atol=1e-3))
    report = {'passed': passed, 'real_test_contexts': 4, 'mode': mode,
              'num_q_outputs': network_args.num_actions,
              'max_absolute_q_error': error, 'weights': 'best.keras.weights.h5',
              'validation': 'Newly trained weights loaded by original Keras DeepQLearner; outputs compared with GPU checkpoint.',
              'adaptation': 'num_actions=3 for BIO, not the original two-class task' if config.get('span_labels') else None}
    (args.run / 'trained_keras_compatibility.json').write_text(json.dumps(report, indent=2), encoding='utf-8')
    print(json.dumps(report))
    if not passed:
        raise RuntimeError('Trained Keras compatibility failed')


if __name__ == '__main__':
    main()
