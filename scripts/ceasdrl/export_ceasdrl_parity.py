"""Run in py36: export upstream Keras forward, DQN-loss and gradient fixtures."""
import argparse
import contextlib
import io
import json
import os
import sys
from pathlib import Path

from ceasdrl_local import ROOT, VENDOR, args_for


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)
    os.environ['CUDA_VISIBLE_DEVICES'] = '-1'
    os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
    sys.path.insert(0, str(VENDOR))
    import numpy as np
    import tensorflow as tf
    from keras import backend as K
    from keras.backend.tensorflow_backend import set_session
    from EADQN import DeepQLearner
    from cEnvironment import Environment
    set_session(tf.Session(config=tf.ConfigProto(intra_op_parallelism_threads=4, inter_op_parallelism_threads=1)))
    rng = np.random.RandomState(135)
    for mode, length in [('act', 600), ('arg', 100)]:
        with contextlib.redirect_stdout(io.StringIO()):
            net = DeepQLearner(args_for(mode, length, 868), mode, 'channels_last')
            net.load_weights(str(ROOT / 'data/ceasdrl/weights' / ('cooking_{}_elmo.h5'.format(mode))))
        words = rng.normal(0, .2, (4, length, 868)).astype(np.float32)
        words[:, 35:] = 0
        before = np.zeros((4, length), dtype=np.float32)
        before[:, :12] = rng.randint(1, 3, (4, 12))
        actions = np.array([0, 1, 1, 0], dtype=np.int64)
        after = before.copy()
        after[:, 12] = actions + 1
        distance = np.zeros((4, length), dtype=np.float32)
        distance[:, :35] = np.abs(np.arange(35) - 10)
        components = [words]
        if mode == 'arg':
            components.append(np.repeat(distance[:, :, None], 868, axis=2))
        pre = np.concatenate(components + [np.repeat(before[:, :, None], 868, axis=2)], axis=2)[:, :, :, None]
        post = np.concatenate(components + [np.repeat(after[:, :, None], 868, axis=2)], axis=2)[:, :, :, None]
        q = net.model.predict_on_batch(pre)
        postq = net.target_model.predict_on_batch(post)
        rewards = np.array([50., 100., -100., -50.], dtype=np.float32)
        terminal = np.array([False, False, True, True])
        targets = q.copy()
        targets[np.arange(4), actions] = rewards + .9 * postq.max(axis=1) * (~terminal)
        expected = K.placeholder(shape=(4, 2))
        loss = K.mean(K.square(net.model.output - expected))
        gradients = K.gradients(loss, net.model.trainable_weights)
        values = K.function([net.model.input, expected], [loss] + gradients)([pre, targets])
        fixture = dict(words=words, before=before, after=after, distance=distance, actions=actions,
                       rewards=rewards, terminal=terminal, q=q, targets=targets, loss=values[0])
        for i, value in enumerate(values[1:]):
            fixture['gradient_{}'.format(i)] = value
        np.savez_compressed(str(args.output / (mode + '.npz')), **fixture)
        print('Exported {} Q/loss/12 parameter gradients'.format(mode), flush=True)
    # Compare binary reward branches, both shaping regions and terminal bonus.
    cases = []
    for length in [100, 600]:
        for i in [0, 5, 20]:
            for positives_before in [0, i]:
                for label in [1, 2]:
                    for action in [0, 1]:
                        env = Environment.__new__(Environment)
                        env.agent_mode, env.num_words = 'act', length
                        env.ra, env.reward_base, env.action_rate, env.use_act_rate = [1, 2, 3], 50., .1, 1
                        env.state = np.zeros((length, 1), dtype=np.float32)
                        env.state[:i] = 1
                        env.state[:positives_before] = 2
                        env.text_vec = np.ones((length, 1), dtype=np.float32)
                        env.text_vec[i] = label
                        env.current_text = {'tokens': ['x'] * (i + 1)}
                        env.terminal_flag = False
                        reward = env.act(action, i) + 2
                        cases.append([length, action, label == 2, positives_before + action, True, reward])
    (args.output / 'reward_cases.json').write_text(json.dumps(cases), encoding='utf-8')


if __name__ == '__main__':
    main()
