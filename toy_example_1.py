import numpy as np
import pandas as pd
from tqdm import tqdm
from matplotlib import pyplot as plt
import json
import pickle as pkl
import fctns
import argparse
import os
import warnings
with warnings.catch_warnings():  
    warnings.filterwarnings("ignore",category=FutureWarning)
    import tensorflow as tf

os.environ['TF_CPP_MIN_LOG_LEVEL']='3'

parser = argparse.ArgumentParser(description='toy example 1')
parser.add_argument('--N', type=int, default=5, help='number of trials')
parser.add_argument('--dim', type=int, default=5, help='size of categorical variables')
parser.add_argument('--gpu_num', type=int, default=0, help='gpu index')
parser.add_argument('--steps', type=int, default=50000, help='number of training steps')
parser.add_argument('--save_dir', type=str, default='.', help='where to save the saveouts')
args = parser.parse_args()
os.environ['CUDA_VISIBLE_DEVICES'] = str(args.gpu_num)

def run_experiment(draw_fn, **kwargs):
    """
    runs toy experiment 1
    """
    # setup
    logits = tf.Variable(np.random.uniform(size=(args.N,args.dim)), dtype=tf.float32)
    probits = tf.nn.softmax(logits)
    draw = draw_fn(logits, **kwargs)
    gt = np.random.uniform(size=(args.N,args.dim))
    gt = gt / gt.sum(axis=-1, keepdims=True)
    true = tf.constant(gt, dtype=tf.float32)

    # KL div loss based on sampling
    loss = tf.reduce_mean(-true*(tf.log(draw+1e-8) - tf.log(true+1e-8)))

    # KL loss based on true probits
    true_loss = tf.reduce_mean(-true*(tf.log(probits+1e-8) - tf.log(true+1e-8)))

    # optimization step
    grad_log = tf.gradients(loss, logits)[0]
    train_op = logits.assign(logits - .01*grad_log)

    # training loop
    losses = []
    with tf.Session() as sess:
        sess.run(tf.global_variables_initializer())
        for i in range(args.steps):
            _, l, p = sess.run([train_op, true_loss, probits])
            if i == 0:
                p0 = p
            losses.append(l)
    return losses

def main():
    """
    main
    """
    gumbel_temps = [.1,1,10]
    il_consts = [.1,.01,.001]
    gumbel_losses = {}
    il_losses = {}
    
    for temp in gumbel_temps:
        gumbel_losses[str(temp)] = run_experiment(fctns.gumbel_softmax, temp=temp)
    
    for const in il_consts:
        il_losses[str(const)] = run_experiment(fctns.il_draw, const=const)
        
    losses = {'gumbel':gumbel_losses, 'il':il_losses}
    pkl.dump(losses, open(os.path.join(args.save_dir, 'toyexp_1_losses.pkl'), 'wb')) 
    
    fig = plt.figure(figsize=(15,10))
    for i,(k, _losses) in enumerate(losses['il'].items()):
        ax = fig.add_subplot(2,3,i+1)
        ax.set_title(f'intermediate_step={k}')
        ax.plot(_losses)
    for i,(k, _losses) in enumerate(losses['gumbel'].items()):
        ax = fig.add_subplot(2,3,i+3+1)
        ax.set_title(f'temp={k}')
        ax.plot(_losses)
    
    plt.savefig(os.path.join(args.save_dir, 'toyexp_1_losses.png'))
    
if __name__ == '__main__':
    main()