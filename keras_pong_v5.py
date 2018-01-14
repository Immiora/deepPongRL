# https://gist.github.com/karpathy/a4166c7fe253700972fcbc77e4ea32c5

from __future__ import print_function
import os
import gym
import sys
import argparse
import sys, glob
import numpy as np
import pickle
from keras import backend as K
import matplotlib.pyplot as plt
from keras.models import Sequential
from keras.callbacks import ModelCheckpoint
from keras.layers import Input, Dense, Reshape
from keras.layers.wrappers import TimeDistributed
from keras.optimizers import Adam, Adamax, RMSprop
from keras.layers.advanced_activations import PReLU
from keras.layers.normalization import BatchNormalization
from keras.layers.core import Activation, Dropout, Flatten
from keras.layers.convolutional import UpSampling2D, Convolution2D
from keras.models import model_from_json


def pong_preprocess_screen(I):
  I = I[35:195]
  I = I[::2, ::2, 0]
  I[I == 144] = 0
  I[I == 109] = 0
  I[I != 0] = 1
  return I.astype(np.float).ravel()

def discount_rewards(r):
  gamma = 0.99
  discounted_r = np.zeros_like(r)
  running_add = 0
  for t in reversed(range(0, r.size)):
    if r[t] != 0: running_add = 0
    running_add = running_add * gamma + r[t]
    discounted_r[t] = running_add
  return discounted_r

def learning_model(args, learning_rate, input_dim=80*80, number_of_inputs = 2):
  if args.resume == False:
    model = Sequential()
    if args.model == 'mlp':
      model.add(Reshape((1,80,80), input_shape=(input_dim,)))
      model.add(Flatten())
      model.add(Dense(args.ndense, activation = 'relu'))
      for l in range(args.nlayers):
       model.add(Dense(args.ndense))
      model.add(Dense(number_of_inputs, activation='softmax'))
      opt = RMSprop(lr=learning_rate)

    elif args.model == 'cnn':
      model.add(Reshape((1, 80,80), input_shape=(input_dim,)))
      model.add(Convolution2D(args.nfilters1, args.size_filters1, 
                  args.size_filters1, subsample=(4, 4), border_mode='same', 
                  activation='relu', init='he_uniform'))

      if args.nlayers > 1:
                model.add(Convolution2D(args.nfilters2, args.size_filters2, 
                  args.size_filters2, subsample=(4, 4), border_mode='same', 
                  activation='relu', init='he_uniform'))
      model.add(Flatten())
      model.add(Dense(args.ndense))

      model.add(Activation('relu'))
      model.add(Dense(number_of_inputs, activation='softmax'))
      opt = Adam(lr=learning_rate)

    else:
      raise Exception("Unknown model specification")

    model.compile(loss='categorical_crossentropy', optimizer=opt)
    
  else:
    json_file = open(args.output + 'model.json', 'r')
    loaded_model_json = json_file.read()
    json_file.close()
    model = model_from_json(loaded_model_json)
    model.compile(loss='categorical_crossentropy', optimizer=Adam(lr=learning_rate))
    model.load_weights(args.output + 'pong_model_checkpoint.h5')
  return model


def main():

  # Parameters

  parser = argparse.ArgumentParser(description='Keras Pong RL')
  parser.add_argument('--batchsize', '-b', type=int, default=10,
                      help='Number of episodes to run before update')
  parser.add_argument('--nepoch', '-e', type=int, default=20000,
                      help='Number of episodes to run')
  parser.add_argument('--model', '-m', default='mlp',
                      help='Model to run: mlp or cnn')
  parser.add_argument('--nlayers', '-l', type=int, default=1,
                      help='Number of hidden layers')
  parser.add_argument('--ndense', '-d', type=int, default=50,
                      help='Number of units in dense layers')
  parser.add_argument('--nfilters1', '-k', type = int, default= 48,
                      help='Number of conv filters')
  parser.add_argument('--size_filters1', '-s', type = int, default = 9,
                      help='Size of conv filters')
  parser.add_argument('--nfilters2', '-k2', type = int, default= 48,
                      help='Number of conv filters')
  parser.add_argument('--size_filters2', '-s2', type = int, default=6,
                      help='Size of conv filters')
  parser.add_argument('--regularizer', '-r', default=0.,
                      help='Amount of dropout')
  parser.add_argument('--resume', '-c', default=False,
                      help='Continue training from previous state')
  parser.add_argument('--output', '-o', default='./',
                      help='Output dir')
  args = parser.parse_args()

  # args.size_filters1 = (args.size_filters1, args.size_filters1)
  # args.size_filters2 = (args.size_filters2, args.size_filters2)

  input_dim = 80 * 80
  render = False
  learning_rate = 0.001

  if not os.path.exists(args.output):
    os.makedirs(args.output)

  #Initialize
  env = gym.make("Pong-v0")
  number_of_inputs = 2 # env.action_space.n #This is incorrect for Pong (?)

  observation = env.reset()
  prev_x = None
  xs, dlogps, drs, probs = [],[],[],[]
  running_reward = None
  reward_sum = 0
  episode_number = 0
  train_X = []
  train_y = []


  # Define the main model (WIP)
  print(args.model)
  model = learning_model(args, learning_rate)

  # Serialize model to JSON
  model_json = model.to_json()
  with open(args.output + "model.json", "w") as json_file:
      json_file.write(model_json)
  reward_hist = np.zeros(args.nepoch) * np.nan


  # Begin training
  iter = 0
  while episode_number < args.nepoch:
    if render:
      env.render()

    # Preprocess, consider the frame difference as features
    cur_x = pong_preprocess_screen(observation)
    x = cur_x - prev_x if prev_x is not None else np.zeros(input_dim)
    prev_x = cur_x

    # Predict probabilities from the Keras model
    aprob = ((model.predict(x.reshape([1,x.shape[0]]), batch_size=1).flatten()))
    xs.append(x)
    probs.append((model.predict(x.reshape([1,x.shape[0]]), batch_size=1).flatten()))
    aprob = aprob/np.sum(aprob)
    action = np.random.choice(number_of_inputs, 1, p=aprob)[0]
    y = np.zeros([number_of_inputs])
    y[action] = 1
    dlogps.append(np.array(y).astype('float32') - aprob)
    observation, reward, done, info = env.step(action + 2)
    reward_sum += reward
    drs.append(reward)

    if done:
      episode_number += 1
      epx = np.vstack(xs)
      epdlogp = np.vstack(dlogps)
      epr = np.vstack(drs)
      discounted_epr = discount_rewards(epr)
      discounted_epr -= np.mean(discounted_epr)
      discounted_epr /= np.std(discounted_epr)
      epdlogp *= discounted_epr

      # Slowly prepare the training batch
      train_X.append(xs)
      train_y.append(epdlogp)
      xs,dlogps,drs = [],[],[]

      # Periodically update the model
      if episode_number % args.batchsize == 0:
        y_train = probs + learning_rate * np.squeeze(np.vstack(train_y)) #Hacky WIP
        print('Training Snapshot:')
        print(y_train)
        model.train_on_batch(np.squeeze(np.vstack(train_X)), y_train)

        # Clear the batch
        train_X = []
        train_y = []
        probs = []
        os.remove(args.output + 'pong_model_checkpoint.h5') if os.path.exists(args.output + 'pong_model_checkpoint.h5') else None
        model.save_weights(args.output + 'pong_model_checkpoint.h5')
        reward_hist[iter] = reward_sum
        iter += 1
        np.save(args.output + 'rewardsum_history.npy', reward_hist)

        # Reset the current environment nad print the current results
      running_reward = reward_sum if running_reward is None else running_reward * 0.99 + reward_sum * 0.01
      print('Environment reset imminent. Total Episode Reward: %.2f. Running Mean: %.2f' % (reward_sum, running_reward))
      reward_sum = 0
      observation = env.reset()
      prev_x = None

    if reward != 0:
      print('Episode %d Result: ' % episode_number + 'Defeat!' if reward == -1 else 'VICTORY!')


if __name__ == '__main__':
    main()