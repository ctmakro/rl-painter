from __future__ import print_function

# Deep Deterministic Policy Gradient Method
# David Silver et al.

# implemented in plain Keras, by Qin Yongliang
# 2017 01 13

# heavily optimized for speed, lots of numpy flowed into tensorflow
# 2017 01 14

# changed to canton library implementation, much simpler code
# 2017 02 18

# tailored for rl-painter
# 2017 10 29

# gym boilerplate
import numpy as np
import gym
from gym import wrappers
from gym.spaces import Discrete, Box

from math import *
import random
import time

from rpm import rpm # replay memory implementation

import tensorflow as tf
import canton as ct
from canton import *

from plotter import interprocess_plotter as plotter

class nnagent(object):
    def __init__(self,
    action_space,
    discount_factor=.99, # gamma
    ):
        self.rpm = rpm(1000000) # 1M history
        self.plotter = plotter(num_lines=1)
        self.render = True
        self.training = True

        num_of_actions = 8
        self.outputdims = num_of_actions
        self.discount_factor = discount_factor

        ids,ods = None,num_of_actions

        self.actor = self.create_actor_network(ids,ods)
        self.critic = self.create_critic_network(ids,ods)
        self.actor_target = self.create_actor_network(ids,ods)
        self.critic_target = self.create_critic_network(ids,ods)

        self.feed,self.joint_inference,sync_target = self.train_step_gen()

        sess = ct.get_session()
        sess.run(tf.global_variables_initializer())

        sync_target()

    def create_feature_network(self):
        stacking = Lambda(lambda i:tf.concat([i[0],i[1]],axis=3))
        rect = Act('selu')
        c = Can()
        c.add(stacking)
        c.add(Lambda(lambda i:i/255.-0.5)) # normalize the images
        def ca(i,o,k,s):
            c.add(Conv2D(i,o,k=k,std=s,stddev=1))
            c.add(rect)

        ca(6,16,3,1)
        ca(16,32,3,2)
        ca(32,32,3,2)
        ca(32,32,3,2)
        ca(32,64,3,2)

        c.add(Lambda(lambda i:tf.reduce_mean(i,axis=[1,2])))
        c.chain()
        return c

    # a = actor(s) : predict actions given state
    def create_actor_network(self,inputdims,outputdims):
        c = Can()
        rect = Act('selu')
        c.add(self.create_feature_network())
        c.add(Dense(64,64,stddev=1))
        c.add(rect)
        c.add(Dense(64,outputdims,stddev=1))
        c.add(Act('sigmoid'))
        c.chain()
        return c

    # q = critic(s,a) : predict q given state and action
    def create_critic_network(self,inputdims,actiondims):
        c = Can()
        rect = Act('selu')
        fn = c.add(self.create_feature_network())
        concat = Lambda(lambda x:tf.concat([x[0],x[1]],axis=1))
        # concat state and action

        den1 = c.add(Dense(64+actiondims,64,stddev=1))
        den2 = c.add(Dense(64, 1, stddev=1))

        def call(i):
            state = i[0]
            action = i[1]

            feat = fn(state)

            concated = concat([feat,action])
            h1 = rect(den1(concated))
            q = den2(h1)
            return q

        c.set_function(call)
        return c

    def train_step_gen(self):
        s1 = [ph([None,None,3]),ph([None,None,3])] # target, canvas
        a1 = ph([self.outputdims])
        r1 = ph([1])
        isdone = ph([1])
        s2 = [ph([None,None,3]),ph([None,None,3])]

        # 1. update the critic
        a2 = self.actor_target(s2)
        q2 = self.critic_target([s2,a2])
        q1_target = r1 + (1-isdone) * self.discount_factor * q2
        q1_predict = self.critic([s1,a1])
        critic_loss = tf.reduce_mean((q1_target - q1_predict)**2)
        # produce better prediction

        # 2. update the actor
        a1_predict = self.actor(s1)
        q1_predict = self.critic([s1,a1_predict])
        actor_loss = tf.reduce_mean(- q1_predict)
        # maximize q1_predict -> better actor

        # 3. shift the weights (aka target network)
        tau = tf.Variable(1e-3) # original paper: 1e-3. need more stabilization
        aw = self.actor.get_weights()
        atw = self.actor_target.get_weights()
        cw = self.critic.get_weights()
        ctw = self.critic_target.get_weights()

        one_m_tau = 1-tau

        shift1 = [tf.assign(atw[i], aw[i]*tau + atw[i]*(one_m_tau))
            for i,_ in enumerate(aw)]
        shift2 = [tf.assign(ctw[i], cw[i]*tau + ctw[i]*(one_m_tau))
            for i,_ in enumerate(cw)]

        # 4. inference
        a_infer = self.actor(s1)
        q_infer = self.critic([s1,a_infer])

        # 5. L2 weight decay on critic
        decay_c = tf.reduce_sum([tf.reduce_sum(w**2) for w in cw])* 0.0001
        # decay_a = tf.reduce_sum([tf.reduce_sum(w**2) for w in aw])* 0.0001

        # optimizer on
        # actor is harder to stabilize...
        opt_actor = tf.train.AdamOptimizer(1e-4)
        opt_critic = tf.train.AdamOptimizer(3e-4)
        # opt_actor = tf.train.MomentumOptimizer(1e-1,momentum=0.9)
        cstep = opt_critic.minimize(critic_loss, var_list=cw)
        astep = opt_actor.minimize(actor_loss, var_list=aw)

        self.feedcounter=0
        def feed(memory):
            [s1d,a1d,r1d,isdoned,s2d] = memory # d suffix means data
            sess = ct.get_session()
            res = sess.run([critic_loss,actor_loss,
                cstep,astep,shift1,shift2],
                feed_dict={
                s1[0]:s1d[0],s2[0]:s2d[0],
                s1[1]:s1d[1],s2[1]:s2d[1],
                a1:a1d,r1:r1d,isdone:isdoned,tau:1e-3
                })

            #debug purposes
            self.feedcounter+=1
            # if self.feedcounter%10==0:
            if True:
                print(' '*30, 'closs: {:6.4f} aloss: {:6.4f}'.format(
                    res[0],res[1]),end='\r')

            # return res[0],res[1] # closs, aloss

        def joint_inference(state):
            sess = ct.get_session()
            res = sess.run([a_infer,q_infer],feed_dict={
                s1[k]:state[k] for k in [0,1]
            })
            return res

        def sync_target():
            sess = ct.get_session()
            sess.run([shift1,shift2],feed_dict={tau:1.})

        return feed,joint_inference,sync_target

    def train(self):
        memory = self.rpm
        batch_size = 32

        if memory.size() > 1000:
            #if enough samples in memory
            # sample randomly a minibatch from memory
            [s1,a1,r1,isdone,s2] = memory.sample_batch(batch_size)
            # print(s1.shape,a1.shape,r1.shape,isdone.shape,s2.shape)

            self.feed([s1,a1,r1,isdone,s2])

    def feed_one(self,tup):
        self.rpm.add(tup)

    # gymnastics
    def play(self,env,max_steps=-1,realtime=False,noise_level=0.): # play 1 episode
        timer = time.time()
        max_steps = max_steps if max_steps > 0 else 50000
        steps = 0
        total_reward = 0

        observation = env.reset()

        while True and steps <= max_steps:
            steps +=1

            action = self.act(observation) # a1

            # exploration
            exploration_noise = np.random.normal(size=(self.outputdims,))*noise_level
            action += exploration_noise
            action = np.clip(action,0.,1.)

            # o2, r1,
            new_observation, reward, done, _info = env.step(action)

            # d1
            isdone = 1 if done else 0
            total_reward += reward

            # feed into replay memory
            if self.training == True:
                self.feed_one(
                (observation,action,reward,isdone,new_observation)) # s1,a1,r1,isdone,s2

            observation = new_observation
            if self.render==True and (steps%30==0 or realtime==True):
                env.render()
            if done :
                break

            if self.training == True:
                self.train()

        # print('episode done in',steps,'steps',time.time()-timer,'second total reward',total_reward)
        totaltime = time.time()-timer
        print('episode done in {} steps in {:.2f} sec, {:.4f} sec/step, got reward :{:.2f}'.format(
        steps,totaltime,totaltime/steps,total_reward
        ))

        self.plotter.pushys([total_reward])
        return

    # one step of action, given observation
    def act(self,obs):
        actor,critic = self.actor,self.critic
        obs = [np.reshape(ob,(1,)+ob.shape) for ob in obs]

        # actions = actor.infer(obs)
        # q = critic.infer([obs,actions])[0]
        [actions,q] = self.joint_inference(obs)
        actions,q = actions[0],q[0]

        return actions

    def save_weights(self):
        networks = ['actor','critic','actor_target','critic_target']
        for name in networks:
            network = getattr(self,name)
            network.save_weights('ddpg_'+name+'.npz')

    def load_weights(self):
        networks = ['actor','critic','actor_target','critic_target']
        for name in networks:
            network = getattr(self,name)
            network.load_weights('ddpg_'+name+'.npz')

if __name__=='__main__':
    from env import CanvasEnv
    e = CanvasEnv()

    agent = nnagent(
        e.action_space,
        discount_factor=.96,
    )

    noise_level = 0.5
    def r(ep):
        global noise_level,e
        # agent.render = True
        for i in range(ep):
            noise_level *= .99
            noise_level = max(1e-3, noise_level)
            print('ep',i,'/',ep,'noise_level',noise_level)
            agent.play(e,realtime=True,max_steps=50,noise_level=noise_level)

    def test():
        e = p.env
        agent.render = True
        agent.play(e,realtime=True,max_steps=-1,noise_level=1e-11)
