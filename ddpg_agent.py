#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Sun Mar 26 11:04:47 2017

@author: daniel
"""

from __future__ import print_function

import gym
import numpy as np
import tensorflow as tf
import time
from collections import deque  
import datetime
import cv2
import os
import sys
from gym import wrappers
import argparse
from memory import ReplayMemory as RPM
from tensorflow.python.client import timeline

import actor
import critic
import ou_noise


class DDPGAgent(object):
    
    def __init__(self,sess,env,params):
        self.params=params
        self.xpsize=params['replaymemory']
        self.cnt=0
        self.env=env
        self.sess=sess
        self.current_loss=0
        
#        self.run_options = tf.RunOptions(trace_level=tf.RunOptions.FULL_TRACE)
#        self.run_metadata = tf.RunMetadata()
        
        self.last_reward=tf.Variable(0,name="cum_reward",dtype=tf.float32,trainable=False)
        self.last_q=tf.Variable(0,name="cum_q",dtype=tf.float32,trainable=False)
        self.last_rate=tf.Variable(0,name="rate",dtype=tf.float32,trainable=False)
        self.last_steps=tf.Variable(0,name="episode_steps",dtype=tf.float32,trainable=False)
        self.epoche_reward=tf.Variable(0,name="epoche_reward",dtype=tf.float32,trainable=False)
        self.epoche_value=tf.Variable(0,name="epoche_value",dtype=tf.float32,trainable=False)
        self.epoche_maxreward=tf.Variable(0,name="epoche_max_reward",dtype=tf.float32,trainable=False)
        
        self.eps=params['initexploration']
        
        self.noise=ou_noise.OUNoise(params["actionsize"])
        
        self.ac_predict=actor.ActorNet(sess,env,"actor_predict",params)
        self.ac_target=actor.ActorNet(sess,env,"actor_target",params,train=False)
        self.ac_target.setWeightUpdate(self.ac_predict.params_list)
#        self.last_action=tf.Variable(params["actionsize"]*[0],name="last_action",dtype=tf.float32,trainable=False)
#        self.last_action.assign(self.ac_predict.scaled_out)
        
        self.cr_predict=critic.CriticNet(sess,"critic_predict",params,wd=self.params["weight_decay"])
        self.cr_target=critic.CriticNet(sess,"critic_target",params,train=False)
        self.cr_target.setWeightUpdate(self.cr_predict.params_list)
        
        self.cr_target.updateInitWeights(self.cr_predict.params_list)
        self.ac_target.updateInitWeights(self.ac_predict.params_list)
        
        self.initTraining()
        self.initSummaries()
        
        self.rpm=RPM(params['replaymemory'],frame_shape=params["obssize"],dtype=params["frame_dtype"])
        
#        os.mkdir(self.params['traindir'])
        subdir=datetime.datetime.now().strftime('%d%m%y_%H%M%S')
        self.traindir=os.path.join(params['traindir'], "run_%s"%subdir)
        os.mkdir(self.traindir)
        self.picdir=os.path.join(self.traindir,"pics")
        os.mkdir(self.picdir)
        checkpoint_dir=os.path.join(self.traindir,self.params['checkpoint_dir'])
        os.mkdir(checkpoint_dir)
        
        self.saver = tf.train.Saver()
        
        if params["latest_run"]:
            self.latest_traindir=os.path.join(params['traindir'], "run_%s"%params["latest_run"])
            latest_checkpoint = tf.train.latest_checkpoint(os.path.join(self.latest_traindir,self.params['checkpoint_dir']))
            if latest_checkpoint:
                print("Loading model checkpoint {}...\n".format(latest_checkpoint))
                self.saver.restore(sess, latest_checkpoint)
        
        self.merged = tf.summary.merge_all()
        self.train_writer = tf.summary.FileWriter(self.traindir,sess.graph)
                
        init = tf.global_variables_initializer()
        
        sess.run(init)
#        self.cr_target.updateInitWeights(self.cr_predict.params_list)
#        self.ac_target.updateInitWeights(self.ac_predict.params_list)
        self.sess.run(self.cr_target.initholder)
        self.sess.run(self.ac_target.initholder)
#        sess.graph.finalize()
    
    def __del__(self):
        self.train_writer.close()
        
    def initTraining(self):
        self.global_step = tf.Variable(0, trainable=False)
        
        self.optimizer_critic = tf.train.AdamOptimizer(self.params['learningrate_critic'])
        
        cr_pred=self.cr_predict.estimateQ()
        cr_target=self.cr_target.estimateTarget()
        
        diff=cr_target-cr_pred
        
        self.loss_critic = tf.reduce_mean(self.td_error(diff))
        tf.add_to_collection('losses', self.loss_critic)
        self.loss_critic=tf.add_n(tf.get_collection('losses'))
        
        self.train_critic = self.optimizer_critic.minimize(self.loss_critic,global_step=self.global_step)
        
        
        self.optimizer_actor = tf.train.AdamOptimizer(self.params['learningrate_actor'])
        
        self.train_actor=self.optimizer_actor.apply_gradients(zip(self.ac_predict.actor_gradients, self.ac_predict.params_list))

        
    def variable_summaries(self,var,name='summaries'):
        with tf.name_scope(name.split("/")[-1].split(":")[0]):
            mean = tf.reduce_mean(var)
            tf.summary.scalar('mean', mean)
            with tf.name_scope('stddev'):
                stddev = tf.sqrt(tf.reduce_mean(tf.square(var - mean)))
            tf.summary.scalar('stddev', stddev)
            tf.summary.scalar('max', tf.reduce_max(var))
            tf.summary.scalar('min', tf.reduce_min(var))
            tf.summary.histogram('histogram', var)
            
            
    def initSummaries(self):
        with tf.name_scope("episode_stats"):
            tf.summary.scalar('cum_reward', self.last_reward)
            tf.summary.scalar('steps', self.last_steps)
            tf.summary.scalar('rate', self.last_rate)
        with tf.name_scope("epoche_stats"):
            tf.summary.scalar('epoche_reward', self.epoche_reward)
            tf.summary.scalar('epoche_maxreward', self.epoche_maxreward)
            tf.summary.scalar('epoche_value', self.epoche_value)
        with tf.name_scope("critic"):
            self.variable_summaries(self.cr_predict.out)
            tf.summary.histogram('histogram',tf.to_float(self.cr_predict.out))
            self.variable_summaries(self.cr_target.out)
            tf.summary.histogram('histogram',tf.to_float(self.cr_target.out))
        for w in self.cr_predict.params_list:
            with tf.name_scope("critic_"+w.name.split("/")[-1].split(":")[0]):
                self.variable_summaries(w)
        for w in self.cr_target.params_list:
            with tf.name_scope("critic_target_"+w.name.split("/")[-1].split(":")[0]):
                self.variable_summaries(w)
        for w in self.ac_predict.params_list:
            with tf.name_scope("actor_"+w.name.split("/")[-1].split(":")[0]):
                self.variable_summaries(w)
        with tf.name_scope("actor"):
            self.variable_summaries(self.ac_predict.scaled_out)
            tf.summary.histogram('histogram',tf.to_float(self.ac_predict.scaled_out))
#            self.variable_summaries(self.ac_predict.actor_gradients)
#            tf.summary.histogram('histogram',tf.to_float(self.ac_predict.actor_gradients))
        with tf.name_scope("loss"):
            tf.summary.scalar('loss_critic',self.loss_critic)
            
        
        
    def saveStats(self,reward,steps=0,rate=0):
        ops=[self.last_reward.assign(reward),
             self.last_steps.assign(steps),
             self.last_rate.assign(rate)]
        
        self.sess.run(ops)
#        reward_file=os.path.join(self.traindir, 'rewards.dat')
#        np.savetxt(reward_file,np.array(data))

    
    def epocheStats(self,reward,q,rmax):
        ops=[self.epoche_value.assign(q),
             self.epoche_reward.assign(reward),
             self.epoche_maxreward.assign(rmax)]
        
        self.sess.run(ops)
        
    def td_error(self,x):
        if self.params["huberloss"]:
            # Huber loss
            try:
                return tf.select(tf.abs(x) < 1.0, 0.5 * tf.square(x), tf.abs(x) - 0.5)
            except:
                return tf.where(tf.abs(x) < 1.0, 0.5 * tf.square(x), tf.abs(x) - 0.5)
        else:
            return tf.square(x)
            
        
    def takeAction(self,state=None,eps_ext=None):
        g=0
        
        if state==None:
            a=self.env.action_space.sample()
        else:
            feed=np.expand_dims(state,axis=0)
            a=self.ac_predict.policy(feed)+self.noise.noise()
            g=1
#            self.last_action.assign(np.mean(a)).op.run()
            
        return a,g
    
    def getLoss(self):
        xp_feed_dict=self._sampleTransitionBatch(batchsize=self.params['batchsize'])
        l=self.sess.run(self.loss,feed_dict=xp_feed_dict)
        return l
    
    def addTransition(self,t):
        self.rpm.addTransition(t)
            
    def _sampleCriticBatch(self):
        sample=self.rpm.sampleTransition(batchsize=self.params["batchsize"])
        
        actor_actions=self.ac_target.policy(sample[3])
        
        cr_batch= {self.cr_predict.input_placeholder: np.array(sample[0],dtype=np.float32),
                   self.cr_predict.action_placeholder: np.array(sample[1],dtype=np.float32),
                   self.cr_target.reward_placeholder: np.array(sample[2],dtype=np.float32),
                   self.cr_target.input_placeholder: np.array(sample[3],dtype=np.float32),
                   self.cr_target.action_placeholder: actor_actions}
        
        return cr_batch
    
    def _sampleActorBatch(self,critic_batch):        
        actor_actions=self.ac_predict.policy(critic_batch[self.cr_predict.input_placeholder])
        critic_grads=self.cr_predict.getGradients({self.cr_predict.input_placeholder : critic_batch[self.cr_predict.input_placeholder],
                                                   self.cr_predict.action_placeholder : actor_actions})
        
        ac_batch= {self.ac_predict.input_placeholder : critic_batch[self.cr_predict.input_placeholder],
                   self.ac_predict.gradients_placeholder : critic_grads[0]}
        
        
        return ac_batch
        
    def trainNet(self):
        critic_batch=self._sampleCriticBatch()
        
        self.sess.run([self.train_critic],feed_dict=critic_batch)
        
        actor_batch=self._sampleActorBatch(critic_batch)
        
        self.sess.run([self.train_actor],feed_dict=actor_batch)
        
        # Create the Timeline object, and write it to a json
#        tl = timeline.Timeline(self.run_metadata.step_stats)
#        ctf = tl.generate_chrome_trace_format()
#        with open('timeline.json', 'w') as f:
#            f.write(ctf)
        
        if self.global_step.eval()%self.params['summary_steps']==0:
            z=critic_batch.copy()
            z.update(actor_batch)
            l,summary=self.sess.run([self.loss_critic,self.merged],feed_dict=z)
            self.current_loss=l
            self.train_writer.add_summary(summary, self.global_step.eval())
        
        if self.global_step.eval()%self.params['checkpoint_steps']==0:
            checkpoint_file = os.path.join(self.traindir,self.params['checkpoint_dir'], 'checkpoint')
            name=self.saver.save(self.sess, checkpoint_file, global_step=self.global_step.eval())
            print("Saving checkpoint: %s"%name)
            
        
        return self.current_loss
        
    def updateTarget(self):
        self.cr_target.updateWeights()
        self.ac_target.updateWeights()
        

    def _writeFrame(self,frame,episode,timestep,picdir):
        ep_dir=os.path.join(picdir,"episode_%.5d"%episode)
        if not os.path.exists(ep_dir):
            os.mkdir(ep_dir)
        name = os.path.join(ep_dir,"step_%.4d.png"%timestep)
        cv2.imwrite(name,frame)
        
    def writeFrame(self,frame,episode,timestep):
        self._writeFrame(frame,episode,timestep,self.picdir)
        
        

if __name__ == '__main__':      
    
    parser = argparse.ArgumentParser()
    parser.add_argument("-E","--env", type=str, help="Mujoco task in Gym, (default: InvertedPendulum-v1)",default='MountainCarContinuous-v0')
    parser.add_argument("-d","--dir", type=str, help="Directory where the relevant training info is stored")
    parser.add_argument("-e","--eval", type=str, help="Evaluation directory. Movies are stored here.")
    parser.add_argument("-c","--checkpoint",type=str, help="Directory of latest checkpoint.")
    args = parser.parse_args()
        
    envname=args.env
    env = gym.make(envname)
    evalenv = gym.make(envname)
    
    params={
            "Env":'Pendulum-v0',
            "episodes":1000,
            "epoches":100000,
            "testruns":30,
            "testeps":0.05,
            "testevery":150000,
            "timesteps":500,#10000,
            "batchsize":64,
            "replaymemory":250000,
            "targetupdate":1,
            "discount":0.99,
            "learningrate_actor":1e-4,
            "learningrate_critic":1e-3,
            "huberloss":False,
            "gradientmomentum":0.99,
            "sqgradientmomentum":0.95,
            "mingradientmomentum":0.00,
            "initexploration":1.0,
            "finalexploration":0.1,
            "finalexpframe":250000,
            "replaystartsize":64,
            "framesize":64,
            "frames":1,
            "actionsize": env.action_space.shape[0],
            "obssize": env.observation_space.shape[0],
            "traindir":"./train_dir",
            "summary_steps":100,
            "skip_episodes": 50,
            "framewrite_episodes":100,
            "checkpoint_dir":'checkpoints',
            "checkpoint_steps":200000,
            "latest_run":args.checkpoint,
            "metricupdate":10,
            "frame_shape":(64,64,3),
            "frame_dtype":np.float32,
            "high_dim":False,
            "action_bound":env.action_space.high,
            "tau":0.001,
            "weight_decay":1e-2
    }
    
    def rescaleFrame(frame):
        ret = np.array(cv2.resize(frame,params["frame_shape"]),dtype=np.uint8)
        return ret
    
    params["Env"]=envname
    
    
    tf.reset_default_graph()
    
    with tf.Session() as sess:
        
        ddpga=DDPGAgent(sess,env,params)
        
        np.save(os.path.join(ddpga.traindir,'params_dict.npy'), params)
        epoche_name=os.path.join(ddpga.traindir,"epoche_stats.tsv")
#        epoche_fd=open(epoche_name,'w+')
        
        evalenv = wrappers.Monitor(evalenv, os.path.join(ddpga.traindir,'monitor'), video_callable=lambda x:x%20==0)
        
        rp_dtype=params["frame_dtype"]
        if params["high_dim"]:
            fshape=np.append(params["frame_shape"],params["frames"])
        else:
#            fshape=np.append(env.obs_dim,params["frames"])
            fshape=params["obssize"]
            
        c=0
        epoche_done=False
        t1Frame=0.0001
        t2Frame=0
        for e in xrange(params['epoches']):
            #episode loop
            print("Starting epoche {}".format(e))
            ep_ctr=0
            t1=time.clock()
            for i in xrange(1,params['episodes']):
                if epoche_done:
                    break

                f = env.reset()
                
                action,_ = ddpga.takeAction()
                
                obs=np.zeros(fshape,dtype=rp_dtype)
                
                for k in range(params["frames"]):
                    f, r, done, _ = env.step(action)
                    
                    if params["high_dim"]:
                        step_obs=rescaleFrame(f)
                    else:
                        step_obs=f
                    obs=step_obs
                
                # time steps
                rewards=[]
                ts=[]
                
                
                rcum=0
                for t in xrange(params['timesteps']):
                    done=False
                    
                    obsNew=np.zeros(fshape,dtype=rp_dtype)
                    
                    if c<params['replaystartsize']:
                        action,g = ddpga.takeAction()
                    else:
                        action,g = ddpga.takeAction(obs)
                    
                    
                    rcum_steps=0    
                    for k in range(params["frames"]):
                        f, r, d, _ = env.step(action)
#                        a=env.render()
                        
                        if params["high_dim"]:
                            step_obs=rescaleFrame(f)
                        else:
                            step_obs=f
                            
                        obsNew=step_obs
                        
                        c+=1
                        rcum_steps+=r
                        rcum+=r
                        ep_ctr+=1
                        
                        if d:
                            done=True
                    if len(obsNew.shape)>1:
                        obsNew=np.array(obsNew)[:,0]
                        
                    ddpga.addTransition([obs,action, rcum_steps,obsNew, np.array([(not done)],dtype=np.bool)])
                    
                    obs=obsNew
                    
                    
                    loss=-1.
                    if c>=params['replaystartsize']:
                        loss=ddpga.trainNet()
                        
                    ddpga.updateTarget()
                    
                    if c%params["testevery"]==0:
                        epoche_done=True
                        
                            
                    if done: 
#                        if i%params["metricupdate"]==0:
                        dtFrame=(t2Frame-t1Frame)
                        t2=time.clock()
                        if t>0:
                            rate=ep_ctr/(t2-t1)
                            print("\r[Epis: {} || it-rate: {} || Loss: {} || Reward: {}|| Frame: {}]".format(i,rate,loss,rcum,c),end='')
                            
                        sys.stdout.flush()
                        ddpga.saveStats(rcum,t,ep_ctr/(t2-t1))
                        break
                    
                dtFrame=(t2Frame-t1Frame)
                t2=time.clock()
                if t>0:
                    rate=ep_ctr/(t2-t1)
                    print("\r[Epoch: {} || it-rate: {} || Loss: {} || Reward: {}|| Frame: {}]".format(e,rate,loss,rcum,c),end='')
                                
                    sys.stdout.flush()
#                    if rcum.shape>1:
                    ddpga.saveStats(rcum,t,ep_ctr/(t2-t1))
                    
                
                
            
#            testq=[]
#            testreward=[]                    
#            for s in range(1,params['testruns']):
#                f = evalenv.reset()
#                
#                action,_ = ddpga.takeAction()
#                
#                obs=np.zeros((84,84,4),dtype=np.uint8)
#                obsNew=np.zeros((84,84,4),dtype=np.uint8)
#                for k in range(4):
#                    f, r, done, _ = env.step(action)
#                    
#                    rframe=rescaleFrame(f)
#                    fframe=np.array(getYChannel(rframe)[:,:,-1]).astype(np.uint8)
#                    obs[:,:,k]=fframe
#                
#                rcum=r
#                qmean=[]
#                done=False
#                for t in xrange(params['timesteps']):
#                    action,g = ddpga.takeAction(obs,params['testeps'])
#                    
#                    for k in range(4):
#                        f, r, d, _ = evalenv.step(action)
#                        rframe=rescaleFrame(f)
#                        fframe=getYChannel(rframe)[:,:,-1]
#                        obsNew[:,:,k]=fframe
#                        
#                        rcum+=r
#                        
#                        if d:
#                            done=True
#                            break
#                    
#                    q=ddpga.q_predict.meanQ(obsNew)
#                    qmean.append(q)
#                    
#                    obs=obsNew
#                    
#                    if done:
#                        testq.append(np.mean(qmean))
#                        testreward.append(rcum)
#                        if s%10==0:
#                            print("[Test: {} || Reward: {} || Mean Q: {}]".format(s,rcum,np.mean(qmean)))
##                        sys.stdout.flush()
#                        break
#            
#            qepoche=np.mean(testq)
#            qepoche_std=np.std(testq)
#            repoche=np.mean(testreward)
#            rmax=np.max(testreward)
#            repoche_std=np.std(testreward)
#            epoche_fd.write("%d\t%.5f\t%.5f\t%.5f\t%.5f\n"%(e,qepoche,qepoche_std,repoche,repoche_std))
#            ddpga.epocheStats(repoche,qepoche,rmax)
#            print("Test stats after epoche {}: Q: {} ({}) || R: {} ({})".format(e,qepoche,qepoche_std,repoche,repoche_std)) 
#            epoche_done=False
#                    
#                
#        epoche_fd.close()
        env.close()