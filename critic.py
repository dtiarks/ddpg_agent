#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Sun Mar 26 11:08:28 2017

@author: daniel
"""

import numpy as np
import tensorflow as tf

class CriticNet(object):
    def __init__(self,sess,name,params,train=True,wd=None):
        self.params=params
        self.sess=sess
        self.name=name
        self.input_shape=[None ,self.params["obssize"]] #add to hyperparamters
        with tf.name_scope(self.name):
            self.input_placeholder = tf.placeholder(tf.float32,shape=self.input_shape,name="input_plh")
            self.target_placeholder = tf.placeholder(tf.int32,shape=[None,],name="target_plh")
            self.reward_placeholder = tf.placeholder(tf.float32,shape=[None,],name="reward_plh")
            self.action_placeholder = tf.placeholder(tf.float32,shape=[None,1],name="action_plh")
            self.done_placeholder = tf.placeholder(tf.float32,shape=[None,],name="done_plh")
        
        self.train=train
        self.wd=wd
        
        self.params_list=[]
        self.wholder=[]
        self.initholder=[]
        
        if params["high_dim"]:
            self.buildHighDimNet()
        else:
            self.buildLowDimNet()
            
        self.gradients=tf.gradients(self.out,self.action_placeholder)
            
    def buildLowDimNet(self):
        input_layer = self.input_placeholder
        action_layer = self.action_placeholder
        
        with tf.name_scope(self.name):
            with tf.name_scope('fc1'):
                self.W_fc1 = self._weight_variable([self.params["obssize"], 400],"W_fc1",vals=(-1./np.sqrt(self.params["obssize"]),1./np.sqrt(self.params["obssize"])))
                self.params_list.append(self.W_fc1)
                
                self.b_fc1 = self._bias_variable([400],"b_fc1",vals=(-1./np.sqrt(self.params["obssize"]),1./np.sqrt(self.params["obssize"])))
                self.params_list.append(self.b_fc1)
            
                h_fc1 = tf.nn.relu(tf.matmul(input_layer, self.W_fc1) + self.b_fc1)
                
            with tf.name_scope('fc2'):
                self.W_fc2 = self._weight_variable([400+self.params["actionsize"], 300],"W_fc2",vals=(-1./np.sqrt(400),1./np.sqrt(400)))
                self.params_list.append(self.W_fc2)
                
                self.b_fc2 = self._bias_variable([300],"b_fc2",vals=(-1./np.sqrt(400),1./np.sqrt(400)))
                self.params_list.append(self.b_fc2)
                
                h_fc1a=tf.concat(1,[h_fc1,action_layer])
                h_fc2 = tf.nn.relu(tf.matmul(h_fc1a, self.W_fc2) + self.b_fc2)
                
#                self.W_fc2_action = self._weight_variable([self.params['actionsize'], 300],"W_fc2_action",vals=(-1./np.sqrt(self.params['actionsize']),1./np.sqrt(self.params['actionsize'])))
#                self.params_list.append(self.W_fc2_action)
            
                #h_fc2 = tf.nn.relu(tf.matmul(h_fc1, self.W_fc2) + tf.matmul(action_layer, self.W_fc2_action) + self.b_fc2)
                
            with tf.name_scope('output'):
                self.W_fc3 = self._weight_variable([300, 1],"W_out")
                self.params_list.append(self.W_fc3)
                
                self.b_fc3 = self._bias_variable([1],"b_out")
                self.params_list.append(self.b_fc3)
                
            self.out=tf.squeeze(tf.matmul(h_fc2, self.W_fc3) + self.b_fc3,[1])
                              
        return self.out
        
    def buildHighDimNet(self):
        input_layer = self.input_placeholder

        with tf.name_scope(self.name):
            with tf.name_scope('conv1'):
                # 8x8 conv, 4 inputs, 32 outputs, stride=4
                self.W_conv1 = self._weight_variable([8, 8, 4, 32],"W_conv1")
                self.b_conv1 = self._bias_variable([32],"b_conv1")
                h_conv1 = tf.nn.relu(self._conv2d(input_layer, self.W_conv1, 4) + self.b_conv1)
    
            with tf.name_scope('conv2'):
                # 4x4 conv, 32 inputs, 64 outputs, stride=2
                self.W_conv2 = self._weight_variable([4, 4, 32, 64],"W_conv2")
                self.b_conv2 = self._bias_variable([64],"b_conv2")
                h_conv2 = tf.nn.relu(self._conv2d(h_conv1, self.W_conv2, 2) + self.b_conv2)
                
            with tf.name_scope('conv3'):
                # 3x3 conv, 64 inputs, 64 outputs, stride=1
                self.W_conv3 = self._weight_variable([3, 3, 64, 64],"W_conv3")
                self.b_conv3 = self._bias_variable([64],"b_conv3")
                h_conv3 = tf.nn.relu(self._conv2d(h_conv2, self.W_conv3, 1) + self.b_conv3)
            
            dim=h_conv3.get_shape()
            dims=np.array([d.value for d in dim])
            reshaped_dim = np.prod(dims[1:])
            with tf.name_scope('dense1'):
                self.W_fc1 = self._weight_variable([reshaped_dim, 512],"W_fc1")
                self.b_fc1 = self._bias_variable([512],"b_fc1")
    
                h_conv3_flat = tf.reshape(h_conv3, [-1, reshaped_dim])
                h_fc1 = tf.nn.relu(tf.matmul(h_conv3_flat, self.W_fc1) + self.b_fc1)
                
            with tf.name_scope('output'):
                self.W_fc2 = self._weight_variable([512, self.params['actionsize']],"W_fc2")
                self.b_fc2 = self._bias_variable([self.params['actionsize']],"b_fc2")
    
                self.action_logits=tf.add(tf.matmul(h_fc1, self.W_fc2), self.b_fc2,"logits")

        
        self.greedy_actions=tf.argmax(self.action_logits,1)
        return self.action_logits
    
    
    def meanQ(self,state_feed):
        feed=np.expand_dims(state_feed,axis=0)
        s=np.array(feed,dtype=np.float32)/255.
        q = self.sess.run(self.action_logits,
                          feed_dict={self.images_placeholder: s})
        
        qmean=np.mean(q)

        return qmean
    
    def estimateQ(self):
        return self.out
    
    def estimateTarget(self):
        ret=self.reward_placeholder+tf.scalar_mul(self.params['discount'],self.out)
        return ret
    
    def getGradients(self,feed):
        ret=self.sess.run(self.gradients,feed_dict=feed)
        return ret
    
    def setWeightUpdate(self,wlist=None):
        if wlist is not None:
            for i in range(len(wlist)):
                self.wholder.append(self.params_list[i].assign(tf.scalar_mul(1.-self.params['tau'],self.params_list[i])+tf.scalar_mul(self.params['tau'],wlist[i])))
                
    def updateInitWeights(self,wlist=None):
        holder=[]
        if wlist:
            for i in range(len(wlist)):
                self.initholder.append(self.params_list[i].assign(wlist[i]))
                
#            self.sess.run(holder)
    
    def updateWeights(self):
        self.sess.run(self.wholder)


    def _weight_variable(self,shape,name=None,vals=(-3*1e-3,3*1e-3)):
        initial = tf.random_uniform(shape, minval=vals[0],maxval=vals[1])
        var=tf.Variable(initial,trainable=self.train,name=name)
        if self.wd:
            weight_decay = tf.multiply(tf.nn.l2_loss(var), self.wd, name='weight_loss')
            tf.add_to_collection('losses', weight_decay)
        return var

    def _bias_variable(self,shape,name=None,vals=(-3*1e-3,3*1e-3)):
        initial = tf.random_uniform(shape, minval=vals[0],maxval=vals[1])
        var=tf.Variable(initial,trainable=self.train,name=name)
        if self.wd:
            weight_decay = tf.multiply(tf.nn.l2_loss(var), self.wd, name='weight_loss')
            tf.add_to_collection('losses', weight_decay)
        return var

    def _conv2d(self,x, W, s):
        return tf.nn.conv2d(x, W, strides=[1, s, s, 1], padding='VALID')