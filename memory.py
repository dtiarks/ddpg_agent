#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Wed Mar 15 18:26:22 2017

@author: daniel
"""

import numpy as np
import time

class ReplayMemory():
    
    frame_shape=(84,84,4)
    
    def __init__(self,size,frame_shape=(84,84,4),dtype=np.float32):
        self.size=np.array([size],dtype=np.uint32)
        
        self.tail=0
        self.count=0
        self.index=np.arange(self.size[0])
        self.frame_dtype=dtype
        
        
        
        self.frame_buffer=np.empty(np.append(self.size,frame_shape),dtype=self.frame_dtype)
        self.action_buffer=np.empty((size,1),dtype=np.float32)
        self.reward_buffer=np.empty(size,dtype=np.float32)
        self.new_frame_buffer=np.empty(np.append(self.size,frame_shape),dtype=self.frame_dtype)
        self.done_buffer=np.empty(self.size,dtype=np.bool)
        
        
    def addTransition(self,trans):
        self.tail=self.count%self.size[0]
            
        self.frame_buffer[self.tail,...]=trans[0]
        self.action_buffer[self.tail,...]=trans[1]
        self.reward_buffer[self.tail]=trans[2]
        self.new_frame_buffer[self.tail,...]=trans[3]
        self.done_buffer[self.tail,...]=trans[4]
        
        self.count+=1
        
            
    def sampleTransition(self,batchsize=32):
        if self.count>self.size[0]:
            maxidx=self.size[0]
        else:
            maxidx=self.count
        
        idx=np.random.choice(self.index[:maxidx],size=batchsize)
        
        ret=[self.frame_buffer[idx,...],
             self.action_buffer[idx],
             self.reward_buffer[idx],
             self.new_frame_buffer[idx,...],
             self.done_buffer[idx]]
        
        return ret
        
        
        
if __name__ == '__main__':   
    M=ReplayMemory(200000)
    s=10*np.random.rand(84,84,4)
    
    
    for i in range(5000):
        t=[s,2*i,3*i,s,5*i]
        M.addTransition(t)
        
    t1=time.clock()
    for i in range(2):
        print M.sampleTransition()[0].shape        
        
    t2=time.clock()
    print i/(t2-t1)