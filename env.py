# the reinforcement learning painting environment.
# run this file directly to test functionality.
# Qin Yongliang 20171029

import numpy as np

# OpenCV3.x opencv-python
import cv2

# provide shorthand for auto-scaled imshow(). the library is located at github/ctmakro/cv2tools
from cv2tools import vis,filt

# we provide a Gym-like interface. instead of inheriting directly, we steal only the Box descriptor.
import gym
from gym.spaces import Box

lena = cv2.imread('Lenna_neutrual.png').astype('uint8') #saves space
smaller_lena = cv2.resize(lena,dsize=(64,64),interpolation=cv2.INTER_CUBIC)

def load_random_image():
    # load a random image and return. byref is preferred.
    # here we return the same image everytime.
    bias = np.random.uniform(size=(1,1,3))*0.3
    gain = np.random.uniform(size=(1,1,3))*0.7 + 0.7
    randomized_lena = np.clip(smaller_lena * gain + bias, 0, 255).astype('uint8')
    return randomized_lena

# Environment Specification
# observation: tuple(image,image)
# action: Box(8) clamped to [0,1]

class CanvasEnv:
    def __init__(self):
        self.action_dims = ad = 8
        self.action_space = Box(np.array([0.]*ad), np.array([1.]*ad))
        self.target_drawn = False

    def reset(self):
        target = load_random_image()
        # target should be a 3-channel colored image of shape (H,W,3) in uint8
        self.target = target
        self.target_drawn = False
        self.canvas = np.zeros(shape=target.shape, dtype='uint8')+int(0.8*255)
        self.lastdiff = self.diff()
        return self.observation()

    def diff(self):
        # calculate difference between two image. you can use different metrics to encourage different characteristics.
        se = (self.target.astype('float32') - self.canvas.astype('float32'))**2
        mse = np.mean(se)/255.
        return mse

    def observation(self):
        return self.target,self.canvas

    def step(self,action):
        # unpack the parameters
        x1,y1,x2,y2,radius,r,g,b = [np.clip(action[i],0.,1.) for i in range(self.action_dims)]

        # expand range
        x1,y1,x2,y2 = [(v*1.2 - 0.1) for v in [x1,y1,x2,y2]]

        # scaler
        height,width,depth = self.canvas.shape
        sheight,swidth = height*16,width*16 # leftshift bits

        if False:
            # paint a stroke
            cv2.line(
                self.canvas,
                (int(x1*swidth),int(y1*sheight)), # point coord
                (int(x2*swidth),int(y2*sheight)),
                (int(r*255),int(g*255),int(b*255)), # color
                int(radius**2*width*0.2+4), # thickness
                cv2.LINE_AA, # antialiasing
                4 # rightshift bits
            )
        else:
            # paint a dot
            cv2.circle(
                self.canvas,
                (int(x1*swidth),int(y1*sheight)), # point coord
                int(radius**2*width*0.2*16+4*16), # radius
                (int(r*255),int(g*255),int(b*255)), # color
                -1,
                cv2.LINE_AA,
                4
            )

        # calculate reward
        diff = self.diff()
        reward = self.lastdiff - diff # reward is positive if diff decreased
        self.lastdiff = diff

        return self.observation(), reward, False, None # o,r,d,i

    def render(self):
        if self.target_drawn == False:
            vis.show_autoscaled(self.target,limit=300,name='target')
            self.target_drawn = True
        vis.show_autoscaled(self.canvas,limit=300,name='canvas')

if __name__ == '__main__':
    env = CanvasEnv()
    o = env.reset()
    for step in range(2000):
        o,r,d,i = env.step(env.action_space.sample())
        print('step {} reward {}'.format(step,r))
        env.render()
