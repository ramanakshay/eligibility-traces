import gymnasium
from gymnasium import spaces
import cv2
import numpy as np

class ParticleEnv(gymnasium.Env):
	metadata = {"render_modes": ["human", "rgb_array"], "render_fps": 4}
	
	def __init__(self, render_mode='rgb_array', height=84, width=84, step_size=5, reward_type='dense', 
		  		 reward_scale=None, block=None, start=None, goal=None):
		super(ParticleEnv, self).__init__()

		self.height = height
		self.width = width
		self.step_size = step_size
		self.reward_type = reward_type
		self.reward_scale = np.sqrt(height**2 + width**2) if reward_scale is None else reward_scale
		self.block = block
		self.diagonal = np.sqrt(height**2 + width**2)
		self.render_mode = render_mode
		
		'''
		Define observation space which blocked in between.
		0: Traversable blocks
		1: Blocked
		2: Goal
		'''
		self.observation_space = spaces.Box(low = np.array([0,0],dtype=np.float32), 
									   		high = np.array([self.height-1, self.width-1],dtype=np.float32),
									  		dtype = np.float32)
		
		self.action_space = spaces.Box(low = np.array([-step_size, -step_size],dtype=np.float32), 
									   high = np.array([step_size, step_size],dtype=np.float32),
									   dtype = np.float32)
		
		# Set initial start
		self.start_box = start if start != None else [0,int(0.2*self.height),
													  0, int(0.2*self.width)]

		# Set initial goal
		self.goal_box = goal if goal != None else [self.height - 10, self.height,
												   self.width - 10, self.width]
		# self.goal = np.array([np.random.randint(self.goal_box[0], self.goal_box[1]),
		# np.random.randint(self.goal_box[2], self.goal_box[3])]).astype(np.int32)
		self.goal = np.array([int(0.5*(self.goal_box[0] + self.goal_box[1])), int(0.5*(self.goal_box[2] + self.goal_box[3]))]).astype(np.int32)
	
	def step(self, action):
		prev_state = self.state
		self.state = np.array([self.state[0] + self.step_size * action[0], self.state[1] + self.step_size * action[1]], dtype=np.float32)

		# Clip height and width
		self.state[0] = np.clip(self.state[0], 0, self.height-1)
		self.state[1] = np.clip(self.state[1], 0, self.width-1)
		
		if self.observation[int(self.state[0]), int(self.state[1])]==1 or \
		   self.state[0] in [0,self.height-1] or self.state[1] in [0,self.width-1]:
			reward = -1
			self.state = prev_state
			# print('blocked')
			done = False
		# elif (self.start_box[0] < self.state[0] < self.start_box[1]) and \
		# 	 (self.start_box[2] < self.state[1] < self.start_box[3]):
		# 	# print('BAD')
		# 	reward =
		# 	done = False
		elif self.observation[int(self.state[0]), int(self.state[1])] == 2:
			reward = +100
			done = True
		else:
			reward = -np.linalg.norm(self.state - self.goal) / self.reward_scale if self.reward_type == 'dense' else 0
			done = False
		self._step += 1
		
		info = {}
		info['is_success'] = 1 if reward==1 else 0 

		# Normalize state
		state = np.array([self.state[0] / self.height, self.state[1] / self.width]).astype(np.float32)
		
		self.render()
		return state, reward, done, done,  info
	
	def reset(self, start_state=None, reset_goal=False, goal_state=None, seed=None, options=None):
		# reset_goal = True
		super().reset(seed=seed)
		start_state = np.array(start_state).astype(np.float32) if start_state is not None else None

		# set start state
		if start_state is None:
			self.state = np.array([np.random.randint(self.start_box[0], self.start_box[1]),
								   np.random.randint(self.start_box[2], self.start_box[3])]).astype(np.int32)
		else:
			# start_state[0], start_state[1] = start_state[0] * self.height, start_state[1] * self.width
			self.state = np.array(start_state).astype(np.int32)

		# set goal state
		if reset_goal:
			if goal_state is not None:
				# goal_state[0], goal_state[1] = goal_state[0] * self.height, goal_state[1] * self.width
				self.goal = np.array(goal_state).astype(np.int32)
			else:
				goal = np.array([np.random.randint(self.goal_box[0], self.goal_box[1]),
				np.random.randint(self.goal_box[2], self.goal_box[3])]).astype(np.int32)
				# while np.linalg.norm(self.state - goal) < self.diagonal / 4:
				# 	goal = np.array([np.random.randint(0, self.height), np.random.randint(0, self.width)]).astype(np.int32)
				self.goal = goal

		# observation image
		self.observation = np.zeros((self.height, self.width)).astype(np.uint8)		
		# Set blocked regions
		if self.block is not None:
			for region in self.block:
				block_hmin, block_hmax = int(region[0]), int(region[1])
				block_wmin, block_wmax = int(region[2]), int(region[3])
				for h in range(block_hmin, block_hmax+1):
					for w in range(block_wmin, block_wmax+1):
						self.observation[h, w] = 1

		
		
		# Set goal regions
		goal_hmin, goal_hmax = int(self.goal[0]-10), int(self.goal[0]+10)
		goal_wmin, goal_wmax = int(self.goal[1]-10), int(self.goal[1]+10)
		goal_hmin, goal_hmax = max(0, goal_hmin), min(self.height-1, goal_hmax)
		goal_wmin, goal_wmax = max(0, goal_wmin), min(self.width-1, goal_wmax)
		for h in range(goal_hmin, goal_hmax+1):
			for w in range(goal_wmin, goal_wmax+1):
				self.observation[h,w] = 2
	
		
		self._step = 0
		
		info = {}
		info['is_success'] = 1

		# Normalize state
		state = np.array([self.state[0] / self.height, self.state[1] / self.width]).astype(np.float32)
		
		self.render()
		return state, info


	def render(self):
		# img = np.ones(self.observation.shape).astype(np.uint8) * 255
		# Identify blocked region
		# blocked = np.where(self.observation == 1)
		# img[blocked] = 0
		# hmin, hmax, wmin, wmax = 0,70,40,65
		# img[hmin:hmax, wmin:wmax]  = 0
		img = np.where(self.observation==1, 0, 255).astype(np.uint8)
		
		# hmin, hmax, wmin, wmax = 30,99,50,75
		# img[hmin:hmax, wmin:wmax]  = 0

		hmin, hmax = max(0, self.goal[0]-10), min(self.height-1, self.goal[0] + 10)
		wmin, wmax = max(0, self.goal[1]-10), min(self.width-1, self.goal[1] + 10)
		hmin, hmax, wmin, wmax = int(hmin), int(hmax), int(wmin), int(wmax)
		img[hmin:hmax, wmin:wmax] = 64

		# Mark state
		img[max(0, int(self.state[0])-5):min(self.height-1, int(self.state[0])+5), max(0, int(self.state[1])-5):min(self.width-1, int(self.state[1])+5)] = 128

		width, height = 300, 300
		if width is not None and height is not None:
			dim = (int(width), int(height))
			img = cv2.resize(img, dim, interpolation = cv2.INTER_AREA)
		img = img[..., None]

		if self.render_mode=='rgb_array':
			return cv2.cvtColor(img ,cv2.COLOR_GRAY2RGB)
		else:
			cv2.imshow("Render", img)
			cv2.waitKey(15)

# Code to test the environment
if __name__ == '__main__':
	env = ParticleEnv(height=640, width=640, step_size=10, reward_type='dense', reward_scale=None, start=None, goal=None, block=None)

	for i in range(10):
		state = env.reset()
		done = False
		while not done:
			action = env.action_space.sample()
			next_state, reward, done, info = env.step(action)
			env.render()
			print("State: ", state, "Action: ", action, "Next State: ", next_state, "Reward: ", reward, "Done: ", done, "Info: ", info)
			state = next_state
		print("Episode: ", i)
		print("Final State: ", state)
		print("Final Reward: ", reward)
		print("Final Done: ", done)
		print("Final Info: ", info)
		print("\n\n\n")