
import numpy as np
# from pyibl import Agent
import random as random
import itertools
from math import exp
import math 
import sys 
from collections import deque
from itertools import count
from speedyibl import Agent 

class IBLAgent_td(Agent):
	mkid = next(itertools.count())

	def __init__(self, config):
		super(IBLAgent_td, self).__init__(default_utility=config.cog.default_utility,lendeque=config.cog.max_references, outcome=False)
		self.c = config
# 		self.select_position()
		self.outcomes = {}
		# self.alpha = 0.1
		self.epsilon = self.c.eps.initial
		self.__ep = 0
		self.goods = 0
		'''
		:param int agentID: Agent's ID
		:param dict config: Dictionary containing hyperparameters
		'''
		self.episodeCounter = 0
		self.c.id = IBLAgent_td.mkid
		self.Temps = {}

	def generate_outcomes(self, y, x, holding):
		self.outcomes[(y, x, holding)] = [self.default_utility]*self.c.outputs
	
	def move(self, y, x, holding, explore=True):
		# '''
		# Returns an action from the ibl agent instance.
		# '''
		holding = holding*self.goods 
		self.t += 1
		# if (s_hash) not in self.options:
		# 	self.generate_options(s_hash)
		if self.episodeCounter > self.__ep:
			self.epsilon = max((self.epsilon * self.c.eps.discount), 0)
		if self.episodeCounter > self.__ep:
			self.__ep += 1
		# if explore and random.random() < self.__epsilon:
		# 	self.drl.action = random.randrange(self.c.outputs)
		# 	options = [{"action": self.drl.action, "state": s_hash}]
		# else:
		if (y, x, holding) not in self.outcomes:
			self.last_action = random.randrange(self.c.outputs)
			self.generate_outcomes(y, x, holding)
		elif explore and random.random() < self.epsilon:
			self.last_action = self.boltzchoose(y, x, holding)
		else:
			self.last_action = self.choose_td(y, x, holding)
		
		self.option = (y, x, holding, self.last_action)

		self.x = x
		self.y = y
		self.holding = holding

		return self.last_action



	def feedback(self, reward, terminal, y, x, holding):
		# '''
		# Feedback is passed to the deep rl agent instance.
		# :param float: Reward received during transition
		# :param boolean: Indicates if the transition is terminal
		# :param tensor: State/Observation
		# '''
		if self.c.mamethod == 'leniency':
			if (self.y, self.x, self.holding) not in self.Temps:
				self.Temps[(self.y, self.x, self.holding)] = np.ones(self.c.outputs)*self.c.len.max
			temp_action = self.Temps[(self.y, self.x, self.holding)][self.last_action]
			self.leniency = 1 - np.exp(-self.c.len.theta*temp_action)

		if terminal:
			self.episodeCounter += 1
# 		
		self.respond(None)
	
		if y == self.y and x == self.x:
			outcome = reward
		else:
			if self.holding + holding == 1:
				self.goods = y*x
			if terminal > 0:
				best_utility = 0
			elif (y, x, holding*self.goods) in self.outcomes:
				utilities = self.blend_compute(self.t, y, x, holding*self.goods)
				best_utility = max(utilities,key=lambda x:x[0])[0]
			else:
				best_utility = self.default_utility

			outcome = reward + self.c.gamma*best_utility - self.outcomes[(self.y, self.x, self.holding)][self.last_action]
		
		if self.c.mamethod == 'leniency':
			if outcome > 0 or random.random() > self.leniency: # self.drl.replay_memory._episode[-1][7]:
				self.outcomes[(self.y, self.x, self.holding)][self.last_action] += self.c.alpha*outcome

			if (y, x, holding*self.goods) not in self.Temps:
				temp_mean = self.c.len.max
			else:
				temp_mean = np.mean(self.Temps[(y, x, holding*self.goods)])
			if terminal:
				self.Temps[(self.y, self.x, self.holding)][self.last_action] = self.c.len.delta*self.Temps[(self.y, self.x, self.holding)][self.last_action]
			else:
				self.Temps[(self.y, self.x, self.holding)][self.last_action] = self.c.len.delta*((1-self.c.len.tau)*self.Temps[(self.y, self.x, self.holding)][self.last_action] + self.c.len.tau*temp_mean)
			
		elif self.c.mamethod == 'hysteretic':
			if outcome > 0:
				self.outcomes[(self.y, self.x, self.holding)][self.last_action] += self.c.alpha*outcome
			else:
				self.outcomes[(self.y, self.x, self.holding)][self.last_action] += self.c.hys.beta*outcome
		else:
			self.outcomes[(self.y, self.x, self.holding)][self.last_action] += self.c.alpha*outcome
	
	def blend_compute(self, t, y, x, holding):
		outcomes = self.outcomes[(y, x, holding)]
		blends = []
		for a,i in zip(range(self.c.outputs),count()):
			o = (y, x, holding, a)
			if o in self.instance_history:
				p = self.CompProbability(t,o)
				result = outcomes[a]*p[0] + p[1]*self.default_utility
				blends.append((result,i))
			else:
				blends.append((self.default_utility,i))
		return blends 

	def choose_td(self, y, x, holding):
		utilities = self.blend_compute(self.t, y, x, holding)
		best_utility = max(utilities,key=lambda x:x[0])[0]
		best = random.choice(list(filter(lambda x: x[0]==best_utility,utilities)))[1]
		return best 
	
	def boltzchoose(self, y, x, holding):
		utilities = self.blend_compute(self.t, y, x, holding)
		actions = []
		P = []
		for u in utilities:
			actions.append(u[1])
			P.append(u[0])
		P = np.asarray(P)
		# print(P)
		P = np.exp(P/0.8)
		P = P/sum(P)
		best = np.random.choice(actions,p=P)
		return best 

