import os
import numpy as np
import params_priors as p
import matplotlib.pyplot as plt
import time
import matplotlib.animation as animation
#import priors_tabular as PR

def Qlearn_multirun_tab():
	retlog=[]
	log_leftcountperc=[]
	for i in range(p.Nruns):
		print("Run no:",i)
		QA,QB,QA_double,QB_double,ret,leftcountperc=main_Qlearning_tab()
		if i==0:
			retlog=ret
			log_leftcountperc=leftcountperc
		else:
			retlog=np.vstack((retlog,ret))
			log_leftcountperc=np.vstack((log_leftcountperc,leftcountperc))
		if (i+1)/p.Nruns==0.25:
			print('25% runs complete')
		elif (i+1)/p.Nruns==0.5:
			print('50% runs complete')
		elif (i+1)/p.Nruns==0.75:
			print('75% runs complete')
		elif (i+1)==p.Nruns:
			print('100% runs complete')
	meanreturns=(np.mean(retlog,axis=0))
	meanleft=(np.mean(log_leftcountperc,axis=0))
	plt.plot(meanreturns)
	plt.show()
	return QA,QB,QA_double,QB_double,retlog,log_leftcountperc

def main_Qlearning_tab():
	QA=np.zeros((1,p.A_asize))#np.random.uniform(low=-0, high=0.01, size=(1,))*np.ones((1,p.A_asize))
	QA_double=np.zeros((1,p.A_asize))#np.random.uniform(low=-0, high=0.01, size=(1,))*np.ones((1,p.A_asize))
	QB=np.zeros((1,p.B_asize))#np.random.uniform(low=-0, high=0.01, size=(1,))*np.ones((1,p.B_asize))
	QB_double=np.zeros((1,p.B_asize))#np.random.uniform(low=-0, high=0.01, size=(1,))*np.ones((1,p.B_asize))
	returns=[]
	returns_left=[]
	for i in range(p.episodes):
		if (i+1)/p.episodes==0.25:
			print('25% episodes done')
		elif (i+1)/p.episodes==0.5:
			print('50% episodes done')
		elif (i+1)/p.episodes==0.75:
			print('75% episodes done')
		elif (i+1)/p.episodes==1:
			print('100% episodes done')
			#print(fr)
		QA,QB,QA_double,QB_double,ret,leftcountperc=Qtabular(QA,QB,QA_double,QB_double,i)
		if i%1==0:
			returns.append(ret)
			returns_left.append(leftcountperc)
	return QA,QB,QA_double,QB_double,returns,returns_left

def transition(state,act):
	if state==p.stateA:
		if act==0:
			new_state=p.stateB
		elif act==1:
			new_state=p.state_term_neur
	if state==p.stateB:
		new_state=p.state_term_rew
	return new_state

def maxQ_tab(QA,QB,QA_double,QB_double,state):
	Qlist=[]
	if state==p.stateA:
		asize=p.A_asize
		Q=QA[0]+QA_double[0]
		for i in range(asize):
			Qlist.append(Q[i])
		tab_maxQ=np.max(Qlist)
	elif state==p.stateB:
		asize=p.B_asize
		Q=QB[0]+QB_double[0]
		for i in range(asize):
			Qlist.append(Q[i])
		tab_maxQ=np.max(Qlist)
	if state==p.stateA or state==p.stateB:
		maxind=[]
		for j in range(len(Qlist)):
			if tab_maxQ==Qlist[j]:
				maxind.append(j)
		if len(maxind)>1:
			optact=maxind[np.random.randint(len(maxind))]
		else:
			optact=maxind[0]
		tab_minQ=np.min(Qlist)
	else: 
		tab_minQ=0
		tab_maxQ=0
		optact=0
	return tab_maxQ,tab_minQ, optact

def maxQ_double(QA,QB,state):
	Qlist=[]
	if state==p.stateA:
		asize=p.A_asize
		Q=QA[0]
		for i in range(asize):
			Qlist.append(Q[i])
		tab_maxQ=np.max(Qlist)
	elif state==p.stateB:
		asize=p.B_asize
		Q=QB[0]
		for i in range(asize):
			Qlist.append(Q[i])
		tab_maxQ=np.max(Qlist)
	if state==p.stateA or state==p.stateB:
		maxind=[]
		for j in range(len(Qlist)):
			if tab_maxQ==Qlist[j]:
				maxind.append(j)
		if len(maxind)>1:
			optact=maxind[np.random.randint(len(maxind))]
		else:
			optact=maxind[0]
		tab_minQ=np.min(Qlist)
	else: 
		tab_minQ=0
		tab_maxQ=0
		optact=0
	return tab_maxQ,tab_minQ, optact


def env_step(state,action):
	if state==p.state_term_rew:
		#reward=p.meanreward-1+(2*1*np.random.random_sample())
		reward=p.meanreward+np.random.uniform(-p.uncertain,p.uncertain,1)
	else:
		reward=0
	return reward

def epsilon_greedy(state,eps_live,QA,QB,QA_double,QB_double):
	if state==p.stateA:
		asize=p.A_asize
		if eps_live>np.random.sample():
			a=np.random.randint(asize)
		else: 
			Qmax,Qmin,a=maxQ_tab(QA,QB,QA_double,QB_double,state)
	elif state==p.stateB:
		asize=p.B_asize
		if eps_live>np.random.sample():
			a=np.random.randint(asize)
		else: 
			Qmax,Qmin,a=maxQ_tab(QA,QB,QA_double,QB_double,state)
	return a

def Qtabular(QA,QB,QA_double,QB_double,episode_no):
	initial_state=p.stateA
	state=initial_state
	count=0
	breakflag=0
	#eps_live=1-(p.epsilon_decay*episode_no)
	eps_live=0.1
	doneflag=0
	ret=0
	leftcount=0
	while doneflag==0:
		if state==p.state_term_neur or state==p.state_term_rew:
			break
		#epsilon greedy exploration
		a=epsilon_greedy(state,eps_live,QA,QB,QA_double,QB_double)
		next_state=transition(state,a)
		R=env_step(next_state,a)
		ret=ret+R
		if 0.5>np.random.sample():
			maxQd,minQd,optact_d=maxQ_double(QA,QB,next_state)
			if state==p.stateA and next_state==p.stateB:
				count=count+1
				if a==0:
					leftcount=leftcount+1
				QA[0][a]=QA[0][a]+p.alpha*(R+(p.gamma*QB_double[0][optact_d])-QA[0][a])
			elif state==p.stateA:
				count=count+1
				if a==0:
					leftcount=leftcount+1
				QA[0][a]=QA[0][a]+p.alpha*(R-QA[0][a])
			if state==p.stateB and next_state==p.stateA:
				QB[0][a]=QB[0][a]+p.alpha*(R+(p.gamma*QA_double[0][optact_d])-QB[0][a])
			elif state==p.stateB:
				QB[0][a]=QB[0][a]+p.alpha*(R-QB[0][a])

		else:
			maxQd,minQd,optact_d=maxQ_double(QA_double,QB_double,next_state)
			if state==p.stateA and next_state==p.stateB:
				QA_double[0][a]=QA_double[0][a]+p.alpha*(R+(p.gamma*QB[0][optact_d])-QA_double[0][a])
				count=count+1
				if a==0:
					leftcount=leftcount+1
			elif state==p.stateA:
				QA_double[0][a]=QA_double[0][a]+p.alpha*(R-QA_double[0][a])
				count=count+1
				if a==0:
					leftcount=leftcount+1
			if state==p.stateB and next_state==p.stateA:
				QB_double[0][a]=QB_double[0][a]+p.alpha*(R+(p.gamma*QA[0][optact_d])-QB_double[0][a])
			elif state==p.stateB:
				QB_double[0][a]=QB_double[0][a]+p.alpha*(R-QB_double[0][a])
		
		state=next_state
		percleft=QA[0][0]
		if state==p.state_term_neur or state==p.state_term_rew:
			doneflag=1
			break
		'''	
		if QA[0][0]>QA[0][1]:
			lflag=1
		else:
			lflag=0
		'''
	return QA,QB,QA_double,QB_double,ret,leftcount/count

#######################################
if __name__=="__main__":
	QA,QB,QA_double,QB_double,retlog,log_leftcountperc=Qlearn_multirun_tab()
	np.savez("double_"+str(p.Nruns)+"runs_R_"+str(p.meanreward)+"uncertain_"+str(p.uncertain)+".npy.npz",QA,QB,QA_double,QB_double,retlog,log_leftcountperc)