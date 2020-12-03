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
	QAloglog=[]
	for i in range(p.Nruns):
		print("Run no:",i)
		QA,QB,ret,QAlog=main_Qlearning_tab()
		if i==0:
			retlog=ret
			QAloglog=QAlog
			#log_leftcountperc=leftcountperc
		else:
			retlog=np.vstack((retlog,ret))
			QAloglog=np.vstack((QAloglog,QAlog))
			#log_leftcountperc=np.vstack((log_leftcountperc,leftcountperc))
		if (i+1)/p.Nruns==0.25:
			print('25% runs complete')
		elif (i+1)/p.Nruns==0.5:
			print('50% runs complete')
		elif (i+1)/p.Nruns==0.75:
			print('75% runs complete')
		elif (i+1)==p.Nruns:
			print('100% runs complete')
	meanreturns=(np.mean(retlog,axis=0))
	#meanleft=(np.mean(log_leftcountperc,axis=0))
	plt.plot(meanreturns)
	plt.show()
	return QA,QB,retlog,QAloglog

def main_Qlearning_tab():
	QA=np.zeros((1,p.NQ,p.A_asize))#np.random.uniform(low=-0, high=0.01, size=(1,))*np.ones((1,p.NQ,p.A_asize))
	QB=np.zeros((1,p.NQ,p.B_asize))#np.random.uniform(low=-0, high=0.01, size=(1,))*np.ones((1,p.NQ,p.B_asize))#np.zeros((1,p.NQ,p.B_asize))
	returns=[]
	QAlog =[]
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
		QA,QB,ret,QAval=Qtabular(QA,QB,i)
		if i%1==0:
			returns.append(ret)
			QAlog.append(QAval)
	return QA,QB,returns,QAlog

def transition(state,act):
	if state==p.stateA:
		if act==0:
			new_state=p.stateB
		elif act==1:
			new_state=p.state_term_neur
	if state==p.stateB:
		new_state=p.state_term_rew
	return new_state

def maxQ_tab(QA,QB,state):
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
		reward=p.meanreward+np.random.uniform(-p.uncertain,p.uncertain,1)
	else:
		reward=0
	return reward

def epsilon_greedy(state,eps_live,QA,QB):
	if state==p.stateA:
		asize=p.A_asize
		if eps_live>np.random.sample():
			a=np.random.randint(asize)
		else: 
			Qmax,Qmin,a=maxQ_tab(QA,QB,state)
	elif state==p.stateB:
		asize=p.B_asize
		if eps_live>np.random.sample():
			a=np.random.randint(asize)
		else: 
			Qmax,Qmin,a=maxQ_tab(QA,QB,state)
	return a


def minQN(QA,QB,state):
	min_QN=[]
	if state==p.stateA:
		for j in range(p.A_asize):
			Qall=[]
			for i in range(p.NQ):
				Qall.append(QA[0][i][j])
			min_QN.append(np.min(Qall))
			#min_QN_B=QB[0][arg]
	elif state==p.stateB:
		for j in range(p.B_asize):
			Qall=[]
			for i in range(p.NQ):
				Qall.append(QB[0][i][j])
			min_QN.append(np.min(Qall))
			#min_QN_B=QB[0][arg]
	return min_QN

def minQNmod(QA,QB,state):
	min_QNA=[]
	min_QNB=[]
	for j in range(p.A_asize):
		Qall=[]
		for i in range(p.NQ):
			Qall.append(QA[0][i][j])
		min_QNA.append(np.min(Qall))
		#min_QN_B=QB[0][arg]

	for j in range(p.B_asize):
		Qall=[]
		for i in range(p.NQ):
			Qall.append(QB[0][i][j])
		min_QNB.append(np.min(Qall))
		#min_QN_B=QB[0][arg]
	return min_QNA, min_QNB

def epsilon_greedy_min(state,eps_live,QA,QB):
	if state==p.stateA:
		min_QNA,min_QNB=minQNmod(QA,QB,state)
		asize=p.A_asize
		if eps_live>np.random.sample():
			a=np.random.randint(asize)
		else: 
			Qmax,a=maxQ_tab_newnew(min_QNA,min_QNB,state)


	elif state==p.stateB:
		min_QNA,min_QNB=minQNmod(QA,QB,state)
		bsize=p.B_asize
		if eps_live>np.random.sample():
			a=np.random.randint(bsize)
		else: 
			Qmax,a=maxQ_tab_newnew(min_QNA,min_QNB,state)
	return a

def epsilon_greedy(state,eps_live,QA,QB):
	if state==p.stateA:
		asize=p.A_asize
		if eps_live>np.random.sample():
			a=np.random.randint(asize)
		else: 
			Qmax,Qmin,a=maxQ_tab(QA,QB,state)
	elif state==p.stateB:
		asize=p.B_asize
		if eps_live>np.random.sample():
			a=np.random.randint(asize)
		else: 
			Qmax,Qmin,a=maxQ_tab(QA,QB,state)
	return a

def maxQ_tab_new(Qcurr,state):

	Qlist=[]
	if state==p.stateA or state==p.stateB:
		if state==p.stateA:
			asize=p.A_asize
		elif state==p.stateB:
			asize=p.B_asize
		for i in range(asize):
			Qlist.append(Qcurr[i])
		Qmaxnext=np.max(Qlist)
		optact=np.argmax(Qlist)
	else:
		Qmaxnext=0
		optact=0
	return Qmaxnext,optact

def maxQ_tab_newnew(QcurrA,QcurrB,state):

	Qlist=[]
	if state==p.stateA:
		asize=p.A_asize
		for i in range(asize):
			Qlist.append(QcurrA[i])
		Qmaxnext=np.max(Qlist)
		maxinds=[]
		for j in range(len(Qlist)):
			if Qlist[j]==Qmaxnext:
				maxinds.append(j)
		if len(maxinds)>1:
			optact=maxinds[np.random.randint(len(maxinds))]
		else:
			optact=maxinds[0]
	elif state==p.stateB:
		asize=p.B_asize
		for i in range(asize):
			Qlist.append(QcurrB[i])
		Qmaxnext=np.max(Qlist)
		maxinds=[]
		for j in range(len(Qlist)):
			if Qlist[j]==Qmaxnext:
				maxinds.append(j)
		if len(maxinds)>1:
			optact=maxinds[np.random.randint(len(maxinds))]
		else:
			optact=maxinds[0]		
	else:
		Qmaxnext=0
		optact=0
	return Qmaxnext,optact

def findmeanval(QA,QB):
	meanval=(np.sum(QA[0][0:p.A_asize])+np.sum(QB[0][0:p.B_asize]))/(p.A_asize+p.B_asize)
	return meanval

def Qtabular(QA,QB,episode_no):
	initial_state=p.stateA
	state=initial_state
	count=0
	breakflag=0
	#eps_live=1-(p.epsilon_decay*episode_no)
	eps_live=0.1
	doneflag=0
	ret=0
	leftcount=0
	Vmax=0
	Vmin=0
	count=0
	while doneflag==0:
		if state==p.state_term_neur or state==p.state_term_rew:
			break
		a=epsilon_greedy_min(state,eps_live,QA,QB)
		next_state=transition(state,a)
		R=env_step(next_state,a)
		ret=ret+R
		minQnextA,minQnextB=minQNmod(QA,QB,next_state)
		Qmaxnext,optact=maxQ_tab_newnew(minQnextA,minQnextB,next_state)
		rng=np.random.default_rng()
		numbers = rng.choice(p.NQ, size=1, replace=False)
		for i in range(len(numbers)):
			QsubA=QA[0][numbers[i]]
			QsubB=QB[0][numbers[i]]
			if state==p.stateA:
				count=count+1
				#QA[0][a]=QA[0][a]+p.alpha*(R+(p.gamma*Qminnext)-QA[0][a])
				QsubA[a]=QsubA[a]+p.alpha*(R+(p.gamma*(Qmaxnext))-QsubA[a])
				if a==0:
					leftcount=leftcount+1
			elif state==p.stateB:
				#QB[0][a]=QB[0][a]+p.alpha*(R+(p.gamma*Qminnext)-QB[0][a])
				QsubB[a]=QsubB[a]+p.alpha*(R+(p.gamma*Qmaxnext)-QsubB[a])
			QA[0][numbers[i]]=QsubA
			QB[0][numbers[i]]=QsubB
		percleft=QA[0][0][0]-QA[0][0][1]
		if percleft<=0:
			percleft=0
		if state==p.state_term_neur or state==p.state_term_rew:
			doneflag=1
			break
		state=next_state
		#count=count+1
	return QA,QB,ret,leftcount/count

#######################################
if __name__=="__main__":
	QA,QB,retlog,QAloglog=Qlearn_multirun_tab()
	np.savez(str(p.NQ)+"maxmin_"+str(p.Nruns)+"runs_R_"+str(p.meanreward)+"uncertain_"+str(p.uncertain)+".npy.npz",QA,QB,retlog,QAloglog)