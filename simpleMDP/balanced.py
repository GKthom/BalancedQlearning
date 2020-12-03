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
	log_betatrack=[]
	for i in range(p.Nruns):
		print("Run no:",i)
		QA,QB,ret,leftcountperc,betamatA,betamatB,betatrack=main_Qlearning_tab()
		print(len(betatrack))
		if i==0:
			retlog=ret
			log_leftcountperc=leftcountperc
			log_betatrack=betatrack[0:4900]
		else:
			retlog=np.vstack((retlog,ret))
			log_leftcountperc=np.vstack((log_leftcountperc,leftcountperc))
			log_betatrack=np.vstack((log_betatrack,betatrack[0:4900]))
		if (i+1)/p.Nruns==0.25:
			print('25% runs complete')
		elif (i+1)/p.Nruns==0.5:
			print('50% runs complete')
		elif (i+1)/p.Nruns==0.75:
			print('75% runs complete')
		elif (i+1)==p.Nruns:
			print('100% runs complete')
		#print(retlog)
		#time.sleep(4)
	meanreturns=(np.mean(retlog,axis=0))
	meanleft=(np.mean(log_leftcountperc,axis=0))
	plt.plot(meanreturns)
	plt.show()
	return QA,QB,retlog,log_leftcountperc,betamatA,betamatB,log_betatrack

def main_Qlearning_tab():
	QA=np.zeros((1,p.A_asize))#$np.random.uniform(low=-0, high=0.01, size=(1,))*np.ones((1,p.A_asize))#np.zeros((1,p.A_asize))#
	QB=np.zeros((1,p.B_asize))##np.random.uniform(low=-0, high=0.01, size=(1,))*np.ones((1,p.B_asize))#np.zeros((1,p.B_asize))#np.random.uniform(low=-0, high=0.01, size=(1,))*np.ones((1,p.B_asize))
	betamatA=1*np.random.rand(1,p.A_asize)
	betamatB=1*np.random.rand(1,p.B_asize)
	returns=[]
	returns_left=[]
	betatrack=[]
	betatrackmean=[]
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
		QA,QB,ret,leftcountperc,betamatA,betamatB,betatrack=Qtabular(QA,QB,i,betamatA,betamatB,betatrack)
		if i%1==0:
			returns.append(ret)
			returns_left.append(leftcountperc)
	return QA,QB,returns,returns_left,betamatA,betamatB,betatrack

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
		#tab_maxQ=np.mean(Qlist)
	else: 
		tab_minQ=0
		tab_maxQ=0
		optact=0
	return tab_maxQ,tab_minQ, optact

def env_step(state,action):
	if state==p.state_term_rew:
		#reward=p.meanreward-1+(2*np.random.random_sample())
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

def deltadashmethod(QAold,QBold,QAcurrent,QBcurrent,state,a,R,next_state,meanval,episode_no,betamatA,betamatB,betaold):

	betaold=1
	betainit=betaold
	betamax=1
	betamin=0
	etamin=0
	Qmaxnextold,Qminnextold,optactold=maxQ_tab(QAold,QBold,next_state)
	Qmaxnext,Qminnext,optact=maxQ_tab(QAcurrent,QBcurrent,next_state)
	eta=0.2
	if state==p.stateA:
		beta_n_1=betaold#betamatA[0][a]
		td_n_1=R+p.gamma*((beta_n_1*Qmaxnext)+(1-beta_n_1)*Qminnext)-QAcurrent[0][a]
		beta_n=betaold#betamatA[0][a]
		td_n=R+p.gamma*((beta_n*Qmaxnextold)+(1-beta_n)*Qminnextold)-QAold[0][a]
		delb=td_n_1-((1-p.alpha)*td_n)
		delbnew=delb+eta*td_n
		deltadash=td_n_1+eta*td_n
		if Qmaxnext>Qminnext:
			beta=(deltadash-R+QAcurrent[0][a]-p.gamma*Qminnext)/(Qmaxnext-Qminnext)/p.gamma
		else:
			beta=betamax#betamatA[0][a]
	elif state==p.stateB:
		beta_n_1=betaold#betamatB[0][a]
		td_n_1=R+p.gamma*((beta_n_1*Qmaxnext)+(1-beta_n_1)*Qminnext)-QBcurrent[0][a]
		beta_n=betaold#betamatB[0][a]
		td_n=R+p.gamma*((beta_n*Qmaxnextold)+(1-beta_n)*Qminnextold)-QBold[0][a]
		delb=td_n_1-((1-p.alpha)*td_n)
		delbnew=delb+eta*td_n
		deltadash=td_n_1+eta*td_n
		if Qmaxnext>Qminnext:
			beta=(deltadash-R+QBcurrent[0][a]-p.gamma*Qminnext)/(Qmaxnext-Qminnext)/p.gamma
		else:
			beta=betamax#betamatB[0][a]
	else:
		beta=betamax
	if beta>betamax:
		beta=betamax
	elif beta<betamin:
		beta=betamin
	return beta

def findmeanval(QA,QB):
	meanval=(np.sum(QA[0][0:p.A_asize])+np.sum(QB[0][0:p.B_asize]))/(p.A_asize+p.B_asize)
	return meanval

def Qtabular(QA,QB,episode_no,betamatA,betamatB,betatrack):
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
	QAold=QA.copy()
	QBold=QB.copy()
	QAcurrent=QA.copy()
	QBcurrent=QB.copy()
	betaold=1
	while doneflag==0:
		if state==p.state_term_neur or state==p.state_term_rew:
			break
		#epsilon greedy exploration
		a=epsilon_greedy(state,eps_live,QA,QB)
		next_state=transition(state,a)
		R=env_step(next_state,a)
		ret=ret+R
		meanval=findmeanval(QA,QB)
		betascore=deltadashmethod(QAold,QBold, QAcurrent, QBcurrent,state,a,R,next_state,meanval,episode_no,betamatA,betamatB,betaold)
		betaold=betascore
		Qmaxnext,Qminnext,optact=maxQ_tab(QA,QB,next_state)
		QAold=QA.copy()
		QBold=QB.copy()
		if state==p.stateA:
			count=count+1
			betamatA[0][a]=betascore
			if a==0:
				betatrack.append(betascore)
			elif len(betatrack)>0:  
				betatrack.append(betatrack[len(betatrack)-1])
			#QA[0][a]=QA[0][a]+p.alpha*deltadash
			QA[0][a]=QA[0][a]+p.alpha*(R+(p.gamma*((1-betascore)*Qminnext)+(betascore*Qmaxnext))-QA[0][a])
			if a==0:
				leftcount=leftcount+1
		elif state==p.stateB:
			#betatrack.append(betascore)
			betamatB[0][a]=betascore
			#QB[0][a]=QB[0][a]+p.alpha*deltadash
			QB[0][a]=QB[0][a]+p.alpha*(R+(p.gamma*((1-betascore)*Qminnext)+(betascore*Qmaxnext))-QB[0][a])
		
		QAcurrent=QA.copy()
		QBcurrent=QB.copy()
		if state==p.state_term_neur or state==p.state_term_rew:
			doneflag=1
			break
		state=next_state

		#count=count+1
	return QA,QB,ret,leftcount/count,betamatA,betamatB,betatrack

#######################################
if __name__=="__main__":
	QA,QB,retlog,log_leftcountperc,betamatA,betamatB,betatrack=Qlearn_multirun_tab()
	np.savez("balanced"+str(p.Nruns)+"runs_R_"+str(p.meanreward)+"uncertain_"+str(p.uncertain)+".npy.npz",QA,QB,retlog,log_leftcountperc,betatrack)