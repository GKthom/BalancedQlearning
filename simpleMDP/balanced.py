import os
import numpy as np
import params as p
import matplotlib.pyplot as plt
import time
import matplotlib.animation as animation
import matplotlib.patches as patches


def Qlearn_multirun_tab():
	#This function just runs multiple instances of 
	#Q-learning. Doing so helps obtain an average performance 
	#measure over multiple runs.
	retlog=[] # log of returns of all episodes, in all runs
	bumpcountlog=[]
	for i in range(p.Nruns):
		print("Run no:",i)
		Q,ret,betamatrix,bumpcountret,visitmap,ret_totcnt=main_Qlearning_tab()#call Q learning
		if i==0:
			retlog=ret_totcnt
		else:
			retlog=np.vstack((retlog,ret_totcnt))
		if (i+1)/p.Nruns==0.25:
			print('25% runs complete')
		elif (i+1)/p.Nruns==0.5:
			print('50% runs complete')
		elif (i+1)/p.Nruns==0.75:
			print('75% runs complete')
		elif (i+1)==p.Nruns:
			print('100% runs complete')
	return Q, retlog,betamatrix,bumpcountlog,visitmap

def main_Qlearning_tab():
	#This calls the main Q learning algorithm
	Q=np.zeros((p.a,p.b,p.A)) # initialize Q function as zeros
	betamatrix=np.ones((p.a,p.b,p.A))
	visitmap=np.zeros((p.a,p.b,p.A))
	goal_state=p.targ#target point
	returns=[]#stores returns for each episode
	bumpcountret=[]
	ret=0
	Qimall=[]
	totcnt=0
	ret_totcnt=[]
	
	while totcnt<p.totcountlim:

		if (totcnt+1)/p.totcountlim==0.25:
			print('25% episodes done')
		elif (totcnt+1)/p.totcountlim==0.5:
			print('50% episodes done')
		elif (totcnt+1)/p.totcountlim==0.75:
			print('75% episodes done')
		elif (totcnt+1)/p.totcountlim==1:
			print('100% episodes done')
		Q,ret,betamatrix,bumpcount,visitmap,ret_totcnt,totcnt=Qtabular(Q,betamatrix,visitmap,totcnt,returns,ret_totcnt)#call Q learning
		if totcnt%1==0:
			returns.append(ret)#compute return offline- can also be done online, but this way, a better estimate can be obtained
			bumpcountret.append(bumpcount)
			#print(totcnt)
	return Q, returns,betamatrix,bumpcountret,visitmap,ret_totcnt

def deltadashmethod(Qold,Qcurrent,state,a,R,next_state,episode_no,betainit):
	beta_n=betainit
	beta_n_1=betainit
	betamax=1
	betamin=0
	roundedstate=staterounding(state)
	Qmaxnextold,Qminnextold,optactold=maxQ_tab(Qold,next_state)
	Qmaxnext,Qminnext,optact=maxQ_tab(Qcurrent,next_state)
	eta=0.2
	td_n_1=R+p.gamma*((beta_n_1*Qmaxnext)+(1-beta_n_1)*Qminnext)-Qcurrent[roundedstate[0],roundedstate[1],a]
	td_n=R+p.gamma*((beta_n*Qmaxnextold)+(1-beta_n)*Qminnextold)-Qold[roundedstate[0],roundedstate[1],a]
	delb=td_n_1-((1-p.alpha)*td_n)
	delbnew=delb+eta*td_n
	deltadash=td_n_1+eta*td_n
	if Qmaxnext>Qminnext:
		beta=(deltadash-R+Qcurrent[roundedstate[0],roundedstate[1],a]-p.gamma*Qminnext)/(Qmaxnext-Qminnext)/p.gamma
	else:
		beta=betamax
	if beta>betamax:
		beta=betamax
	elif beta<betamin:
		beta=betamin
	return deltadash,beta

def Qtabular(Q,betamatrix,visitmap,totcnt,returns,ret_totcnt):
	initial_state=np.array([(p.a-1)*np.random.random_sample(), (p.b-1)*np.random.random_sample()])
	#initial_state=np.array([2,2])
	Qold=Q.copy()
	Qcurrent=Q.copy()
	rounded_initial_state=staterounding(initial_state)
	while p.world[rounded_initial_state[0],rounded_initial_state[1]]==1:
		initial_state=np.array([(p.a-1)*np.random.random_sample(), (p.b-1)*np.random.random_sample()])
		rounded_initial_state=staterounding(initial_state)
	state=initial_state.copy()
	count=0
	breakflag=0

	eps_live=1-(totcnt/p.totcountlim)
	ret=0
	bumpcount=0
	target_state=p.targ
	betainit=1
	#while np.linalg.norm(state-target_state)>p.thresh:
	for i in range(p.breakthresh):
		count=count+1
		if breakflag==1 or totcnt>=p.totcountlim:
			break
		if count>p.breakthresh:
			breakflag=1
		if eps_live>np.random.sample():
			a=np.random.randint(p.A)
		else:
			Qmax,Qmin,a=maxQ_tab(Q,state)
		next_state=transition(state,a)
		roundedstate=staterounding(state)
		roundednextstate=staterounding(next_state)
		visitmap[roundednextstate[0],roundednextstate[1],a]=visitmap[roundednextstate[0],roundednextstate[1],a]+1
		if p.world[roundednextstate[0],roundednextstate[1]]==0 or next_state[0]>=p.a or next_state[0]<=0 or next_state[1]>=p.b or next_state[1]<=0:	
			if np.linalg.norm(next_state-target_state)<=p.thresh:
				R=p.highreward	
			else:
				R=p.livingpenalty
		else: 
			R=p.penalty
			next_state=state.copy()
			bumpcount=bumpcount+1
		totcnt+=1
		if len(returns)==0:
			ret_totcnt.append(0)
		elif len(returns)<=100:
			ret_totcnt.append(np.mean(returns))
		else: 
			ret_totcnt.append(np.mean(returns[(len(returns)-100):len(returns)]))
		ret=ret+R
		deltadash,betascore=deltadashmethod(Qold,Qcurrent,state,a,R,next_state,totcnt,betainit)#find beta'
		betainit=((betainit*totcnt)+betascore)/(totcnt+1)
		betamatrix[roundedstate[0],roundedstate[1],a]=betascore
		Qmaxnext,Qminnext,aoptnext=maxQ_tab(Q,next_state)
		Qold=Q.copy()
		Qtarget=R+p.gamma*((betascore*Qmaxnext)+((1-betascore)*Qminnext))-Q[roundedstate[0],roundedstate[1],a]
		Q[roundedstate[0],roundedstate[1],a]=Q[roundedstate[0],roundedstate[1],a]+(p.alpha*Qtarget)
		Qcurrent=Q.copy()
		if np.linalg.norm(next_state-target_state)<=p.thresh:
			break
		state=next_state.copy()
	return Q,ret,betamatrix,bumpcount,visitmap,ret_totcnt,totcnt


def maxQ_tab(Q,state):
	#get max of Q values and corresponding action
	Qlist=[]
	roundedstate=staterounding(state)
	for i in range(p.A):
		Qlist.append(Q[roundedstate[0],roundedstate[1],i])
	tab_maxQ=np.max(Qlist)
	tab_minQ=np.min(Qlist)
	maxind=[]
	for j in range(len(Qlist)):
		if tab_maxQ==Qlist[j]:
			maxind.append(j)
	#print(maxind)
	if len(maxind)>1:
		optact=maxind[np.random.randint(len(maxind))]
	else:
		optact=maxind[0]
	return tab_maxQ,tab_minQ,optact

def optpol_visualize(Qp):
	for i in range(p.a):
		for j in range(p.b):
			if p.world[i,j]==0:
				Qmaxopt,Qminopt,optact=maxQ_tab(Qp,[i,j])
				if optact==0:
					plt.scatter(i,j,color='red')
				elif optact==1:
					plt.scatter(i,j,color='green')
				elif optact==2:
					plt.scatter(i,j,color='blue')
				elif optact==3:
					plt.scatter(i,j,color='yellow')

	plotmap(p.world)
	plt.show()

def transition(state,act):
	n1 = np.random.uniform(low=-0.2, high=0.2, size=(1,))# x noise
	n2 = np.random.uniform(low=-0.2, high=0.2, size=(1,))# y noise
	new_state=state.copy()
	if act==0:
		new_state[0]=state[0]
		new_state[1]=state[1]+1#move up
	elif act==1:
		new_state[0]=state[0]+1#move right
		new_state[1]=state[1]
	elif act==2:
		new_state[0]=state[0]
		new_state[1]=state[1]-1#move down
	elif act==3:
		new_state[0]=state[0]-1#move left
		new_state[1]=state[1]

	new_state[0]=new_state[0]+n1
	new_state[1]=new_state[1]+n2
	return new_state

########Additional functions for visualization######
def plotmap(worldmap):
	#plots the obstacle map
	for i in range(p.a):
		for j in range(p.b):
			if worldmap[i,j]>0:
				plt.scatter(i,j,color='black')
	plt.show()

def staterounding(state):
	#rounds off states
	roundedstate=[0,0]
	roundedstate[0]=int(np.around(state[0]))
	roundedstate[1]=int(np.around(state[1]))
	if roundedstate[0]>=(p.a-1):
		roundedstate[0]=p.a-2
	elif roundedstate[0]<1:
		roundedstate[0]=1
	if roundedstate[1]>=(p.b-1):
		roundedstate[1]=p.b-2
	elif roundedstate[1]<=0:
		roundedstate[1]=1
	return roundedstate

def opt_pol(Q,state,goal_state):
	#shows optimal policy
	plt.figure(0)
	plt.ion()
	for i in range(p.a):
		for j in range(p.b):
			if p.world[i,j]>0:
				plt.scatter(i,j,color='black')
	plt.show()
	pol=[]
	statelog=[]
	count=1
	while np.linalg.norm(state-goal_state)>=1:
		Qm,Qmin,a=maxQ_tab(Q,state)
		if np.random.sample()>0.9:
			a=np.random.randint(p.A)
		next_state=transition(state,a)
		roundednextstate=staterounding(next_state)
		if p.world[roundednextstate[0],roundednextstate[1]]==1:
			next_state=state.copy()
		pol.append(a)
		statelog.append(state)
		print(state)
		plt.ylim(0, p.b)
		plt.xlim(0, p.a)
		plt.scatter(state[0],state[1],(60-count*0.4),color='blue')
		plt.draw()
		plt.pause(0.1)
		state=next_state.copy()
		print(count)
		if count>=100:
			break
		count=count+1
	return statelog,pol

def mapQ(Q):
	#plots a map of the value function
	fig=plt.figure(1)
	plt.ion
	Qmap=np.zeros((p.a,p.b))
	for i in range(p.a):
		for j in range(p.b):
 			Qav=0
 			for k in range(p.A):
 				Qav=Qav+Q[i,j,k]
 			Qmap[i,j]=Qav
	Qmap=Qmap-np.min(Qmap)
	if np.max(Qmap)>0:
		Qmap=Qmap/np.max(Qmap)

	return Qmap


def betamap_vid(betamatrix):
	Q=betamatrix
	#plt.figure(1)
	#fig=plt.figure(0)
	#plt.ion
	Qmap=np.zeros((p.a,p.b))
	for i in range(p.a):
		for j in range(p.b):
 			Qav=0
 			for k in range(p.A):
 				Qav=Qav+Q[i,j,k]
 			Qmap[i,j]=Qav
 	#plt.imshow(Qmap)
	Qfig=plt.imshow(np.rot90(np.fliplr(Qmap)),cmap="hot")
	return Qfig


#######################################
if __name__=="__main__":
	Q,retlog,betamatrix,bumpcountlog,visitmap=Qlearn_multirun_tab()
	mr=(np.mean(retlog,axis=0))
	csr=[]
	for i in range(len(mr)):
		if i>0:			
			csr.append(np.sum(mr[0:i])/i)
	np.savez("balanced_"+str(p.Nruns)+"_runs.npy.npz",retlog,bumpcountlog,betamatrix,Q,visitmap)
