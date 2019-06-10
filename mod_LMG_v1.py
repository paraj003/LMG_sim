# -*- coding: utf-8 -*-
## Module containing all definitions necessary to run a quench for the LMG model. Use python 3.
##v1.0 with new defined Hamiltonian with parameters γz and γy
import numpy as np
import os
import h5py
from scipy import linalg as LA
from scipy.special import binom as bm
from sympy.physics.quantum.cg import CG
import numpy.matlib

## Class definition to define Hamiltonian

##define Hamiltonian parameters
class Ham_params:
    def __init__(self, N:int,S:float,J:float,γz:float,γy:float,Γ:float):
        self.N=int(N) #number of spins, keep it even
        if S<=N/2:
            self.S=float(S) #spin sector
        else:
            raise Exception("Total spin S must be smaller than N/2")
        self.J=float(J) #Ising hopping
        self.γz=float(γz) #z-direction interaction  
        self.γy=float(γy) #y direction interaction
        self.Γ=float(Γ) #Transverse field
    def paramstr(self):
        #returns a string that contains the parameters of the Hamiltonian
        return'L_'+str(int(self.N))+',S_'+str(float(self.S))+',J_'+str(float(self.J))+',Γ_'+str(float(self.Γ))+',γz_'+str(float(self.γz))+',γy_'+str(float(self.γy))

    def paramstrwithoutLS(self):
        #returns a string that contains the parameters of the Hamiltonian
        return'J_'+str(float(self.J))+',Γ_'+str(float(self.Γ))+',γz_'+str(float(self.γz))+',γy_'+str(float(self.γy))


##function definitions
def LMG_matrixelement(X:Ham_params,M:float,Mprime:float):
    ##computes the matrix element <S,M|H|S,M'>
    value=0 
    if abs(M-Mprime)<10**-5:
        value= (X.J/2)*(X.γz+X.γy*(1-2*X.S*(X.S+1)/X.N))-(M**2)*X.J*(2*X.γz-X.γy)/X.N
    elif abs(M-(Mprime-2))<10**-5:
        value= X.J*X.γy/(2*X.N)*np.sqrt((X.S*(X.S+1)-(M+2)*(M+1))*(X.S*(X.S+1)-M*(M+1)))
    elif abs(M-(Mprime+2))<10**-5:
        value= X.J*X.γy/(2*X.N)*np.sqrt((X.S*(X.S+1)-(M-2)*(M-1))*(X.S*(X.S+1)-M*(M-1)))
    elif abs(M-(Mprime-1))<10**-5:
        value=-X.Γ*np.sqrt(X.S*(X.S+1)-M*(M+1))
    elif abs(M-(Mprime+1))<10**-5:
        value=-X.Γ*np.sqrt(X.S*(X.S+1)-M*(M-1))
    return value         
def LMG_generateHam(X:Ham_params):
    ##Generate (2*S+1,2*S+1) matrix.
    Ham=np.zeros((int(2*X.S+1),int(2*X.S+1)))
    Marr=np.linspace(-X.S,X.S,int(2*X.S+1))
    for p in range(np.size(Marr)):
        for q in range(np.size(Marr)):
            Ham[p,q]=LMG_matrixelement(X,Marr[p],Marr[q])
    return Ham
def magnetizationz2(state,X:Ham_params):
    ##takes in a column vector denoting wavefunction, and calcuates average magnetization along z direction as defined for ising spins (See lyx file for def)
    Marr=np.linspace(-X.S,X.S,int(2*X.S+1))
    magsq=4/X.N**2*np.sum(np.square(np.abs(state)*Marr))
    return magsq
def magnetizationϕ2(state,X:Ham_params,Az:complex,Ay:complex):
    ##calculates magnetization squared along 'ϕ'-direction: Sϕ=(Az*Sz+Ay*Sy) (see lyx file for def)
    Marr=np.linspace(-X.S,X.S,int(2*X.S+1))
    A_Marr=np.zeros(np.size(Marr),dtype=complex)
    if X.S==0:
        magsq=0
        return magsq
    else: 
        A_Marr[0]=(Az*X.S*state[0]-np.sqrt(2*X.S)*state[1]*Ay/(2*1j))
        A_Marr[-1]=(Az*X.S*state[-1]+np.sqrt(2*X.S)*state[-2]*Ay/(2*1j))
        for m in range(1,np.size(Marr)-1):
            A_Marr[m]=(Az*Marr[m]*state[m]+Ay/(2*1j)*(np.sqrt(X.S*(X.S+1)-Marr[m]*(Marr[m]-1))*state[m-1]-np.sqrt(X.S*(X.S+1)-Marr[m]*(Marr[m]+1))*state[m+1]))
        magsq=4/X.N**2*np.sum(np.square(np.abs(A_Marr)))
        return magsq 

def Sz2(state,X:Ham_params):
    ##takes in a column vector denoting wavefunction, and calcuates average Sz squared.
    Marr=np.linspace(-X.S,X.S,(2*X.S+1))
    Szsq=np.sum(np.square(np.abs(state)*Marr))
    return Szsq 

def Sϕ2(state,X:Ham_params,Az:complex,Ay:complex):
    ##takes in a column vector denoting wavefunction, and calcuates average <Sϕdag*Sϕ> Sϕ=Az*Sz+Ay*Sy.
    Marr=np.linspace(-X.S,X.S,(2*X.S+1))
    Sϕsq=X.N**2/4*magnetizationϕ2(state,X,Az,Ay)
    return Sϕsq 

def time_evolved_Sz2(InitState,Nsteps,U_dt,X:Ham_params):
    ##returns an array with the Sz^2 calculated at intervals of dt for Nsteps
    Sz2arr=np.zeros(Nsteps)
    ψ_t=np.copy(InitState)
    for p in np.arange(Nsteps):
        ##print(p) #print(p, end='\r', flush=True)
        ψ_t=np.dot(U_dt,ψ_t)
        Sz2arr[p]=Sz2(ψ_t,X)
    return Sz2arr

def time_evolved_Sϕ2(InitState,Nsteps,U_dt,X:Ham_params,Az:complex,Ay:complex):
    ##returns an array with the Sz^2 calculated at intervals of dt for Nsteps
    Sϕ2arr=np.zeros(Nsteps)
    ψ_t=np.copy(InitState)
    for p in np.arange(Nsteps):
        #print(p) #print(p, end='\r', flush=True)
        ψ_t=np.dot(U_dt,ψ_t)
        Sϕ2arr[p]=Sϕ2(ψ_t,X,Az,Ay)
    return Sϕ2arr

def time_evolved_Sϕ2_exact(InitState,tarr,energies,eigenvecs,X:Ham_params,Az:complex,Ay:complex):
    ##returns an array with the Sphi^2 calculated at times in tarr by computing exact unitary.
    Sϕ2arr=np.zeros(np.size(tarr))
    for t1,q in zip(tarr,range(np.size(tarr))):
        Sϕ2arr[q]=np.real(twotimecorrelation(X,[t1],[t1],InitState,energies,eigenvecs,Az,Ay)[0,0])
    return Sϕ2arr
###########ENTANGLEMENT ENTROPY##########################
def CGmatrix(SA,SB,S,directory):
    ##Define a ClebschGordan matrix using sympy library, returns [(2SA+1)(2SB+1)]X[(2S+1)]  matrix that changes for |S,M> basis to |SA,MA;SB,MB>
    ##directory='data/CGmats/'
    filename=directory+'CGmat_SA_'+str(float(SA))+'_SB_'+str(float(SB))+'_S_'+str(float(S))+'.hdf5'
    if (not os.path.exists(filename)) :
        print("Running CGmatrix_to_file")
        CGmatrix_to_file(SA,SB,S,directory)
    ##print("Loading CGmatrix: "+filename)
    with h5py.File(filename, "r") as f:
        cgmat_data= f["cgmat_data"][...]
    ##print(np.shape(cgmat_data))
    cgmat=np.zeros((int((2*SA+1)*(2*SB+1)),int(2*S+1)))
    for p in range(np.size(cgmat_data,0)):
        cgmat[int(cgmat_data[p,1]),int(cgmat_data[p,0])]=cgmat_data[p,2]
    return cgmat  
def CGmatrix_to_file(SA,SB,S,directory):
    ##Define a ClebschGordan matrix using sympy library, returns  an array of tuples, (p,q,CG(SA,MA[q],SB,MB[q],S,M[p])) 
    ##which can be converted into the desired matrix  matrix that changes for |S,M> basis to |SA,MA;SB,MB>
    if S > SA+SB:
        raise Exception('S should be less than SA+SB')
    ##directory='data/CGmats/'
    if not os.path.exists(directory):
        os.makedirs(directory)
    filename=directory+'CGmat_SA_'+str(float(SA))+'_SB_'+str(float(SB))+'_S_'+str(float(S))+'.hdf5'
    if not os.path.exists(filename):   
        cgmat_data=[]
        MAarr=np.matlib.repmat(np.linspace(-SA,SA,int(2*SA+1)),1,int(2*SB+1))[0,:]
        MBarr=np.reshape(np.matlib.repmat(np.linspace(-SB,SB,int(2*SB+1)),int(2*SA+1),1),(1,int((2*SA+1)*(2*SB+1))),order='F')[0,:]
        Marr=np.linspace(-S,S,int(2*S+1))
        for p in range(np.size(Marr)):
            Msumlist=np.where(MAarr+MBarr==Marr[p])[0]
            #print(p, end='\r', flush=True)
            for q in Msumlist:
                    cgmat_data.append([p,q,CG(SA,MAarr[q],SB,MBarr[q],S,Marr[p]).doit().evalf()])
        cgmat_datanp=np.array([cgmat_data_i for cgmat_data_i in cgmat_data])
        cgmat_datanpf=cgmat_datanp.astype(float)
        print("Saving to file: "+filename) 
        with h5py.File(filename, "w") as f:
            f.create_dataset("cgmat_data", cgmat_datanpf.shape, dtype=cgmat_datanpf.dtype, data=cgmat_datanpf)   
        
def Reduced_ρ(GStateAB,SA,SB):
    ##takes in state written in basis of subsystems A and B and traces out B
    GStateAB_matrix=np.reshape(GStateAB,(int(2*SB+1),int(2*SA+1)))
    ρA=np.zeros((int(2*SA+1),int(2*SA+1)),dtype=complex)
    for p in range(int(2*SA+1)):
        for q in range(int(2*SA+1)):
            ρA[p,q]=np.dot(GStateAB_matrix[:,p],np.conjugate(GStateAB_matrix[:,q]))
    return ρA
def EEntropy_VN(ρA):
    ρeigvals=LA.eigvals(ρA)
    return np.real(-np.dot(ρeigvals,np.log(ρeigvals)))

################Two-time correlation#####################
def LMG_Ut(t,energies,eigenvecs):
    ##given row vector of energies, and matrix of eigenvectors of Hamiltonian, it returns the unitary at a particular time
    return np.dot(np.dot(eigenvecs,LA.expm(-1j*np.diag(energies)*t)),np.transpose(np.conjugate(eigenvecs)))


def Sϕ_on_state(X:Ham_params,state,Az:complex,Ay:complex):
    ##returns a vector : Sϕ|Ψ>. See lyx file for derivation
    Marr=np.linspace(-X.S,X.S,int(2*X.S+1))
    B_Marr=np.zeros(np.size(Marr),dtype=complex)
    if X.S==0:
        return B_Marr
    else:
        B_Marr[0]=(Az*X.S*state[0]-np.sqrt(2*X.S)*state[1]*Ay/(2*1j))
        B_Marr[-1]=(Az*X.S*state[-1]+np.sqrt(2*X.S)*state[-2]*Ay/(2*1j))
        for m in range(1,np.size(Marr)-1):
            B_Marr[m]=(Az*Marr[m]*state[m]+Ay/(2*1j)*(np.sqrt(X.S*(X.S+1)-Marr[m]*(Marr[m]-1))*state[m-1]-np.sqrt(X.S*(X.S+1)-Marr[m]*(Marr[m]+1))*state[m+1]))
        return B_Marr


def Sϕt_on_state(X:Ham_params,state,U_t,Az:complex,Ay:complex):
    ##returns a vector : Sϕ(t)|Ψ>=Udag*Sϕ*U|Ψ>
    return np.dot(np.transpose(np.conjugate(U_t)),Sϕ_on_state(X,np.dot(U_t,state),Az,Ay))


def twotimecorrelation(X:Ham_params,t1arr,t2arr,state,energies,eigenvecs,Az:complex,Ay:complex):
    ##Calculate <Sϕ(t2)Sϕ(t1)> for different values of t1 and t2 and return an array
    ##construct unitary at time tt.
    correlationarr=np.zeros((np.size(t1arr),np.size(t2arr)),dtype=complex)
    for t1,q in zip(t1arr,range(np.size(t1arr))):
        U_t1=LMG_Ut(t1,energies,eigenvecs)
        Sϕt1=Sϕt_on_state(X,state,U_t1,Az,Ay)
        for t2,r in zip(t2arr,range(np.size(t2arr))):
            U_t2=LMG_Ut(t2,energies,eigenvecs)
            Sϕt2=Sϕt_on_state(X,state,U_t2,Az,Ay)
            correlationarr[q,r]=np.dot(np.transpose(np.conjugate(Sϕt2)),Sϕt1)
    return correlationarr

def twotimecommutator(X:Ham_params,t1:float,t2:float,state,energies,eigenvecs,Az:complex,Ay:complex):
    ##Returns an array :element 0 =<Sϕ(t2)Sϕ(t1)-Sϕ(t1)Sϕ(t2)>
    ##                  element 1= <Sϕ(t2)Sϕ(t1)+Sϕ(t1)Sϕ(t2)>
    ##construct unitary at time tt.
    U_t1=LMG_Ut(t1,energies,eigenvecs)
    U_t2=LMG_Ut(t2,energies,eigenvecs)
    Sϕt1=Sϕt_on_state(X,state,U_t1,Az,Ay)
    Sϕt2=Sϕt_on_state(X,state,U_t2,Az,Ay)
    commutatorarr=np.zeros(2,1,dtype=complex)
    commutatorarr[0]=np.dot(np.transpose(np.conjugate(Sϕt2)),Sϕt1)-np.dot(np.transpose(np.conjugate(Sϕt1)),Sϕt2)
    commutatorarr[1]=np.dot(np.transpose(np.conjugate(Sϕt2)),Sϕt1)+np.dot(np.transpose(np.conjugate(Sϕt1)),Sϕt2)
    return commutatorarr

##############FINITE TEMPERATURE FUNCTIONS#################
###########Only done for γy=0, OR  γz=0.
def finitetemp_criticalβ(X:Ham_params):
    if X.γy==0:
        return 1/(2*X.Γ)*(np.log((X.J*X.γz+X.Γ)/(X.J*X.γz-X.Γ)))
    elif X.γz==0:
        return 1/(2*X.Γ)*(np.log((X.J*X.γy+X.Γ)/(X.J*X.γy-X.Γ)))
    else:
        print("Can't calculate critical field in this parameter regime. Set either one of γy or γz to zero.")
        return

def Finitetempmagnetizationϕ2(X:Ham_params,β,Az:complex,Ay:complex):
    Sarr=np.arange(0,X.N/2+1)
    ##print(Sarr)
    expectvalarr=np.zeros(np.shape(Sarr))
    partitionfunctionarr=np.zeros(np.shape(Sarr))
    minenergies=np.zeros(np.shape(Sarr))
    for s in Sarr:
        ##print(s)
        paramvalsS=Ham_params(N=X.N,S=s,J=X.J,γz=X.γz,γy=X.γy,Γ=X.Γ)
        Ham=LMG_generateHam(paramvalsS)
        energies,eigenvecs=LA.eig(Ham)
        minenergies[int(s)]=np.min(np.real(energies))
        Mvals=np.zeros(np.shape(energies))
        probvals=np.zeros(np.shape(energies))
        shiftedenergies=np.real(energies)-minenergies[int(s)] #(to shift the zero of the energies)
        for p in range(np.size(energies)):
            Mvals[p]=magnetizationϕ2(eigenvecs[:,p],paramvalsS,Az,Ay)
            #print(shiftedenergies)
            probvals[p]=np.exp(-β*shiftedenergies[p])
        partitionfunctionarr[int(s)]=np.sum(probvals)
        expectvalarr[int(s)]=np.dot(probvals,Mvals)
    Ds=np.zeros(np.shape(Sarr))#multiplicities of each spin sector
    Ds[int(X.N/2)]=1
    for p in range(int(X.N/2)):
        Ds[p]=bm(X.N,int(X.N/2)-p)-bm(X.N,int(X.N/2)-p-1)
    ##print(Ds)
    minenergiesshifted=minenergies-np.min(minenergies)
    ##print(np.exp(-β*minenergiesshifted)*Ds)
    expectvalshifted=np.dot((np.exp(-β*minenergiesshifted)*Ds),expectvalarr)
    partitionfunctionshifted=np.dot((np.exp(-β*minenergiesshifted)*Ds),partitionfunctionarr)
    expectval=expectvalshifted/partitionfunctionshifted
    return expectval

def finitetemp_twotimecorrelation(X:Ham_params,t1arr,t2arr,β:float,Az:complex,Ay:complex):
    ##obtain the finitetemperature correlator <Sϕ(t2)Sϕ(t1)>_β
    Sarr=np.arange(0,X.N/2+1)
    expectvalSarr=np.zeros((np.size(t1arr),np.size(t2arr),np.size(Sarr)),dtype=complex)
    expectvalshifted=np.zeros((np.size(t1arr),np.size(t2arr)),dtype=complex)
    partitionfunctionarr=np.zeros(np.shape(Sarr))
    minenergies=np.zeros(np.shape(Sarr))
    for s in Sarr:
        ##print(s)
        paramvalsS=Ham_params(N=X.N,S=s,J=X.J,γz=X.γz,γy=X.γy,Γ=X.Γ)
        Ham=LMG_generateHam(paramvalsS)
        energies,eigenvecs=LA.eig(Ham)
        minenergies[int(s)]=np.min(np.real(energies))
        correlationvals=np.zeros((np.size(t1arr),np.size(t2arr),np.size(energies)),dtype=complex)
        probvals=np.zeros(np.shape(energies))
        shiftedenergies=np.real(energies)-minenergies[int(s)] #(to shift the zero of the energies)
        for p in range(np.size(energies)):
            correlationvals[:,:,p]=twotimecorrelation(paramvalsS,t1arr,t2arr,eigenvecs[:,p],energies,eigenvecs,Az,Ay)
            probvals[p]=np.exp(-β*shiftedenergies[p])
            partitionfunctionarr[int(s)]=np.sum(probvals)
        for t1,q in zip(t1arr,range(np.size(t1arr))):
            for t2,r in zip(t2arr,range(np.size(t2arr))):
                expectvalSarr[q,r,int(s)]=np.dot(probvals,correlationvals[q,r,:])
    Ds=np.zeros(np.shape(Sarr))#multiplicities of each spin sector
    Ds[int(X.N/2)]=1
    for p in range(int(X.N/2)):
        Ds[p]=bm(X.N,int(X.N/2)-p)-bm(X.N,int(X.N/2)-p-1)
    ##print(Ds)
    minenergiesshifted=minenergies-np.min(minenergies)
    ##print(np.exp(-β*minenergiesshifted)*Ds)
    partitionfunctionshifted=np.dot((np.exp(-β*minenergiesshifted)*Ds),partitionfunctionarr)
    for t1,q in zip(t1arr,range(np.size(t1arr))):
            for t2,r in zip(t2arr,range(np.size(t2arr))):
                expectvalshifted[q,r]=np.dot((np.exp(-β*minenergiesshifted)*Ds),expectvalSarr[q,r,:])
    expectvalarr=expectvalshifted/partitionfunctionshifted
    return expectvalarr
    


###############saving data###############################
def arrtostr(tarr):
    ##returns a string with the time array
    return '['+str(tarr[0])+'_'+str(np.divide((tarr[-1]-tarr[0]),(np.size(tarr)-1),out=np.zeros_like((tarr[-1]-tarr[0])), where=np.size(tarr)!=1))+'_'+str(tarr[-1])+']'
def save_data_Sz2t(paramvals0:Ham_params,paramvalsf:Ham_params,Sz2arr,initstate,Nsteps,dt):
    ## saves data in a h5py dictionary
    directory='data/Sz2t/'
    if not os.path.exists(directory):
        os.makedirs(directory)
    filename=directory+'Sz2t_[0_'+str(dt)+'_'+str(dt*Nsteps)+']_from_'+paramvals0.paramstr()+'_to_'+paramvalsf.paramstr()+'.hdf5'
    print(filename)
    with h5py.File(filename, "w") as f:
        f.create_dataset("Sz2arr", Sz2arr.shape, dtype=Sz2arr.dtype, data=Sz2arr)
        f.create_dataset("InitState", initstate.shape, dtype=initstate.dtype, data=initstate)
        f.close()
    with open("list_of_Sz2t.txt", "a") as myfile:
        myfile.write(filename+ "\n")
 
def save_data_Sϕ2t(paramvals0:Ham_params,paramvalsf:Ham_params,Sϕ2arr,Az,Ay,initstate,Nsteps,dt):
    ## saves data in a h5py dictionary
    directory='data/Sϕ2t/'
    if not os.path.exists(directory):
        os.makedirs(directory)
    filename=directory+'Sϕ2t_Az_'+str(float(Az))+'_Ay_'+str(float(Ay))+'_[0_'+str(dt)+'_'+str(dt*Nsteps)+']_from_'+paramvals0.paramstr()+'_to_'+paramvalsf.paramstr()+'.hdf5'
    print(filename)
    with h5py.File(filename, "w") as f:
        f.create_dataset("Sϕ2arr", Sϕ2arr.shape, dtype=Sϕ2arr.dtype, data=Sϕ2arr)
        f.create_dataset("InitState", initstate.shape, dtype=initstate.dtype, data=initstate)
        f.close()
    with open("list_of_Sϕ2t.txt", "a") as myfile:
        myfile.write(filename+ "\n")

def save_data_twotimecorrelation(paramvals0:Ham_params,paramvalsf:Ham_params,correlationarr,t1arr,t2arr,Az,Ay):
    ## saves data in a h5py dictionary
    directory='data/Twotimecorrelation/'
    if not os.path.exists(directory):
        os.makedirs(directory)
    filename=directory+'Twotimecorrelator_Az_'+str(float(Az))+'_Ay_'+str(float(Ay))+'_t1_'+arrtostr(t1arr)+'_t2_'+arrtostr(t2arr)+'_from_'+paramvals0.paramstr()+'_to_'+paramvalsf.paramstr()+'.hdf5'
    print(filename)
    with h5py.File(filename, "w") as f:
        f.create_dataset("correlationarr", correlationarr.shape, dtype=correlationarr.dtype, data=correlationarr)
        f.create_dataset("t1arr", t1arr.shape, dtype=t1arr.dtype, data=t1arr)
        f.create_dataset("t2arr", t2arr.shape, dtype=t2arr.dtype, data=t2arr)
        f.close()
    with open("list_of_twotimecorrelators.txt", "a") as myfile:
        myfile.write(filename+ "\n")

def save_data_finitetemp_twotimecorrelation(β,paramvals:Ham_params,correlationarr,t1arr,t2arr,Az,Ay):
    ## saves data in a h5py dictionary
    directory='data/FiniteTempTwotimecorrelation/'
    if not os.path.exists(directory):
        os.makedirs(directory)
    filename=directory+'FiniteTempTwotimecorrelator_β_'+str(β)+'_Az_'+str(float(Az))+'_Ay_'+str(float(Ay))+'_t1_'+arrtostr(t1arr)+'_t2_'+arrtostr(t2arr)+'_from_'+paramvals.paramstr()+'.hdf5'
    print(filename)
    with h5py.File(filename, "w") as f:
        f.create_dataset("correlationarr", correlationarr.shape, dtype=correlationarr.dtype, data=correlationarr)
        f.create_dataset("t1arr", t1arr.shape, dtype=t1arr.dtype, data=t1arr)
        f.create_dataset("t2arr", t2arr.shape, dtype=t2arr.dtype, data=t2arr)
        f.create_dataset("β", β.shape, dtype=β.dtype, data=β)
        f.close()
    with open("list_of_finitetemptwotimecorrelators.txt", "a") as myfile:
        myfile.write(filename+ "\n")


def save_data_EE(paramvals0:Ham_params,paramvalsf:Ham_params,entropyarr,tarr,initstate,La_arr):
    ## saves data in a h5py dictionary
    directory='data/EE/'
    if not os.path.exists(directory):
        os.makedirs(directory)
    filename=directory+'EE_LA_['+str(La_arr[0])+'_'+str(La_arr[-1])+']_t_'+arrtostr(tarr)+'_from_'+paramvals0.paramstr()+'_to_'+paramvalsf.paramstr()+'.hdf5'
    print(filename)
    with h5py.File(filename, "w") as f:
        f.create_dataset("entropyarr", entropyarr.shape, dtype=entropyarr.dtype, data=entropyarr)
        f.create_dataset("tarr", tarr.shape, dtype=tarr.dtype, data=tarr)
        f.create_dataset("InitState", initstate.shape, dtype=initstate.dtype, data=initstate)
        f.create_dataset("La_arr", La_arr.shape, dtype=La_arr.dtype, data=La_arr)
        f.close()
    with open("list_of_entropy.txt", "a") as myfile:
        myfile.write(filename+ "\n")
def save_data_Energies_Eigenvecs(paramvals:Ham_params,energies,eigenvecs,n_energies):
    # saves data in a h5py dictionary
    directory='data/Energies_Eigenvecs/'
    if not os.path.exists(directory):
        os.makedirs(directory)
    filename=directory+'Energies_Eigenvecs_'+paramvals.paramstr()+'n_energies_'+str(int(n_energies))+'.hdf5'
    print(filename)
    with h5py.File(filename, "w") as f:
        f.create_dataset("energies", energies.shape, dtype=energies.dtype, data=energies)
        f.create_dataset("eigenvecs", eigenvecs.shape, dtype=eigenvecs.dtype, data=eigenvecs)
        f.close()
    with open("list_of_ED_energies_eigenvecs.txt", "a") as myfile:
        myfile.write(filename+ "\n")

