#!/usr/bin/env python
# coding: utf-8

# In[3]:


import numpy as np
import matplotlib.pyplot as plt
import soundfile as sf
import sys


# In[3]:


def get_f_mean(deltat,y,basef,maxdf):
    spectrum=np.fft.rfft(y[:])#Calculamos el espectro de frecuencias de la señal
    
    frequencies = np.fft.rfftfreq(len(y),d=deltat) #Creamos el array con las frequencias correspondientes
    
    magnitude = np.abs(spectrum)#Transformamos los espectros en amplitudes presentes
    N=len(y)#Obtenemos los indices para hacer la mascara del espectro de frecuencias
    samplerate=1/deltat
    k1= (basef-maxdf)*N/samplerate
    k2= (basef+maxdf)*N/samplerate
    mascara=np.array([i>k1 and i< k2 for i in range(len(magnitude))])#Creamos la máscara
    magnitude = magnitude*mascara
    f_mean = np.sum(magnitude*frequencies)/np.sum(magnitude) #Obtenemos la media de las frecuencias ponderada por sus magnitudes
    
    return f_mean


# In[4]:


def get_f_harmonic_mean(deltat,y,basef,maxdf):
    spectrum=np.fft.rfft(y[:])[1:]#Calculamos el espectro de frecuencias de la señal
    
    frequencies = np.fft.rfftfreq(len(y),d=deltat)[1:] #Creamos el array con las frequencias correspondientes
    
    magnitude = np.abs(spectrum) #Transformamos los espectros en amplitudes presentes
    N=len(y)#Obtenemos los indices para hacer la mascara del espectro de frecuencias
    samplerate = 1/deltat
    k1= (basef-maxdf)*N/samplerate
    k2= (basef+maxdf)*N/samplerate
    mascara=np.array([i>k1 and i< k2 for i in range(len(magnitude))])#Creamos la máscara
    magnitude = magnitude*mascara
    f_harmonic_mean = np.sum(magnitude)/np.sum(magnitude/frequencies) #Obtenemos la media harmonica de las frecuencias ponderada por sus magnitudes
    
    return f_harmonic_mean  


# In[5]:


def get_f_max(deltat,y,basef,maxdf):
    
    spectrum= np.fft.rfft(y[:])

    frequencies = np.fft.rfftfreq(len(y),d=deltat)

    magnitude = np.abs(spectrum)
    N=len(y)#Obtenemos los indices para hacer la mascara del espectro de frecuencias
    samplerate=1/deltat
    k1= (basef-maxdf)*N/samplerate
    k2= (basef+maxdf)*N/samplerate
    mascara=np.array([i>k1 and i< k2 for i in range(len(magnitude))])#Creamos la máscara
    magnitude = mascara*magnitude #Aplicamos la mascara
    a= np.argmax(magnitude)
    f_max = frequencies[a]#Obtenemos la frecuencia con mayor amplitud
    return f_max
        


# In[6]:


#Loop principal
#t: array con el tiempo
#y: array con la amplitud
#windowsize:Tamaño de ventana(numero de samples)
#stepsize: tamaño del paso
#basef: frecuencia base que usaremos de centro
#maxdf: define el tamaño del intervalo para buscar frecuencias
def getfmeanvst(t,y,windowsize,stepsize,basef,maxdf,dt):
#Calculamos el numero de frames que podemos extraer
    num_frames = (len(y)-2*windowsize)
    #partimos de y[windowsize] y miramos desde y[i-windowsize] hasta y[i+windowsize]
    frecuencias=np.zeros([num_frames//stepsize,2])
    for i in range(0,num_frames//stepsize):
        frecuencias[i,1]= get_f_mean(dt,y[i*stepsize:i*stepsize+2*windowsize],basef,maxdf)
        frecuencias[i,0]=t[i*stepsize+windowsize]
    return frecuencias


# In[7]:


#Loop principal
#t: array con el tiempo
#y: array con la amplitud
#windowsize:Tamaño de ventana(numero de samples)
#stepsize: tamaño del paso
#basef: frecuencia base que usaremos de centro
#maxdf: define el tamaño del intervalo para buscar frecuencias
def getfharmonicvst(t,y,windowsize,stepsize,basef,maxdf,dt):
#Calculamos el numero de frames que podemos extraer
    num_frames = (len(y)-2*windowsize)
    #partimos de y[windowsize] y miramos desde y[i-windowsize] hasta y[i+windowsize]
    frecuencias=np.zeros([num_frames//stepsize,2])
    for i in range(0,num_frames//stepsize):
        frecuencias[i,1]= get_f_harmonic_mean(dt,y[i*stepsize:i*stepsize+2*windowsize],basef,maxdf)
        frecuencias[i,0]=t[i*stepsize+windowsize]
    return frecuencias


# In[8]:


#Loop principal
#t: array con el tiempo
#y: array con la amplitud
#windowsize:Tamaño de ventana(numero de samples)
#stepsize: tamaño del paso
#basef: frecuencia base que usaremos de centro
#maxdf: define el tamaño del intervalo para buscar frecuencias
def getfmaxvst(t,y,windowsize,stepsize,basef,maxdf,dt):
#Calculamos el numero de frames que podemos extraer
    num_frames = (len(y)-2*windowsize)
    #partimos de y[windowsize] y miramos desde y[i-windowsize] hasta y[i+windowsize]
    frecuencias=np.zeros([num_frames//stepsize,2])
    for i in range(0,num_frames//stepsize):
        frecuencias[i,1]= get_f_max(dt,y[i*stepsize:i*stepsize+2*windowsize],basef,maxdf)
        frecuencias[i,0]=t[i*stepsize+windowsize]
    return frecuencias


# #main
# for file in files:
#     data , frequency = read(file)
#     data = data.clean
#     N = f(dt) # si queremos despejar a N(windowsize) como funcion de dt partimos de la relacion
#     dt= N/f_s, es decir N=dt*f_s
#     
#    df_bin = f_s/N = 1/(N*T_s)=1/dt
#    si queremos encontrar f vs t con un error temporal <= 0.05 ms esto nos da un df_bin de 20Hz 
#    por que df_bin * dt >=1
#    Esto es lo que yo considero optimo para nuestro problema dado que
#    la frecuencia oscila en ese rango

# In[9]:


def limpiarzeros(y):
    indice = np.flatnonzero(y)[0] #devuelve el indice con el primer elemento no nulo
    return y[indice:]


# In[32]:


def plot_fvst(x,y,T_s,windowsize,stepsize,basef,maxdf,df,dt,mymax=True,mymean=True,myharmonic=True,mylabel="Placeholder",filename="Placeholder"):
    if mymax:
        fmaxvst= getfmaxvst(x,y,windowsize,stepsize,basef,maxdf,T_s)
        plt.errorbar(x=fmaxvst[:,0],#Tiempos
            y=fmaxvst[:,1],#Amplitudes
            xerr=dt,yerr=df #Barras de errores
            ,fmt="bo",capsize=3,label=f"{mylabel}(max)")
       # print("fmax shape",fmaxvst.shape)
    if mymean:
        fmeanvst = getfmeanvst(x,y,windowsize,stepsize,basef,maxdf,T_s)
        plt.errorbar(x=fmeanvst[:,0],#Tiempos
                    y=fmeanvst[:,1]#Amplitudes
                    ,xerr = dt, yerr=df 
                    ,fmt="ro",capsize=3,label=f"{mylabel}(mean)")
        #print("fmean shape",fmeanvst.shape)
    if myharmonic:
        fharmonicvst = getfharmonicvst(x,y,windowsize,stepsize,basef,maxdf,T_s)
        plt.errorbar(x=fharmonicvst[:,0],#Tiempos
                   y=fharmonicvst[:,1],#Amplitudes
                   xerr=dt,yerr=df
                    ,fmt="go",capsize=3,label=f"{mylabel}(harmonic)")
        #print("fharmonic shape", fharmonicvst.shape)
    plt.legend()
    plt.xlabel("Tiempo($s$)")
    plt.ylabel("Frecuencia medida(Hz)")
    plt.savefig(f"{filename}-fvst.pdf")
    plt.close()

# In[1]:


def ftov(fobs,femit,vson):
    return vson*(fobs/femit -1)


# In[2]:


def dftodv(dfobs,femit,vson):
    return vson*dfobs/femit


# In[ ]:


def plot_v_vs_t(x,y,T_s,vson,windowsize,stepsize,basef,maxdf,df,dt,mymax=True,mymean=True,myharmonic=True,mylabel="Placeholder",filename="Placeholder"):
    if mymax:
        fmaxvst= getfmaxvst(x,y,windowsize,stepsize,basef,maxdf,T_s)
        plt.errorbar(x=fmaxvst[:,0],#Tiempos
            y=ftov(fmaxvst[:,1],basef,vson),#Velocidades
            xerr=dt,yerr=dftodv(df,basef,vson) #Barras de errores
            ,fmt="bo",capsize=3,label=f"{mylabel}(max)")
       # print("fmax shape",fmaxvst.shape)
    if mymean:
        fmeanvst = getfmeanvst(x,y,windowsize,stepsize,basef,maxdf,T_s)
        plt.errorbar(x=fmeanvst[:,0],#Tiempos
                    y=ftov(fmeanvst[:,1],basef,vson)#Velocidades
                    ,xerr = dt, yerr=dftodv(df,basef,vson) 
                    ,fmt="ro",capsize=3,label=f"{mylabel}(mean)")
        #print("fmean shape",fmeanvst.shape)
    if myharmonic:
        fharmonicvst = getfharmonicvst(x,y,windowsize,stepsize,basef,maxdf,T_s)
        plt.errorbar(x=fharmonicvst[:,0],#Tiempos
                   y=ftov(fharmonicvst[:,1],basef,vson),#Velocidades
                   xerr=dt,yerr=dftodv(df,basef,vson) #Barras de errores
                    ,fmt="go",capsize=3,label=f"{mylabel}(harmonic)")
        #print("fharmonic shape", fharmonicvst.shape)
    plt.legend()
    plt.xlabel("Tiempo($s$)")
    plt.ylabel(r"Velocidad medida ($\frac{m}{s}$)")
    plt.savefig(f"{filename}-vvst.pdf")
    plt.close()

# In[35]:


files=sys.argv[1:]


# In[33]:


def main(files,dt,vson,thislabelfrecuencia,thislabelvelocidad):
    for file in files:
        data, samplerate = sf.read(file)
        T_s = 1/samplerate #Periodo de muestreo
        data = limpiarzeros(data)
        tiempo = np.arange(len(data))*T_s
        WSZ = int(dt//T_s) #El tamaño de ventana expresado en terminos de la incertidumbre
        STSZ = WSZ #Elegimos stesize igual al tamaño de ventana
        df_bin = 1/dt
        plot_fvst(tiempo,data,T_s,WSZ,STSZ,8000,200,df_bin,dt,mylabel=thislabelfrecuencia,filename=file)
        plot_v_vs_t(tiempo,data,T_s,vson,WSZ,STSZ,8000,200,df_bin,dt,mylabel=thislabelvelocidad,filename=file)

# In[34]:


main(files,0.25,343,"frecuencia medida","velocidad medida")

