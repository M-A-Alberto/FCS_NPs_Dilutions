# -*- coding: utf-8 -*-
"""
Created on Fri Apr 12 07:17:10 2024

@author: Alberto
"""


import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
import pandas as pd
import os
from scipy.stats import ttest_rel,ttest_ind, shapiro, f_oneway, tukey_hsd,chisquare
from iapws import IAPWS97 
from scipy.optimize import curve_fit

from sklearn.metrics import r2_score 


os.chdir(r"XXXXX") #Directory containing folders media (Water,DMEM)>Repetition (1-6)>Dilution (1_100,1_1000,1_2000)>Data file

sns.set_theme()
sns.set_style("ticks")
sns.set_context("paper",font_scale=1.5)
sns.set_palette("tab10")


def diffusion(tau,g0,td):
    

    y = g0/((1+(tau/td))*(1+(w**2/z0**2)*(tau/td))**(1/2))
    
    return y

#Define constants

kb = 1.38e-23

T = 273.15+36

mu = IAPWS97(T=T, x=0).mu



medias = ["H2Od","DMEM"]
D_values = pd.DataFrame()


k = 0
for media in medias:
    
    for rep in os.listdir(f"{media}"):
        for concentration in os.listdir(f"{media}/{rep}"):
            
            C = int(concentration.split(" ")[-1].replace("_",":")[2:])
            
            C = 1.2e15/C
            
            
            
            sizes = []
            
            if rep in ["4","5","6"]:
                
                    
                w = 0.23252
                z0 = 2.18211
                V_eff = np.pi**(3/2)*z0*1e-6*(w*1e-6)**2
            else:
                
                w =  0.23961
                
                z0 = 3.96377
                V_eff = np.pi**(3/2)*z0*1e-6*(w*1e-6)**2
                
                
            for file in os.listdir(f"{media}/{rep}/{concentration}"):
                
                #Read data
                data = pd.read_csv(f"{media}/{rep}/{concentration}/{file}",skiprows=1,sep="\t",skipinitialspace=True)
                
                for col in data.columns:
                    if "Correlation Channel" in col:
                        
                        #Get measurement's curve and fit tu free diffusion model
                        G_exp = data.loc[:,col].dropna()
                        
                        t = data.loc[:len(G_exp)-1,"Time [ms]"]
                        
                        #plt.plot(t,G_exp,"b")
                        
                        pars,cov = curve_fit(diffusion,t,G_exp)
                        
                        (g0,tau_d) = pars
                        
                                                       
                        D_pure = w**2/(4*tau_d*1e-3)
                        
                        G_fit = diffusion(t,*pars)
                        
                        chi_2 = r2_score(G_exp,G_fit)
                        
                        d = 2*kb*T/(6*mu*np.pi*D_pure)*1e12*1e9
                        
                        if chi_2>0.99:
                            
                            #plt.plot(t,G_exp,"b-")
                            #plt.plot(t,G_fit,"r-")
                            
                            D_values.loc[k,"Media"] = media
                            D_values.loc[k,"Concentration (NPs / ml)"] = C
                            D_values.loc[k,"Repetition"] = int(rep)
                            D_values.loc[k,"D ($\mu$m$^2$/s)"] = D_pure
                            D_values.loc[k,"Size (nm)"] = d
                            D_values.loc[k,"G$_0$"] = g0
                            D_values.loc[k,"Measured C (NPs/ml)"] = 1/(g0*V_eff*1e3)
                            
                            k+=1
                            
            #plt.title(f"{media} {C} {rep}")
            #plt.xscale("log")
            #plt.xlabel("Time (ms)")
            #plt.ylabel(r"G ($\tau$)")
            
            #plt.show()
            #plt.close()
                       

#Plot different measurements, calculate statistics and save data
means = D_values.groupby(by=["Media","Concentration (NPs / ml)","Repetition"],as_index=False).mean()

means = means.sort_values(by=["Concentration (NPs / ml)"],ascending=True,ignore_index=True)

for i,val in enumerate(means.loc[:,"Concentration (NPs / ml)"]):
    
    means.loc[i,"Concentration (NPs / ml)"] = f"{val:.2e}"

sns.barplot(data = means,x = "Concentration (NPs / ml)",y = "D ($\mu$m$^2$/s)",hue="Media",errorbar="se",capsize=0.1,err_kws={"linewidth": 1.25,"color":"k"},saturation=1)
plt.savefig("Figures/Diffusion Coefficient.tif",dpi=300,bbox_inches="tight")
plt.show()
plt.close()



sns.barplot(data = means,x = "Concentration (NPs / ml)",y = "Size (nm)",hue="Media",errorbar="se",palette = "Dark2",capsize=0.1,err_kws={"linewidth": 1.25,"color":"k"},saturation=1)
plt.savefig("Figures/Size.tif",dpi=300,bbox_inches="tight")
plt.show()
plt.close()


sns.barplot(data = means,x = "Concentration (NPs / ml)",y = "G$_0$",hue="Media",palette = "Set2",errorbar="se",capsize=0.1,err_kws={"linewidth": 1.25,"color":"k"},saturation=1)


plt.savefig("Figures/g0.tif",dpi=300,bbox_inches="tight")
plt.show()
plt.close()


sns.barplot(data = means,x = "Concentration (NPs / ml)",y = "Measured C (NPs/ml)",hue="Media",errorbar="se",capsize=0.1,err_kws={"linewidth": 1.25,"color":"k"},saturation=1)


plt.savefig("Figures/Measured C.tif",dpi=300,bbox_inches="tight")
plt.show()
plt.close()


means2 = means.groupby(by=["Media","Concentration (NPs / ml)"],as_index=False).mean()

print(means2)

stats = []
names = []
for media in set(means.Media):
    for c in set(means.loc[:,"Concentration (NPs / ml)"]):

        
        
        D = means.loc[(means.loc[:,"Media"]==media) & (means.loc[:,"Concentration (NPs / ml)"] == c),"D ($\mu$m$^2$/s)"]
        
        stats.append(list(D))
        
        name = f"{media} {c}"
        
        names.append(name)
        
        


tukey_stats = tukey_hsd(*stats)
print(tukey_stats)


stds2 = means.groupby(by=["Media","Concentration (NPs / ml)"],as_index=False).sem()





results = pd.DataFrame()

results.loc[:,"Media"] = means2.loc[:,"Media"]

results.loc[:,"Concentration (NPs / ml)"] = means2.loc[:,"Concentration (NPs / ml)"]

results.loc[:,"D (um^2/s)"] = means2.loc[:,"D ($\mu$m$^2$/s)"]

results.loc[:,"D_err (um^2/s)"] = stds2.loc[:,"D ($\mu$m$^2$/s)"]

results.loc[:,"Size (nm)"] = means2.loc[:,"Size (nm)"]

results.loc[:,"Size_err (nm)"] = stds2.loc[:,"Size (nm)"]

results.loc[:,"G$_0$"] = means2.loc[:,"G$_0$"]

results.loc[:,"G$_0$ err"] = stds2.loc[:,"G$_0$"]

print(results)

results.to_csv("Dilutions results.txt",sep="\t",decimal=",")