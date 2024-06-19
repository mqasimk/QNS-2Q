# -*- coding: utf-8 -*-
"""
Created on Tue Aug  2 12:04:51 2022

@author: mqasi
"""

# -*- coding: utf-8 -*-
"""
Created on Mon Aug  1 12:19:21 2022

@author: mqasi
"""

import numpy as np
from qutip import *
from matplotlib import pyplot as plt
import csv
import concurrent.futures
from itertools import repeat
from multiprocessing import freeze_support
from joblib import Parallel, delayed

def S_w(w, num):
    tc=0.5/(1*10**6)
    # if num == 0:
    S0 = 0.25*10**3
    #w0=6*10**3
    # if num == 1:
    #     S0 = 1
    w0=4*10**6#0
    #if w<=2*np.pi/(16*3.1*10**(-4)):
    return(S0*(1/(1+(tc**2)*(np.abs(w)-w0)**2)))
    # return(0.25*np.heaviside(w+w0,1)*np.heaviside(-w+w0,1))

def gen_noise_mat(t0, tf, dt, wmax, num, delta):
    dw = wmax/1000
    t_vec_max = np.arange(t_0, tf, dt)
    size_w = int(2*wmax/dw)
    size_t = np.size(t_vec_max)
    S=np.zeros((size_t,size_w))
    C=np.zeros((size_t,size_w))
    for i in range(size_t):
        for j in range(size_w):
            S[i,j]=np.sqrt(dw*S_w(j*dw, num)/np.pi)*np.sin(j*dw*(t_vec_max[i]+delta))
            C[i,j]=np.sqrt(dw*S_w(j*dw, num)/np.pi)*np.cos(j*dw*(t_vec_max[i]+delta))
    return S, C

def dyn_solver(omega, num_shots, t_vec, S, C, size_w, d, a0, b0, ensemble_size):
    mu = 0
    sig = 1
    nxup = np.zeros(ensemble_size)
    num_shots = ensemble_size
    ketup_y = (1j*basis(2, 1) + basis(2, 0)).unit()
    # p=0
    for j in range(num_shots):
        p=0
        A=np.random.normal(mu, sig, (size_w, 1))
        B=np.random.normal(mu, sig, (size_w, 1))
        f_t_0 = np.dot(S[0], A) + np.dot(C[0], B)
        b_t_0 = np.ndarray.flatten(f_t_0[0:np.size(t_vec)])
        #inter_0 = interpolate.Cubic_Spline(t_vec[0], t_vec[-1], b_t_0)
        #A=np.random.normal(mu, sig, (size_w, 1))
        #B=np.random.normal(mu, sig, (size_w, 1))
        f_t_1 = np.dot(S[1], A) + np.dot(C[1], B)
        b_t_1 = np.ndarray.flatten(f_t_1[0:np.size(t_vec)])
        #inter_1 = interpolate.Cubic_Spline(t_vec[0], t_vec[-1], b_t_1)
        # plt.plot(t_vec, inter(t_vec))
        H0 = (omega/2)*tensor(sigmax(), identity(2), identity(2)) #+ tensor(sigmaz(), identity(2), sigmaz())*1.26e6
        H = [H0, [tensor(sigmaz(), sigmax(), identity(2)), b_t_0], [tensor(sigmaz(), sigmay(), identity(2)), b_t_1], [tensor(identity(2), sigmaz(), sigmaz()), b_t_0]]
        ketup = (basis(2, 1) + basis(2, 0)).unit()
        ketdown = (basis(2, 1) - basis(2, 0)).unit()
        rho0_plus = (1-d) * ketup * ketup.dag() + (d) * ketdown * ketdown.dag()
        rhob = 0.5*basis(2, 0) * basis(2, 0).dag() + 0.5*ketup_y * ketup_y.dag() #ketup * ketup.dag()
        rho0_plus = tensor(rho0_plus, rhob, rho0_plus)
        sol = mesolve(H, rho0_plus, t_vec, options={"nsteps": 100})
        rho_T_plus_10 = sol.states[-1].ptrace(0)[0,1]
        # p += (1/2)*(np.abs(a0)**2 - np.abs(b0)**2 + 1) + (np.abs(a0)**2 + np.abs(b0)**2 - 1)*np.real(rho_T_plus_10)
        p = (1/2)*(np.abs(a0)**2 - np.abs(b0)**2 + 1) + (np.abs(a0)**2 + np.abs(b0)**2 - 1)*np.real(rho_T_plus_10)
        if p > 1.:
            p = 1.
        if p < 0.:
            p = 0.
        nxup[j] = np.random.binomial(1, p)
    px_up = np.sum(nxup)/ensemble_size
    # p=p/num_shots
    # if p > 1.:
    #     p = 1.
    # if p < 0.:
    #     p = 0.
    # px_up = np.random.binomial(ensemble_size, p)/ensemble_size
    px_dn = 1. - px_up
    sigma_x_hat_plus_t = (px_up - px_dn)
    nxup = np.zeros(ensemble_size)
    # p=0
    for j in range(num_shots):
        p=0
        A=np.random.normal(mu, sig, (size_w, 1))
        B=np.random.normal(mu, sig, (size_w, 1))
        f_t_0 = np.dot(S[0], A) + np.dot(C[0], B)
        b_t_0 = np.ndarray.flatten(f_t_0[0:np.size(t_vec)])
        #inter_0 = interpolate.Cubic_Spline(t_vec[0], t_vec[-1], b_t_0)
        #A=np.random.normal(mu, sig, (size_w, 1))
        #B=np.random.normal(mu, sig, (size_w, 1))
        f_t_1 = np.dot(S[1], A) + np.dot(C[1], B)
        b_t_1 = np.ndarray.flatten(f_t_1[0:np.size(t_vec)])
        #inter_1 = interpolate.Cubic_Spline(t_vec[0], t_vec[-1], b_t_1)
        # plt.plot(t_vec, inter(t_vec))
        H0 = (omega/2)*tensor(sigmax(), identity(2), identity(2)) #+ tensor(sigmaz(), identity(2), sigmaz())*1.26e6
        H = [H0, [tensor(sigmaz(), sigmax(), identity(2)), b_t_0], [tensor(sigmaz(), sigmay(), identity(2)), b_t_1], [tensor(identity(2), sigmaz(), sigmaz()), b_t_0]]
        ketup = (basis(2, 1) + basis(2, 0)).unit()
        ketdown = (basis(2, 1) - basis(2, 0)).unit()
        rho0_minus = (d) * ketup * ketup.dag() + (1-d) * ketdown * ketdown.dag()
        rhob = 0.5*basis(2, 0) * basis(2, 0).dag() + 0.5*ketup_y * ketup_y.dag()#ketup * ketup.dag()
        rho0_plus = (1-d) * ketup * ketup.dag() + (d) * ketdown * ketdown.dag()
        rho0_minus = tensor(rho0_minus, rhob, rho0_plus)
        sol = mesolve(H, rho0_minus, t_vec, options={"nsteps": 100})
        rho_T_minus_10 = sol.states[-1].ptrace(0)[0,1]
        # p += (1/2)*(np.abs(a0)**2 - np.abs(b0)**2 + 1) + (np.abs(a0)**2 + np.abs(b0)**2 - 1)*np.real(rho_T_minus_10)
        p = (1/2)*(np.abs(a0)**2 - np.abs(b0)**2 + 1) + (np.abs(a0)**2 + np.abs(b0)**2 - 1)*np.real(rho_T_minus_10)
        if p > 1.:
            p = 1.
        if p < 0.:
            p = 0.
        nxup[j]=np.random.binomial(1, p)
    px_up = np.sum(nxup)/ensemble_size
    # p=p/num_shots
    # if p > 1.:
    #     p = 1.
    # if p < 0.:
    #     p = 0.
    # px_up = np.random.binomial(ensemble_size, p)/ensemble_size
    px_dn = 1. - px_up
    sigma_x_hat_minus_t = (px_up - px_dn)
    nyup = np.zeros(ensemble_size)
    # p=0
    for j in range(num_shots):
        p=0
        A=np.random.normal(mu, sig, (size_w, 1))
        B=np.random.normal(mu, sig, (size_w, 1))
        f_t_0 = np.dot(S[0], A) + np.dot(C[0], B)
        b_t_0 = np.ndarray.flatten(f_t_0[0:np.size(t_vec)])
        #inter_0 = interpolate.Cubic_Spline(t_vec[0], t_vec[-1], b_t_0)
        #A=np.random.normal(mu, sig, (size_w, 1))
        #B=np.random.normal(mu, sig, (size_w, 1))
        f_t_1 = np.dot(S[1], A) + np.dot(C[1], B)
        b_t_1 = np.ndarray.flatten(f_t_1[0:np.size(t_vec)])
        #inter_1 = interpolate.Cubic_Spline(t_vec[0], t_vec[-1], b_t_1)
        # plt.plot(t_vec, inter(t_vec))
        H0 = (omega/2)*tensor(sigmax(), identity(2), identity(2)) #+ tensor(sigmaz(), identity(2), sigmaz())*1.26e6
        H = [H0, [tensor(sigmaz(), sigmax(), identity(2)), b_t_0], [tensor(sigmaz(), sigmay(), identity(2)), b_t_1], [tensor(identity(2), sigmaz(), sigmaz()), b_t_0]]
        ketup = (basis(2, 1) + basis(2, 0)).unit()
        ketdown = (basis(2, 1) - basis(2, 0)).unit()
        rho0_minus = (1/2-d) * ketup * ketup.dag() + (1/2+d) * ketdown * ketdown.dag()
        rhob = 0.5*basis(2, 0) * basis(2, 0).dag() + 0.5*ketup_y * ketup_y.dag()#ketup * ketup.dag()
        rho0_plus = (1-d) * ketup * ketup.dag() + (d) * ketdown * ketdown.dag()
        rho0_minus = tensor(rho0_minus, rhob, rho0_plus)
        sol = mesolve(H, rho0_minus, t_vec, options={"nsteps": 100})
        rho_T_minus_10 = sol.states[-1].ptrace(0)[0,1]
        # p += (1/2)*(np.abs(a0)**2 - np.abs(b0)**2 + 1) + (np.abs(a0)**2 + np.abs(b0)**2 - 1)*np.real(rho_T_minus_10)
        p = (1/2)*(np.abs(a0)**2 - np.abs(b0)**2 + 1) + (np.abs(a0)**2 + np.abs(b0)**2 - 1)*np.real(rho_T_minus_10)
        if p > 1.:
            p = 1.
        if p < 0.:
            p = 0.
        nyup[j]=np.random.binomial(1, p)
    py_up = np.sum(nyup)/ensemble_size
    # p=p/num_shots
    # if p > 1.:
    #     p = 1.
    # if p < 0.:
    #     p = 0.
    # py_up=np.random.binomial(ensemble_size, p)/ensemble_size
    py_dn = 1. - py_up
    sigma_y_hat_plus_t = (py_up - py_dn)
    return sigma_x_hat_plus_t, sigma_x_hat_minus_t, sigma_y_hat_plus_t

def sigma_x_t(omega, a0, b0, d, Mmin, Mmax, t_0, T, dt, wmax, S, C, ensemble_size):
    #print(omega)
    S_glist = np.zeros(Mmax-Mmin+1)
    sigma_x_hat_plus = np.zeros(Mmax-Mmin+1)
    sigma_x_hat_minus = np.zeros(Mmax-Mmin+1)
    a1 = np.sqrt(1-np.conj(a0)*a0)
    dw = wmax/1000
    size_w = int(2*wmax/dw)
    num_shots = ensemble_size
    #for i in np.linspace(Mmin, Mmax, Mmax-Mmin+1, dtype=int):
        #t_vec = np.array([x for x in np.arange(t_0, i*T, dt)])
        #sigma_x_hat_plus[i-Mmin], sigma_x_hat_minus[i-Mmin] = dyn_solver(omega, num_shots, t_vec, S, C, size_w, d, a0, b0, ensemble_size)
    results = Parallel(n_jobs = 1, verbose=0)(delayed(dyn_solver)(omega, num_shots, np.arange(t_0, i*T, dt), S, C, size_w, d, a0, b0, ensemble_size) for i in np.linspace(Mmin, Mmax, Mmax-Mmin+1, dtype=int))
    results = np.array(results)
    sigma_x_hat_plus = results[:,0]
    sigma_x_hat_minus = results[:,1]
    sigma_y_hat_plus = results[:,2]
    return sigma_x_hat_plus, sigma_x_hat_minus, sigma_y_hat_plus

def lin_reg(sigma_x_hat_plus, sigma_x_hat_minus, sigma_y_hat_plus, t_vec):
    S_p_t_1 = np.log(2/np.abs(sigma_x_hat_plus - sigma_x_hat_minus))
    x_axis = t_vec
    coef_1 = np.polyfit(x_axis, S_p_t_1, 1)
    S_p_t_2 = np.log(1/np.abs(sigma_x_hat_plus - sigma_y_hat_plus))
    coef_2 = np.polyfit(x_axis, S_p_t_2, 1)
    alpha = np.exp(-coef_1[1])
    alpha_m = np.exp(-coef_2[1])
    alpha_sp = alpha/alpha_m
    S_m_t = ((sigma_x_hat_plus + sigma_x_hat_minus)/2)*coef_1[0]*x_axis/(1-np.exp(-coef_1[0]*x_axis))
    S_m = np.polyfit(x_axis, S_m_t, 1)
    asym = S_m[1]
    S_m_spam = ((sigma_x_hat_plus + sigma_x_hat_minus)/2)*coef_1[0]/(1-np.exp(-coef_1[0]*x_axis))
    return coef_1[0], S_p_t_1[-1]/t_vec[-1], coef_1[1], S_m[0]/alpha_m, S_m_spam[-1], asym, alpha_m, alpha_sp

def lin_reg1(sigma_x_hat_plus, sigma_x_hat_minus, sigma_y_hat_plus, ensemble_size, t_vec):
    delta_x_plus = np.sqrt((1-sigma_x_hat_plus**2)/(2*ensemble_size))+0.0001
    delta_x_minus = np.sqrt((1-sigma_x_hat_minus**2)/(2*ensemble_size))+0.0001
    delta_y_plus = np.sqrt((1-sigma_y_hat_plus**2)/(2*ensemble_size))+0.0001
    S_p_t_1 = np.log(2/np.abs(sigma_x_hat_plus - sigma_x_hat_minus))
    delta_Spt_1 = np.sqrt((1/np.abs(sigma_x_hat_plus - sigma_x_hat_minus)**2)*(delta_x_plus**2+delta_x_minus**2))
    x_axis = t_vec
    coef_1 = np.polyfit(x_axis, S_p_t_1, 1, w=1/delta_Spt_1, cov = 'unscaled')
    S_p_t_2 = np.log(1/np.abs(sigma_x_hat_plus - sigma_y_hat_plus))
    delta_Spt_2 = np.sqrt((1/np.abs(sigma_x_hat_plus - sigma_y_hat_plus)**2)*(delta_x_plus**2+delta_y_plus**2))
    coef_2 = np.polyfit(x_axis, S_p_t_2, 1, w=1/delta_Spt_2, cov = 'unscaled')
    S_m_t = (sigma_x_hat_plus + sigma_x_hat_minus)/2
    delta_Smt = 0.5*np.sqrt(delta_x_plus**2+delta_x_minus**2)
    coef_3 = np.polyfit(x_axis, S_m_t, 1, w = 1/delta_Smt, cov = 'unscaled')
    S_m_spam = ((sigma_x_hat_plus + sigma_x_hat_minus)/2)*coef_1[0][0]/(1-np.exp(-coef_1[0][0]*x_axis))
    
    Sp = coef_1[0][0]
    delta_Sp = np.sqrt(coef_1[1][0,0])
    
    Sph = S_p_t_1[-1]/t_vec[-1]
    delta_Sph = delta_Spt_1[-1]/t_vec[-1]
    
    alpha = np.exp(-coef_1[0][1])
    delta_alpha = alpha*np.sqrt(coef_1[1][1,1])
    
    alpha_m = np.exp(-coef_2[0][1])
    delta_alpha_m = alpha_m*np.sqrt(coef_2[1][1,1])
    
    alpha_sp = alpha/alpha_m
    delta_alpha_sp = np.sqrt((1/alpha_m**2)*(delta_alpha)**2+(alpha**2/alpha_m**4)*(delta_alpha_m)**2)
    
    am_Sm = coef_3[0][0]
    delta_am_Sm = np.sqrt(coef_3[1][0,0])
    
    asym = coef_3[0][1]
    delta_asym = np.sqrt(coef_3[1][1,1])
    
    Sm = am_Sm/alpha_m
    delta_Sm = np.sqrt((1/alpha_m**2)*(delta_am_Sm)**2+(am_Sm**2/alpha_m**4)*(delta_alpha_m)**2)
    
    Smh = S_m_t[-1]/t_vec[-1]
    delta_Smh = delta_Smt[-1]/t_vec[-1]
    
    #return coef_1[0][0], S_p_t_1[-1]/t_vec[-1], coef_1[1], S_m[0]/alpha_m, S_m_spam[-1], asym, alpha_m, alpha_sp
    return np.array([[Sp, delta_Sp], [Sph, delta_Sph], [Sm, delta_Sm], [Smh, delta_Smh], [alpha, delta_alpha], [alpha_m, delta_alpha_m], [alpha_sp, delta_alpha_sp], [asym, delta_asym]])

## Simulation parameters
#%%
mc=9
dt=10**(-7)
T=16*3.1*10**(-7)
t_0 = 0.
ensemble_size = 1000
Mmin = 1
Mmax = 20
wmax=2*np.pi*mc/T
delta = T/10
omega_array = np.array([2*h*np.pi/T for h in np.linspace(-mc, mc, 2*mc+1)])
S_0, C_0 = gen_noise_mat(t_0, Mmax*T, dt, wmax, 0, 0)
S_1, C_1 = gen_noise_mat(t_0, Mmax*T, dt, wmax, 0, delta)
#%%
## Exp Parameters

a0 = np.sqrt(1)
b0 = np.sqrt(1)
d = 0

#omega_array = np.array([2*h*np.pi/T for h in np.linspace(-mc, mc, 2*mc+1)])
sxup_O = np.zeros((np.size(omega_array), Mmax-Mmin+1))
sxdn_O = np.zeros((np.size(omega_array), Mmax-Mmin+1))
sx_y_O = np.zeros((np.size(omega_array), Mmax-Mmin+1))
print("Starting sim...")
results = Parallel(n_jobs = 24, verbose=10)(delayed(sigma_x_t)(omega, a0, b0, d, Mmin, Mmax, t_0, T, dt, wmax, [S_0, S_1], [C_0, C_1], ensemble_size) for omega in omega_array)
#sxup_O[i], sxdn_O[i] = sigma_x_t(omega_array[i], a0, b0, d, Mmin, Mmax, t_0, T, dt, wmax, [S_0, S_1], [C_0, C_1], ensemble_size)
print("Simulation Complete")
#%% Parse results
results=np.array(results)
t_vec = np.linspace(Mmin, Mmax, Mmax-Mmin+1)*T
sxup_O = np.array(results[:,0])
sxdn_O = np.array(results[:,1])
sx_y_O = np.array(results[:,2])
np.save("sxup_O.npy", sxup_O)
np.save("sxdn_O.npy", sxdn_O)
np.save("sx_y_O.npy", sx_y_O)
S_g = np.zeros(np.size(omega_array))
S_g_spam = np.zeros(np.size(omega_array))
err_S_g = np.zeros(np.size(omega_array))
err_S_g_spam = np.zeros(np.size(omega_array))

spam = np.zeros(np.size(omega_array))
S_m = np.zeros(np.size(omega_array))
S_m_spam = np.zeros(np.size(omega_array))
asym = np.zeros(np.size(omega_array))
alpha_m = np.zeros(np.size(omega_array))
alpha_sp = np.zeros(np.size(omega_array))
err_spam = np.zeros(np.size(omega_array))
err_S_m = np.zeros(np.size(omega_array))
err_S_m_spam = np.zeros(np.size(omega_array))
err_asym = np.zeros(np.size(omega_array))
err_alpha_m = np.zeros(np.size(omega_array))
err_alpha_sp = np.zeros(np.size(omega_array))


for i in range(np.size(omega_array)):
    out = lin_reg1(sxup_O[i], sxdn_O[i], sx_y_O[i], ensemble_size, t_vec)
    S_g[i]=out[0,0]
    S_g_spam[i]=out[1,0]
    S_m[i]=out[2,0]
    S_m_spam[i]=out[3,0]
    spam[i]=out[4,0]
    alpha_m[i]=out[5,0]
    alpha_sp[i]=out[6,0]
    asym[i]=out[7,0]
    err_S_g[i]=out[0,1]
    err_S_g_spam[i]=out[1,1]
    err_S_m[i]=out[2,1]
    err_S_m_spam[i]=out[3,1]
    err_spam[i]=out[4,1]
    err_alpha_m[i]=out[5,1]
    err_alpha_sp[i]=out[6,1]
    err_asym[i]=out[7,1]
    
    #S_g[i], S_g_spam[i], spam[i], S_m[i], S_m_spam[i], asym[i], alpha_m[i], alpha_sp[i] = lin_reg(sxup_O[i], sxdn_O[i], syup_O[i], t_vec)
#%%
out=np.array([[S_g, err_S_g], [S_g_spam, err_S_g_spam], [S_m, err_S_m], [S_m_spam, err_S_m_spam], [asym, err_asym], [spam, err_spam], [alpha_m, err_alpha_m], [alpha_sp, err_alpha_sp]])
np.save('omega_arr.npy', omega_array)
np.save('fitparams.npy',out)
#%% Plot the classical spectra
w_axis = omega_array
plt.errorbar(w_axis, S_g, yerr = err_S_g, fmt='gx', ls='-')
plt.plot(np.linspace(-wmax,wmax,1000), 4*(S_w(np.linspace(-wmax,wmax,1000), 0)))
plt.errorbar(w_axis, S_g_spam, yerr=err_S_g_spam, fmt='rx', ls = '--')
plt.title(r'Classical Spectrum')
plt.legend([r'SR QNS',r'$S^+(\omega)$', 'Std QNS'])
plt.xlabel(r'$\omega$')
plt.ylabel(r'$S^+(\omega)$')
plt.show()
plt.savefig('cspectra.jpg')
plt.close()
#plt.show()
#%% Plot the Quantum spectra
plt.plot(np.linspace(-wmax,wmax,1000), 4*(np.sin(np.linspace(-wmax,wmax,1000)*delta))*(S_w(np.linspace(-wmax,wmax,1000), 1)))
plt.errorbar(w_axis, S_m_spam, yerr=err_S_m_spam, fmt='rx', ls='--')
plt.errorbar(w_axis, S_m, yerr=err_S_m, fmt='gx', ls='-')
plt.title(r'Quantum Spectrum')
plt.xlabel(r'$\omega$')
plt.ylabel(r'$S^-(\omega)$')
plt.legend([r'$S^-(\omega)$', r'Std QNS', r'SR QNS'])
plt.show()
plt.savefig('qspectra.jpg')
plt.close()
#plt.plot(np.linspace(-wmax,wmax,1000), 0.5*(S_w(np.linspace(-wmax,wmax,1000), 0)+S_w(np.linspace(-wmax,wmax,1000), 1)))
#plt.show()

#%%
plt.plot(np.linspace(-wmax,wmax,1000), 4*(1+np.sin(np.linspace(-wmax,wmax,1000)*delta))*(S_w(np.linspace(-wmax,wmax,1000), 1)))
plt.errorbar(w_axis, S_m_spam+S_g_spam, yerr=0.5*np.sqrt(err_S_m_spam**2+err_S_g_spam**2), fmt='rx', ls='--')
plt.errorbar(w_axis, S_m+S_g, yerr=0.5*np.sqrt(err_S_m**2+err_S_g**2),fmt='gx-',ls='-')
plt.title('Complete Spectrum')
plt.xlabel(r'$\omega$')
plt.ylabel(r'$S(\omega)$')
plt.legend([r'$S(\omega)$', r'Std QNS', r'SR QNS'])
plt.show()
plt.savefig('spectra.jpg')
plt.close()
#plt.show()
  #%%
#alpha = np.exp(-spam)
# plt.plot(omega_array, (a0**2 + b0**2 - 1)*np.ones(np.size(omega_array))*(1-2*d))
# plt.errorbar(omega_array, spam, yerr = err_spam, fmt='gx')
# plt.ylim(0.5, 1.1)
# plt.title('Combined SPAM Parameter')
# plt.ylabel(r'$\alpha(\omega)$')
# plt.xlabel(r'$\omega$')
# plt.legend([r'$\alpha(\omega)$', r'$\hat\alpha(\omega)$'])
# plt.show()
#
# plt.plot(omega_array, (a0**2+b0**2-1)*np.ones(np.size(omega_array)), 'g')
# plt.errorbar(omega_array, alpha_m, yerr=err_alpha_m, fmt='gx')
# plt.ylim(0.5,1.1)
# plt.title('Measurement error Parameter')
# plt.xlabel(r'$\omega$')
# plt.ylabel(r'$\alpha_M(\omega)$')
# plt.legend([r'$\alpha_M(\omega)$', r'$\hat \alpha_M(\omega)$'])
# plt.show()
#
# aest = (alpha_m+asym+1)/2
# best = (alpha_m-asym+1)/2
# plt.plot(omega_array, (a0**2)*np.ones(np.size(omega_array)), 'g')
# plt.errorbar(omega_array, aest, yerr=0.5*np.sqrt(err_alpha_m**2+err_alpha_sp**2), fmt='gx')
# plt.plot(omega_array, (b0**2)*np.ones(np.size(omega_array)), 'r')
# plt.errorbar(omega_array, best, yerr=0.5*np.sqrt(err_alpha_m**2+err_alpha_sp**2), fmt='rx')
# plt.ylim(0.5,1.1)
# plt.title('Measurement error Parameters')
# plt.xlabel(r'$\omega$')
# plt.legend([r'$a_0(\omega)$', r'$\hat a_0(\omega)$', r'$b_0(\omega)$', r'$\hat b_0(\omega)$'])
# plt.show()
#
# plt.plot(omega_array, (1-2*d)*np.ones(np.size(omega_array)))
# plt.errorbar(omega_array, alpha_sp, yerr=err_alpha_sp, fmt='gx')
# plt.ylim(0.5,1.1)
# plt.title('State-preparation error parameter')
# plt.ylabel(r'$\alpha_{SP}(\omega)$')
# plt.xlabel(r'$\omega$')
# plt.legend([r'$\alpha_{SP}(\omega)$', r'$\hat\alpha_{SP}(\omega)$'])
# plt.show()
#
# plt.plot(omega_array, (a0**2-b0**2)*np.ones(np.size(omega_array)))
# plt.errorbar(omega_array, asym, yerr=err_asym, fmt='gx')
# plt.ylim(-0.1,0.1)
# plt.title('Asymmetry parameter')
# plt.ylabel(r'$\delta(\omega)$')
# plt.xlabel(r'$\omega$')
# plt.legend([r'$\delta(\omega)$', r'$\hat\delta(\omega)$'])
# plt.show()
# #%%
# print(np.exp(-np.polyfit(t_vec,np.log(1/np.abs(sxup_O[1]-sxdn_O[1])),1)[0]))
# #%%
# plt.plot(np.linspace(-wmax,wmax,1000), 4*S_w(np.linspace(-wmax,wmax,1000),1)*(np.sin(np.linspace(-wmax,wmax,1000)*delta)*0+1))
# plt.show()