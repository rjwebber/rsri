# -*- coding: utf-8 -*-
"""
Created on Fri Jul 21 11:24:11 2023

@author: robertwebber
"""

# =====
# SETUP
# =====

# import libraries
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import scipy.sparse

# pretty plots
font = {'family' : 'sans-serif',
        'weight' : 'regular',
        'size'   : 18}
plt.rc('font', **font)
plt.rcParams['axes.linewidth'] = 1.5

#%%

# ====
# RSRI
# ====

def pivotal(v, m):

    nonzeros = v.nonzero()[0]
    N_reduced = nonzeros.size
    result = np.copy(v)
    if m < N_reduced:
        reduced = v[nonzeros]

        # exact preservation
        mag = np.abs(reduced)
        keep = np.zeros(N_reduced, dtype=bool)
        for i in range(m):
            threshold = np.sum(mag[~keep]) / (m - np.sum(keep))
            new = (mag > threshold * (1 - 1e-8))
            if np.sum(new) > np.sum(keep):
                keep = new
            else:
                break
    
        if threshold > 0:
            # random sampling
            amplify = np.zeros(N_reduced, dtype=bool)
            probs = mag / threshold
            probs[keep] = 0
            index1 = probs.nonzero()[0][0]; index2 = index1 + 1
            p = probs[index1]
            for i in range(m - np.sum(keep) - 1):
                # first pass
                p_new = p + probs[index2]
                if p_new < 1:
                    index3 = index2 + 1
                    p_new += probs[index3]
                    while p_new < 1:
                        index3 += 1
                        p_new += probs[index3]
                    # optional second pass
                    u = np.random.uniform() * (p_new - probs[index3])
                    if p < u:
                        index1 = index2
                        p += probs[index1]
                    while p < u:
                        index1 += 1
                        p += probs[index1]
                    index2 = index3
                # transition
                if np.random.uniform() < (1 - probs[index2]) / (2 - p_new):
                    amplify[index1] = True
                    index1 = index2
                else:
                    amplify[index2] = True
                p = p_new - 1
                index2 += 1
            norm = p + np.sum(probs[index2:])
            u = np.random.uniform() * norm
            if u > p:
                index1 = index2
                p += probs[index1]
            while u > p:
                index1 += 1
                p += probs[index1]
            amplify[index1] = True
    
            # assemble new vector
            reduced[~(keep | amplify)] = 0
            reduced[amplify] /= probs[amplify]
        result[nonzeros] = reduced
    
    return result

#%%

# ====================
# LOAD NOTRE DAME DATA
# ====================

alpha = .85

# Notre Dame network
data = pd.read_csv('web-NotreDame.txt', header=3, sep='\t')
data = np.array(data)
counts = np.bincount(data[:, 0])
# np.random.seed(43)
# i = np.random.choice(np.where(counts > 0)[0])
i = 26692
stubs = np.where(counts == 0)[0]
data_new = np.column_stack((stubs, np.repeat(i, stubs.size)))
data_new = np.row_stack((data, data_new))
counts = np.bincount(data_new[:, 0])
vals = 1/counts[data_new[:, 0]]
N = data.max() + 1
matrix = scipy.sparse.csr_array((vals, (data_new[:, 1], data_new[:, 0])), shape=(N, N))
b = np.zeros(N)
b[i] = 1 - alpha

#%%

# =======================
# PROCESS NOTRE DAME DATA
# =======================

# apply richardson iteration
x = np.copy(b)
for i in range(int(50/(1 - alpha))):
    x_new = alpha * matrix.dot(x) + b
    print('Iteration ', i, ' residual: ', np.sum(x_new - x))
    x = x_new
x_ref = np.copy(x)
y = np.sort(x)[::-1]
z = np.cumsum(y[::-1])[::-1]

# apply RSRI
attempts = 10
t = 1000
sample_sizes = np.logspace(0, np.log10(4e5), 40).astype(int)
sample_sizes = np.unique(sample_sizes)
attempts = np.repeat(attempts, sample_sizes.size)
results = np.zeros(sample_sizes.size)
for j in range(sample_sizes.size):
    m = sample_sizes[j]
    trials = np.zeros(attempts[j])
    for k in range(attempts[j]):
        print(k)
        soln = np.zeros((t, N))
        x_hat = np.copy(b)
        for i in range(t):
            if i % 100 == 0:
                print(i)
            sparse_x = pivotal(x_hat, m)
            x_hat = alpha * matrix.dot(sparse_x) + b
            soln[i, :] = x_hat
        trials[k] = np.sum((x_ref - np.mean(soln[(t // 2):, :], axis = 0)) ** 2)
    results[j] = np.mean(trials) ** .5
    print('Sample size: ', m, '\nError: ', results[j])

np.save('notre_dame_decay.npy', z)
np.save('notre_dame_samples.npy', sample_sizes)
np.save('notre_dame_results.npy', results)

#%%

# =====================
# LOAD OPENFLIGHTS DATA
# =====================

alpha = .85

# Open flights network
data = pd.read_csv('openflights.txt', header=None, sep=' ')
data = np.array(data)
data[:, :2] = np.unique(data[:, :2], return_inverse=True)[1].reshape(data[:, :2].shape)
counts = np.bincount(data[:, 0], weights=data[:, 2])
# np.random.seed(44)
# i = np.random.choice(np.where(counts > 0)[0])
i = 1820
stubs = np.where(counts == 0)[0]
data_new = np.column_stack((stubs, np.repeat(i, stubs.size), np.repeat(1, stubs.size)))
data_new = np.row_stack((data, data_new))
counts = np.bincount(data_new[:, 0], weights=data_new[:, 2])
vals = data_new[:, 2]/counts[data_new[:, 0]]
N = data_new.max() + 1
matrix = scipy.sparse.csr_array((vals, (data_new[:, 1], data_new[:, 0])), shape=(N, N))
b = np.zeros(N)
b[i] = 1 - alpha

#%%

# ========================
# PROCESS OPENFLIGHTS DATA
# ========================

# apply richardson iteration
x = np.copy(b)
for i in range(int(50/(1 - alpha))):
    x_new = alpha * matrix.dot(x) + b
    print('Iteration ', i, ' residual: ', np.sum(x_new - x))
    x = x_new
x_ref = np.copy(x)
y = np.sort(x)[::-1]
z = np.cumsum(y[::-1])[::-1]

# apply RSRI
attempts = 10
t = 1000
sample_sizes = np.logspace(0, np.log10(4e5), 40).astype(int)
sample_sizes = np.unique(sample_sizes)
attempts = np.repeat(attempts, sample_sizes.size)
results = np.zeros(sample_sizes.size)
for j in range(sample_sizes.size):
    m = sample_sizes[j]
    trials = np.zeros(attempts[j])
    for k in range(attempts[j]):
        print(k)
        soln = np.zeros((t, N))
        x_hat = np.copy(b)
        for i in range(t):
            if i % 100 == 0:
                print(i)
            sparse_x = pivotal(x_hat, m)
            x_hat = alpha * matrix.dot(sparse_x) + b
            soln[i, :] = x_hat
        trials[k] = np.sum((x_ref - np.mean(soln[(t // 2):, :], axis = 0)) ** 2)
    results[j] = np.mean(trials) ** .5
    print('Sample size: ', m, '\nError: ', results[j])

np.save('openflights_decay.npy', z)
np.save('openflights_samples.npy', sample_sizes)
np.save('openflights_results.npy', results)

#%%

# ================
# LOAD AMAZON DATA
# ================

alpha = .85

# Amazon electronics network
data = pd.read_csv('ratings_Electronics.csv', header=None)
data = np.array(data)
data = data[data[:, 2] == 5, :2]
data[:, 0] = np.unique(data[:, 0], return_inverse=True)[1]
data[:, 1] = np.unique(data[:, 1], return_inverse=True)[1]
data = data.astype(int)
matrix = scipy.sparse.csc_array((np.ones(data.shape[0]), (data[:, 0], data[:, 1])), 
                                shape=(np.max(data[:, 0]) + 1, np.max(data[:, 1]) + 1))
matrix = matrix.T @ matrix
matrix.setdiag(0)
matrix.eliminate_zeros()
N = matrix.shape[0]
norms = matrix.dot(np.ones(N))
norms[norms == 0] = 1
matrix = matrix / norms
matrix = matrix.tocsr()
b = np.zeros(N)
# np.random.seed(42)
# i = np.random.choice(N)
i = 121958
b[i] = 1 - alpha

#%%

# ===================
# PROCESS AMAZON DATA
# ===================

# apply richardson iteration
x = np.copy(b)
for i in range(int(50/(1 - alpha))):
    x_new = alpha * matrix.dot(x) + b
    print('Iteration ', i, ' residual: ', np.sum(x_new - x))
    x = x_new
x_ref = np.copy(x)
y = np.sort(x)[::-1]
z = np.cumsum(y[::-1])[::-1]

# apply RSRI
attempts = 10
t = 1000
sample_sizes = np.logspace(0, np.log10(4e5), 40).astype(int)
sample_sizes = np.unique(sample_sizes)
attempts = np.repeat(attempts, sample_sizes.size)
results = np.zeros(sample_sizes.size)
for j in range(sample_sizes.size):
    m = sample_sizes[j]
    trials = np.zeros(attempts[j])
    for k in range(attempts[j]):
        print(k)
        soln = np.zeros((t, N))
        x_hat = np.copy(b)
        for i in range(t):
            if i % 100 == 0:
                print(i)
            sparse_x = pivotal(x_hat, m)
            x_hat = alpha * matrix.dot(sparse_x) + b
            soln[i, :] = x_hat
        trials[k] = np.sum((x_ref - np.mean(soln[(t // 2):, :], axis = 0)) ** 2)
    results[j] = np.mean(trials) ** .5
    print('Sample size: ', m, '\nError: ', results[j])

np.save('amazon_decay.npy', z)
np.save('amazon_samples.npy', sample_sizes)
np.save('amazon_results.npy', results)

#%%

# ==================================
# COMPARE TO DETERMINISTIC ITERATION
# ==================================

# apply richardson iteration
x = np.copy(b)
for i in range(int(50/(1 - alpha))):
    x_new = alpha * matrix.dot(x) + b
    print('Iteration ', i, ' residual: ', np.sum(x_new - x))
    x = x_new
x_ref = np.copy(x)
y = np.sort(x)[::-1]
z = np.cumsum(y[::-1])[::-1]

# apply deterministic iteration
r = np.copy(b)
x = np.zeros(b.size)
steps = int(1e6)
resid_sum = np.zeros(steps)
resid_error = np.zeros(steps)
error = np.zeros(steps)
for i in range(steps):
    print('Step ', i)
    j = r.argmax()
    val = r[j]
    x[j] += val
    r[j] = 0
    r += alpha * val * matrix[:, [j]].toarray()[:, 0]
    resid_sum[i] = r.sum()
    resid_error[i] = r.max()
    error[i] = np.sum((x - x_ref) ** 2.) ** .5
    print('Residual sum: ', resid_sum[i])
    print('Residual error: ', resid_error[i])
    print('Error: ', error[i])

np.save('error.npy', error)
np.save('resid_error.npy', resid_error)
np.save('resid_sum.npy', resid_sum)


#%%

# ===================
# PLOT FIG. 1 RESULTS
# ===================

# plot decay
error = np.load('error.npy')
resid_error = np.load('resid_error.npy')
resid_sum = np.load('resid_sum.npy')
colors = np.array(['#648FFF', '#785EF0', '#DC267F', '#FE6100', '#FFB000'])
fig, ax = plt.subplots(figsize=(6, 4.5), dpi=200)
ax.loglog(1/np.arange(error.size), '-', color='black', label='$t^{-1}$ scaling')
ax.loglog(error, '-.', color=colors[3], label='Solution error\n$|| \hat{x} - x ||$')
ax.loglog(resid_error, ':', color=colors[1], label='Residual error\n$|| r(\hat{x}) ||_{\infty}$')
ax.set_xlim([1, 1e6])
ax.set_ylim([1e-8, 1])
ax.tick_params(width=1.5, which='both')
ax.set_xlabel('Steps $t$', font)
ax.set_ylabel('Error level', font)
fig.tight_layout()
leg = ax.legend(loc='center left', bbox_to_anchor=(1, 0.5))
leg.get_frame().set_alpha(None)
leg.get_frame().set_facecolor((0, 0, 0, 0))
leg.get_frame().set_linewidth(1.5)
leg.get_frame().set_edgecolor('black')
fig.savefig('slow_deterministic.pdf', bbox_inches='tight')

#%%

# ===================
# PLOT FIG. 2 RESULTS
# ===================

# plot decay
amazon = np.load('amazon_decay.npy')
notre_dame = np.load('notre_dame_decay.npy')
airlines = np.load('openflights_decay.npy')
colors = np.array(['#648FFF', '#785EF0', '#DC267F', '#FE6100', '#FFB000'])
fig, ax = plt.subplots(1, 2, figsize=(10, 4.5), dpi=200)
ax[0].loglog(amazon, ':', color=colors[2], label='Amazon\nelectronics')
ax[0].loglog(notre_dame, '--', color=colors[1], label='Notre Dame\nwebsites')
ax[0].loglog(airlines, '-.', color=colors[0], label='Airports')
ax[0].set_xlim([1, 5e5])
ax[0].set_ylim([1e-10, 1])
ax[0].tick_params(width=1.5, which='both')
ax[0].set_xlabel('Sparsity level $m$', font)
ax[0].set_ylabel('PageRank tail decay', font)

# plot results
amazon_x = np.load('amazon_samples.npy')
amazon_y = np.load('amazon_results.npy')
notre_dame_x = np.load('notre_dame_samples.npy')
notre_dame_y = np.load('notre_dame_results.npy')
airlines_x = np.load('openflights_samples.npy')
airlines_y = np.load('openflights_results.npy')
#fig, ax = plt.subplots(figsize = (6, 4.5), dpi=200, sharey=True)
ax[1].loglog(amazon_x, amazon_x ** -.5 / np.sqrt(1000), '-', color='black', 
          label='$m^{-1/2}$ scaling')
ax[1].loglog(amazon_x, amazon_y, ':', color=colors[2], label='Amazon\nelectronics')
ax[1].loglog(notre_dame_x, notre_dame_y, '--', color=colors[1], label='Notre Dame\nwebsites')
ax[1].loglog(airlines_x, airlines_y, '-.', color=colors[0], label='Flights')
ax[1].set_xlabel('Sparsity level $m$', font)
ax[1].set_ylabel('RSRI error', font)
ax[1].set_xlim([1, 5e5])
ax[1].set_ylim([1e-10, 1])
ax[1].tick_params(width=1.5, which='both')
fig.tight_layout()
leg = ax[1].legend(loc='center left', bbox_to_anchor=(1, 0.5))
leg.get_frame().set_alpha(None)
leg.get_frame().set_facecolor((0, 0, 0, 0))
leg.get_frame().set_linewidth(1.5)
leg.get_frame().set_edgecolor('black')
fig.savefig('error.pdf', bbox_inches='tight')