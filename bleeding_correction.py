'''
A semi-parametric probabilistic blind deconvolution approach to spot bleed correction.
'''
import numpy as np

def imshow_matrix(reads, locations):
    to_plot = np.full(locations.max(axis=0).astype(int)+1, 0)
    to_plot[locations[:,0], locations[:,1]] = reads
    return to_plot

def build_basis_indices(locations):
    '''Creates 4 sets of basis functions: north, south, east, and west.
    Each basis is how far the 2nd element is from the first element.'''
    diffs = (locations[None] - locations[:,None]).astype(int)
    north = np.where(diffs[...,0] >= 0)
    south = np.where(diffs[...,0] < 0)
    east = np.where(diffs[...,1] >= 0)
    west = np.where(diffs[...,1] < 0)

    d_max = locations.max()+1
    basis_idxs = np.zeros((locations.shape[0], locations.shape[0], 4), dtype=int)
    basis_mask = np.zeros((locations.shape[0], locations.shape[0], 4), dtype=bool)

    for i in range(locations.shape[0]):
        # North
        basis_coord = 0
        for j in np.where(diffs[i,:,0] > 0):
            basis_idxs[i,j,basis_coord] = diffs[i,j,0]
            basis_mask[i,j,basis_coord] = True
        basis_coord += 1

        # South
        for j in np.where(diffs[i,:,0] < 0):
            basis_idxs[i,j,basis_coord] = -diffs[i,j,0]
            basis_mask[i,j,basis_coord] = True
        basis_coord += 1

        # East
        for j in np.where(diffs[i,:,1] > 0):
            basis_idxs[i,j,basis_coord] = diffs[i,j,1]
            basis_mask[i,j,basis_coord] = True
        basis_coord += 1

        # West
        for j in np.where(diffs[i,:,1] < 0):
            basis_idxs[i,j,basis_coord] = -diffs[i,j,1]
            basis_mask[i,j,basis_coord] = True
        basis_coord += 1

        # Treat the local spot specially
        basis_mask[i, i] = False
    
    return basis_idxs, basis_mask

def test_build_basic_indices():
    locations = np.array([
        [0,0],
        [1,2],
        [5,0],
        [0,3],
        [1,1]
        ])

    basis_idxs, basis_mask = build_basis_indices(locations)

    # for idx, loc in enumerate(locations):
    #     print(f'Location {loc}')
    #     print(basis_idxs[idx])
    #     print(basis_mask[idx])
    #     print()

def softplus(x):
    return np.log1p(np.exp(-np.abs(x))) + np.maximum(x, 0)

def fit_basis_functions(Reads, tissue_mask, Rates, global_rates, basis_idxs, basis_mask, lam=0, x_init=None):
    import torch
    from autograd_minimize import minimize
    from torch.distributions.poisson import Poisson
    from torch.distributions.multinomial import Multinomial
    from torch.nn import Softmax, Softplus
    from utils import stable_softmax

    local_weight = 100
    N = Reads.sum(axis=0)
    t_Y = torch.Tensor(Reads.T)
    t_Rates = torch.Tensor(Rates)
    t_Beta0 = torch.Tensor(global_rates)
    t_basis_idxs = torch.LongTensor(basis_idxs)
    t_basis_mask = torch.Tensor(basis_mask)
    t_local_mask = torch.Tensor(tissue_mask.astype(float))
    t_local_idxs = torch.LongTensor(np.arange(Reads.shape[0]))
    sm = Softmax(dim=0)
    sp = Softplus()

    # We have a set of basis functions with mappings from spots to locations in each basis
    basis_shape = (basis_idxs.shape[2], basis_idxs.max()+1)
    t_reverse = torch.LongTensor(np.arange(basis_shape[1])[::-1].copy())

    if x_init is None:
        x_init = np.full(basis_shape, -3)
        # x_init = (np.median(Reads, axis=0), )
        # x_init = np.concatenate([x_init[0], x_init[1].reshape(-1)])
        # print(x_init)
    

    def loss(t_Betas):
        t_Betas = sp(t_Betas)
        # t_Beta0, t_Betas = Betas[:Reads.shape[1]], Betas[Reads.shape[1]:]
        t_Betas = t_Betas.reshape(basis_shape)

        # Exponentiate and sum each basis element from N down to the current entry j for each j
        t_Basis = t_Betas[:,t_reverse].cumsum(dim=1)[:,t_reverse]

        # Add all the basis values for this spot
        W = torch.sum(torch.stack([t_Basis[d,t_basis_idxs[:,:,d]] * t_basis_mask[:,:,d] for d in range(basis_shape[0])], dim=0), dim=0)

        # Set the value of each local spot to 1
        W[t_local_idxs, t_local_idxs] += local_weight*t_local_mask

        # Normalize across target spots to get a probability
        t_Weights = sm(W)
        
        # Rate for each spot is bleed prob * spot rate plus the global read prob
        t_Mu = (t_Rates[None] * t_Weights[...,None]).sum(dim=1) + t_Beta0[None]
        
        # print(t_Basis[0,:15])
        # print(t_Basis[1,:15])
        # print(t_Basis[2,:15])
        # print(t_Basis[3,:15])
        
        # Calculate the negative log-likelihood of the data
        L = -torch.stack([Multinomial(total_count=int(N[i]), probs=t_Mu[:,i]).log_prob(t_Y[i]) for i in range(Reads.shape[1])], dim=0).mean()

        if lam > 0:
            # Apply a fused lasso penalty to enforce piecewise linear curves
            L += lam * (t_Betas[:,1:] - t_Betas[:,:-1]).abs().mean()

        # print('Before L2:', L)
        # Add a tiny bit of ridge penalty
        L += 1e-1*(t_Basis**2).sum()
        # print('After L2:', L)

        return L
    
    # Optimize using a 2nd order method with autograd for gradient calculation. Amazing times we live in.
    res = minimize(loss, x_init, method='L-BFGS-B', backend='torch')

    Betas = softplus(res.x)
    # Betas = res.x
    basis_functions = Betas.reshape(basis_shape)[:,::-1].cumsum(axis=1)[:,::-1]

    W = np.sum([basis_functions[d,basis_idxs[:,:,d]] * basis_mask[:,:,d] for d in range(basis_shape[0])], axis=0)
    W[np.arange(Reads.shape[0]), np.arange(Reads.shape[0])] += local_weight*tissue_mask.astype(float)
    Weights = stable_softmax(W, axis=0)

    return basis_functions, Weights, res

def test_fit_basis_functions(Reads, locations, tissue_mask):
    basis_idxs, basis_mask = build_basis_indices(locations)
    Rates = np.copy(Reads)*tissue_mask[:,None]*1.1
    beta0, basis_functions, Weights = fit_basis_functions(Reads, tissue_mask, Rates, basis_idxs, basis_mask)
    plot_basis_functions(basis_functions)

def plot_basis_functions(basis_functions):
    print('Plotting')
    basis_names = ['North', 'South', 'West', 'East']
    for d in range(basis_functions.shape[0]):
        plt.plot(np.arange(basis_functions.shape[1]), basis_functions[d], label=basis_names[d])
    plt.xlabel('Distance along cardinal direction')
    plt.ylabel('Relative bleed probability')
    plt.legend(loc='upper right')
    plt.savefig('plots/basis-functions.pdf', bbox_inches='tight')
    plt.close()

def plot_bleed_vectors(locations, tissue_mask, Rates, Weights):
    # Plot the general directionality of where reads come from in each spot
    for g in range(Rates.shape[1]):
        Contributions = (Rates[None,:,g] * Weights)
        Directions = locations[None] - locations[:,None]
        Vectors = (Directions * Contributions[...,None]).mean(axis=1)
        Vectors = Vectors / np.abs(Vectors).max(axis=0, keepdims=True) # Normalize everything to show relative bleed
        plt.imshow(imshow_matrix(tissue_mask, locations), cmap='viridis', vmin=-1)
        for i, ((y, x), (dy, dx)) in enumerate(zip(locations, Vectors)):
            plt.arrow(x,y,dx,dy, width=0.1*np.sqrt(dx**2+dy**2), head_width=0.2*np.sqrt(dx**2+dy**2), color='black', alpha=np.sqrt(dx**2+dy**2))
        plt.savefig(f'plots/zebrafish/bleed-vectors/{g}.pdf', bbox_inches='tight')
        plt.close()

def fit_spot_rates(Reads, tissue_mask, Weights, x_init=None):
    import torch
    from autograd_minimize import minimize
    from torch.nn import Softplus
    from torch.distributions.multinomial import Multinomial

    # Filter down the weights to only the nonzero rates
    Weights = Weights[:,tissue_mask]
    n_Rates = tissue_mask.sum()

    N = Reads.sum(axis=0)
    t_Y = torch.Tensor(Reads.T)
    # t_Beta0 = torch.Tensor(global_rates)
    t_Weights = torch.Tensor(Weights)
    sp = Softplus()
    
    if x_init is None:
        x_init = np.concatenate([np.median(Reads, axis=0), np.copy(Reads[tissue_mask].reshape(-1)*1.1).clip(1e-2,None)])

    def loss(t_Rates):
        t_Rates = sp(t_Rates)
        t_Beta0 = t_Rates[:Reads.shape[1]]
        t_Rates = t_Rates[Reads.shape[1]:]
        t_Rates = t_Rates.reshape(n_Rates, Reads.shape[1])
        Mu = (t_Rates[None] * t_Weights[...,None]).sum(dim=1) + t_Beta0[None]

        # Calculate the negative log-likelihood of the data
        L = -torch.stack([Multinomial(total_count=int(N[i]), probs=Mu[:,i]).log_prob(t_Y[i]) for i in range(Reads.shape[1])], dim=0).mean()

        # print(Mu.data.numpy()[tissue_mask][:10])
        # print(t_Beta0)
        # print(L)

        return L

    # Optimize using a 2nd order method with autograd for gradient calculation. Amazing times we live in.
    res = minimize(loss, x_init, method='L-BFGS-B', backend='torch')

    Rates = np.zeros(Reads.shape)
    global_rates = softplus(res.x[:Reads.shape[1]])
    Rates[tissue_mask] = softplus(res.x[Reads.shape[1]:]).reshape(n_Rates, Reads.shape[1])

    return global_rates, Rates, res

def decontaminate_spots(Reads, tissue_mask, basis_idxs, basis_mask, n_top=10, rel_tol=1e-4, max_steps=1):
    # Initialize the rates to be the local observed reads
    Rates = np.copy(Reads[:,:n_top]*tissue_mask[:,None]*1.1).clip(1e-2,None)
    global_rates = np.median(Reads[:,:n_top], axis=0)
    basis_init, Rates_init = None, None

    print(f'Fitting basis functions to first {n_top} genes')
    for step in range(max_steps):
        print(f'\nStep {step+1}/{max_steps}')

        basis_functions, Weights, res = fit_basis_functions(Reads[:,:n_top], tissue_mask, Rates, global_rates, basis_idxs, basis_mask, lam=0, x_init=basis_init)
        basis_init = res.x

        global_rates, Rates, res = fit_spot_rates(Reads[:,:n_top], tissue_mask, Weights, x_init=Rates_init)
        Rates_init = res.x
        loss = res.fun

        print(f'\tLoss: {loss:.2f}')

    Rates = np.zeros(Reads.shape)
    global_rates = np.zeros(Reads.shape[1])
    for g in range(Reads.shape[1]):
        print(f'\nGene {g+1}/{Reads.shape[1]}')
        global_rates[g], Rates[:,g:g+1], res = fit_spot_rates(Reads[:,g:g+1], tissue_mask, Weights, x_init=None)

    return global_rates, Rates, basis_functions, Weights

if __name__ == '__main__':
    import matplotlib
    import matplotlib.pyplot as plt

    np.random.seed(14)
    np.set_printoptions(suppress=True, precision=4)

    # Fixes a weird conflict between pytorch and matplotlib
    import os
    os.environ['KMP_DUPLICATE_LIB_OK']='True'


    #### Load real data
    base_path = 'data/zebrafish/'
    locations = np.load(f'{base_path}zebrafish_A1_locations.npy')
    tissue_mask = np.load(f'{base_path}zebrafish_A1_tissue_mask.npy')
    Reads = np.load(f'{base_path}zebrafish_A1_reads.npy')
    true_w = None

    #### Filter down to a subset of genes and spots, to speed things up when debugging
    # Filter down to every 3rd row and column
    # subsample_mask = np.all(locations % 2 == 0, axis=1)
    # locations, tissue_mask, Reads = locations[subsample_mask] // 2, tissue_mask[subsample_mask], Reads[subsample_mask]
    
    # Filter down to only a subset genes, most importantly BRAF (gene 58)
    # Reads = Reads[:,[58] + list(range(2))]

    #### Build the list of basis function indices
    basis_idxs, basis_mask = build_basis_indices(locations)

    #### Fit the model to the data
    n_genes = Reads.shape[1]
    global_rates, fit_Rates, basis_functions, Weights = decontaminate_spots(Reads, tissue_mask, basis_idxs, basis_mask)
    # fit_Counts = np.load(f'{base_path}zebrafish_A1_reads_fixed.npy')
    # basis_functions = np.load(f'{base_path}zebrafish_A1_bleed_basis.npy')
    # Weights = np.load(f'{base_path}zebrafish_A1_bleed_weights.npy')
    # fit_Rates = np.load(f'{base_path}zebrafish_A1_spot_rates.npy')
    # global_rates = np.load(f'{base_path}zebrafish_A1_global_rates.npy')

    #### Quickly estimate the counts as just rounded versions of the rates
    fit_Counts = np.round(fit_Rates)
    np.save(f'{base_path}zebrafish_A1_reads_fixed.npy', fit_Counts)
    np.save(f'{base_path}zebrafish_A1_bleed_basis.npy', basis_functions)
    np.save(f'{base_path}zebrafish_A1_bleed_weights.npy', Weights)
    np.save(f'{base_path}zebrafish_A1_spot_rates.npy', fit_Rates)
    np.save(f'{base_path}zebrafish_A1_global_rates.npy', global_rates)

    #### Plot the results
    import seaborn as sns
    with sns.axes_style('white'):
        plt.rc('font', weight='bold')
        plt.rc('grid', lw=3)
        plt.rc('lines', lw=1)
        plt.rc('xtick', labelsize=14)
        plt.rc('ytick', labelsize=14)
        matplotlib.rcParams['pdf.fonttype'] = 42
        matplotlib.rcParams['ps.fonttype'] = 42
        plot_basis_functions(basis_functions)
        plot_bleed_vectors(locations, tissue_mask, fit_Rates, Weights)
        n_plot_cols = 4
        # n_plots = min(n_genes, 20)
        n_plots = 1
        for i in range(n_genes):
            fig, axarr = plt.subplots(n_plots, n_plot_cols, figsize=(5*n_plot_cols,5*n_plots), sharex=True, sharey=True)
            ax = axarr[i] if n_plots > 1 else axarr
            col_idx = 0
            
            im = ax[col_idx].imshow(imshow_matrix(Reads[:,i], locations))
            plt.colorbar(im, ax=ax[col_idx])
            ax[col_idx].set_title('Bleed counts')
            col_idx += 1

            im = ax[col_idx].imshow(imshow_matrix(Reads[:,i], locations), vmax=15)
            plt.colorbar(im, ax=ax[col_idx])
            ax[col_idx].set_title('Bleed counts up to 15')
            col_idx += 1

            im = ax[col_idx].imshow(imshow_matrix(fit_Counts[:,i], locations))
            plt.colorbar(im, ax=ax[col_idx])
            ax[col_idx].set_title('Denoised counts')
            col_idx += 1

            im = ax[col_idx].imshow(imshow_matrix(fit_Counts[:,i], locations), vmax=15)
            plt.colorbar(im, ax=ax[col_idx])
            ax[col_idx].set_title('Denoised counts up to 15')
            col_idx += 1

            plt.tight_layout()
            plt.savefig(f'plots/zebrafish/debleed/{i}.pdf', bbox_inches='tight')
            plt.close()