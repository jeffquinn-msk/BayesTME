'''
Same as bleed_correction8.py except now we use separate basis functions for in-tissue and out-out-tissue.
The hope is that this enables us to account for tissue friction which seems to be an issue.
'''
import numpy as np

def generate_data(n_rows=30, n_cols=30, n_genes=20, spot_bleed_prob=0.5, bleeding='anisotropic'):
    from scipy.stats import multivariate_normal, multivariate_t
    xygrid = np.meshgrid(np.arange(n_rows), np.arange(n_cols))
    locations = np.array([xygrid[0].reshape(-1), xygrid[1].reshape(-1)]).T

    # In-tissue region is the central half
    tissue_mask = ((locations[:,0] > n_rows / 4) & (locations[:,0] < n_rows / 4 * 3) &
                    (locations[:,1] > n_cols / 4) & (locations[:,1] < n_cols / 4 * 3))

    # Sample the true gene reads
    true_rates = np.zeros((n_rows*n_cols, n_genes))
    true_rates[tissue_mask] = np.random.gamma(20,10,size=(1,n_genes))

    # Make the genes vary in space, except gene 1 which is a control example
    length_scale = 0.2
    gene_bandwidth = 1
    Cov = length_scale * np.exp(-np.array([((l[None] - locations[tissue_mask])**2).sum(axis=-1) for l in locations[tissue_mask]]) / (2*gene_bandwidth**2)) + np.diag(np.ones(tissue_mask.sum())*1e-4)
    for g in range(1,n_genes):
        true_rates[tissue_mask,g] *= np.exp(np.random.multivariate_normal(np.zeros(tissue_mask.sum()), Cov))

        # Insert some regions of sparsity
        start = np.array([n_rows / 2, n_cols / 2])

        # Add a random offset
        start = np.round(start + (np.random.random(size=2)*2-1) * np.array([n_rows / 4, n_cols / 4])).astype(int)

        # Draw a box of sparsity
        width = n_rows // 6
        height = n_cols // 6
        sparsity_mask = ((locations[:,0] >= start[0]) & (locations[:,0] < start[0]+width) &
                         (locations[:,1] >= start[1]) & (locations[:,1] < start[1]+height))

        true_rates[sparsity_mask,g] = 0

    true_counts = np.random.poisson(true_rates*spot_bleed_prob)

    # Add some anisotropic bleeding
    bleed_counts = np.zeros_like(true_counts)
    if bleeding == 'gaussian':
        x, y = np.meshgrid(np.arange(n_rows), np.arange(n_cols))
        pos = np.dstack((x, y))
        for i in range(tissue_mask.sum()):
            x_cor, y_cor = locations[tissue_mask][i]
            rv_gaus = multivariate_normal([x_cor, y_cor], [[5, 1], [1, 5]])
            for g in range(n_genes):
                bleed_counts[:,g] += np.random.multinomial(true_counts[tissue_mask][i,g], rv_gaus.pdf(pos).flatten())
    elif bleeding == 't':
        x, y = np.meshgrid(np.arange(n_rows), np.arange(n_cols))
        pos = np.dstack((x, y))
        for i in range(tissue_mask.sum()):
            x_cor, y_cor = locations[tissue_mask][i]
            rv_t = multivariate_t([x_cor, y_cor], [[20, 3], [3, 30]], df=10)
            for g in range(n_genes):
                bleed_counts[:,g] += np.random.multinomial(true_counts[tissue_mask][i,g], rv_t.pdf(pos).flatten())
    elif bleeding == 'anisotropic':
        Distances = np.zeros((n_rows*n_cols,n_rows*n_cols, 4))
        true_w = np.array([0.2, 0.03, 1.5, 0.05])
        true_BleedProbs = np.zeros((n_rows*n_cols, n_rows*n_cols))
        for i in range(n_rows*n_cols):
            if i % 100 == 0:
                print(i)
            Distances[:,i,0] = (locations[i,0] - locations[:,0]).clip(0,None)**2
            Distances[:,i,1] = (locations[:,0] - locations[i,0]).clip(0,None)**2
            Distances[:,i,2] = (locations[i,1] - locations[:,1]).clip(0,None)**2
            Distances[:,i,3] = (locations[:,1] - locations[i,1]).clip(0,None)**2
            h = np.exp(-Distances[:,i].dot(true_w))
            true_BleedProbs[:,i] = h / h.sum()
            for g in range(n_genes):
                bleed_counts[:,g] += np.random.multinomial(true_counts[i,g], true_BleedProbs[:,i])

    # Add the counts due to non-bleeding
    local_counts = np.random.poisson(true_rates*(1-spot_bleed_prob))
    true_counts += local_counts
    bleed_counts += local_counts

    return locations, tissue_mask, true_rates, true_counts, bleed_counts


def imshow_matrix(reads, locations, fill=False):
    to_plot = np.full(locations.max(axis=0).astype(int)+1, np.nan)
    to_plot[locations[:,0], locations[:,1]] = reads
    if fill:
        missing = np.where(np.isnan(to_plot))
        to_plot[missing[0], missing[1]] = to_plot[np.minimum(missing[0]+1, to_plot.shape[0]-1), missing[1]]
        missing = np.where(np.isnan(to_plot))
        to_plot[missing[0], missing[1]] = to_plot[missing[0]-1, missing[1]]
    return to_plot


def tissue_mask_to_grid(tissue_mask, locations):
    grid = np.zeros(locations.max(axis=0)+1)
    grid[locations[:,0], locations[:,1]] = tissue_mask
    return grid

def build_basis_indices(locations, tissue_mask):
    '''Creates 8 sets of basis functions: north, south, east, west, for in- and out-tissue.
    Each basis is how far the 2nd element is from the first element.'''
    diffs = (locations[None] - locations[:,None]).astype(int)
    # north = np.where(diffs[...,0] >= 0)
    # south = np.where(diffs[...,0] < 0)
    # east = np.where(diffs[...,1] >= 0)
    # west = np.where(diffs[...,1] < 0)

    d_max = locations.max()+1
    basis_idxs = np.zeros((locations.shape[0], locations.shape[0], 8), dtype=int)
    basis_mask = np.zeros((locations.shape[0], locations.shape[0], 8), dtype=bool)

    tissue_grid = tissue_mask_to_grid(tissue_mask, locations)

    for i, l in enumerate(locations):
        if i % 100 == 0:
            print(i)
        # North
        basis_coord = 0
        for j in np.where(diffs[i,:,0] >= 0)[0]:
            # Calculate the amount of in-tissue spots from the 1st to 2nd element going north
            basis_idxs[i,j,basis_coord+4] = tissue_grid[l[0]:l[0]+diffs[i,j,0], l[1]].sum()
            basis_mask[i,j,basis_coord+4] = True

            basis_idxs[i,j,basis_coord] = diffs[i,j,0] - basis_idxs[i,j,basis_coord+4]
            basis_mask[i,j,basis_coord] = True

        basis_coord += 1

        # South
        for j in np.where(diffs[i,:,0] < 0)[0]:
            # Calculate the amount of in-tissue spots from the 1st to 2nd element going south
            basis_idxs[i,j,basis_coord+4] = tissue_grid[l[0]+diffs[i,j,0]:l[0], l[1]].sum()
            basis_mask[i,j,basis_coord+4] = True
            
            basis_idxs[i,j,basis_coord] = -diffs[i,j,0] - basis_idxs[i,j,basis_coord+4]
            basis_mask[i,j,basis_coord] = True
            
        basis_coord += 1

        # East
        for j in np.where(diffs[i,:,1] >= 0)[0]:
            # Calculate the amount of in-tissue spots from the 1st to 2nd element going east
            basis_idxs[i,j,basis_coord+4] = tissue_grid[l[0], l[1]:l[1]+diffs[i,j,1]].sum()
            basis_mask[i,j,basis_coord+4] = True

            basis_idxs[i,j,basis_coord] = diffs[i,j,1] - basis_idxs[i,j,basis_coord+4]
            basis_mask[i,j,basis_coord] = True
        basis_coord += 1

        # West
        for j in np.where(diffs[i,:,1] < 0)[0]:
            # Calculate the amount of in-tissue spots from the 1st to 2nd element going west
            basis_idxs[i,j,basis_coord+4] = tissue_grid[l[0], l[1]+diffs[i,j,1]:l[1]].sum()
            basis_mask[i,j,basis_coord+4] = True

            # Calculate the amount of out-tissue
            basis_idxs[i,j,basis_coord] = -diffs[i,j,1] - basis_idxs[i,j,basis_coord+4]
            basis_mask[i,j,basis_coord] = True

        basis_coord += 1

        # Treat the local spot specially
        basis_mask[i, i] = False
        # basis_mask[i, i, 4] = True 

    return basis_idxs, basis_mask


def softplus(x):
    return np.log1p(np.exp(-np.abs(x))) + np.maximum(x, 0)

def weights_from_basis(basis_functions, basis_idxs, basis_mask, tissue_mask, local_weight):
    from utils import stable_softmax
    W = np.sum([basis_functions[d,basis_idxs[:,:,d]] * basis_mask[:,:,d] for d in range(basis_functions.shape[0])], axis=0)
    W[np.arange(tissue_mask.shape[0]), np.arange(tissue_mask.shape[0])] += local_weight*tissue_mask.astype(float)
    Weights = stable_softmax(W, axis=0)
    return Weights

def fit_basis_functions(Reads, tissue_mask, Rates, global_rates, basis_idxs, basis_mask, lam=0, local_weight=100, x_init=None):
    import torch
    from autograd_minimize import minimize
    from torch.distributions.poisson import Poisson
    from torch.distributions.multinomial import Multinomial
    from torch.nn import Softmax, Softplus
    
    # local_weight = 100
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
        
        # print(t_Basis.data.numpy()[:,:15])
        
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
    res = minimize(loss, x_init, method='L-BFGS-B', backend='torch', options={'maxiter':100})

    Betas = softplus(res.x)
    # Betas = res.x
    basis_functions = Betas.reshape(basis_shape)[:,::-1].cumsum(axis=1)[:,::-1]

    Weights = weights_from_basis(basis_functions, basis_idxs, basis_mask, tissue_mask, local_weight)
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

def rates_from_raw(x, tissue_mask, Reads_shape):
    Rates = np.zeros(Reads_shape)
    global_rates = softplus(x[:Reads_shape[1]])
    Rates[tissue_mask] = softplus(x[Reads_shape[1]:]).reshape(tissue_mask.sum(), Reads_shape[1])
    return global_rates, Rates

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

        # Add a small amount of L2 penalty to reduce variance between spots
        # print('Rates loss before L2:', L)
        L += 1e-1*(t_Rates**2).mean()
        # print(Mu.data.numpy()[tissue_mask][:10])
        # print(t_Beta0)
        # print(L)
        # print('Rates loss after L2:', L)

        return L

    # Optimize using a 2nd order method with autograd for gradient calculation. Amazing times we live in.
    res = minimize(loss, x_init, method='L-BFGS-B', backend='torch')

    global_rates, Rates = rates_from_raw(res.x, tissue_mask, Reads.shape)

    return global_rates, Rates, res

def decontaminate_spots(Reads, tissue_mask, basis_idxs, basis_mask,
                        n_top=10, rel_tol=1e-4, max_steps=5, local_weight=15,
                        basis_init=None, Rates_init=None):
    if Rates_init is None:
        # Initialize the rates to be the local observed reads
        Rates = np.copy(Reads[:,:n_top]*tissue_mask[:,None]*1.1).clip(1e-2,None)
        global_rates = np.median(Reads[:,:n_top], axis=0)
    else:
        global_rates, Rates = rates_from_raw(Rates_init, tissue_mask, (Reads.shape[0], n_top))
    

    print(f'Fitting basis functions to first {n_top} genes')
    for step in range(max_steps):
        print(f'\nStep {step+1}/{max_steps}')

        basis_functions, Weights, res = fit_basis_functions(Reads[:,:n_top], tissue_mask, Rates, global_rates, basis_idxs, basis_mask,
                                                            lam=0, local_weight=local_weight, x_init=basis_init)
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

    return global_rates, Rates, basis_functions, Weights, basis_init, Rates_init

def select_local_weight(Reads, tissue_mask, basis_idxs, basis_mask, 
                        min_weight=1, max_weight=100, n_weights=11, test_pct=0.2, n_top=10):
    from scipy.stats import multinomial
    Reads = Reads[:,:n_top]

    # Build the candidate grid
    # weight_grid = np.exp(np.linspace(np.log(min_weight), np.log(max_weight), n_weights)) # Log-linear grid
    weight_grid = np.linspace(min_weight, max_weight, n_weights) # Linear grid

    # Hold out some non-tissue spots
    n_test = int(np.ceil((~tissue_mask).sum()*test_pct))
    test_idxs = np.random.choice(np.where(~tissue_mask)[0], size=n_test, replace=False)
    test_mask = np.zeros(Reads.shape[0], dtype=bool)
    test_mask[test_idxs] = True
    train = (Reads[~test_mask], tissue_mask[~test_mask], np.array(basis_idxs[~test_mask][:,~test_mask]), np.array(basis_mask[~test_mask][:,~test_mask]))
    test = (Reads[test_mask], tissue_mask[test_mask], np.array(basis_idxs[test_mask][:,test_mask]), np.array(basis_mask[test_mask][:,test_mask]))

    N = Reads[test_mask].sum(axis=0)
    losses = np.zeros(n_weights)
    basis_init, Rates_init = None, None
    best_inits = None
    for widx, local_weight in enumerate(weight_grid):
        res = decontaminate_spots(train[0], train[1], train[2], train[3], local_weight=local_weight, basis_init=basis_init, Rates_init=Rates_init, n_top=n_top)
        global_rates, train_Rates, basis_functions, Weights, basis_init, Rates_init = res

        # Reconstruct the weights and rates for the full dataset
        Weights = weights_from_basis(basis_functions, basis_idxs, basis_mask, tissue_mask, local_weight)
        Rates = np.zeros(Reads.shape)
        Rates[~test_mask] = train_Rates

        # Now calculate the test-set-specific probabilities
        Mu = (Rates[None] * Weights[...,None]).sum(axis=1) + global_rates[None]
        Mu = Mu[test_mask]
        Mu = Mu / Mu.sum(axis=0, keepdims=True)
        L = -np.mean([multinomial.logpmf(Reads[test_mask,i], N[i], Mu[:,i]) for i in range(Reads.shape[1])])

        losses[widx] = L
        for i in range(widx+1):
            print(f'{i}. local_weight={weight_grid[i]} loss={losses[i]:.2f}')
        print()

        if np.argmin(losses[:widx+1]) == widx:
            best_inits = (basis_init, Rates_init)

    best = weight_grid[np.argmin(losses)]
    print(f'Best: {best}')

    best_delta = weight_grid[np.argmax(losses[1:] - losses[:-1]) + 1]
    print(f'Best by delta rule: {best_delta}')

    return best, best_delta, losses, weight_grid, best_inits


if __name__ == '__main__':
    import matplotlib
    import matplotlib.pyplot as plt

    np.random.seed(14)
    np.set_printoptions(suppress=True, precision=2)

    # Fixes a weird conflict between pytorch and matplotlib
    import os
    os.environ['KMP_DUPLICATE_LIB_OK']='True'

    #### Load fake data
    base_path = '../data/bleedsim/studentt/'
    plot_path = '../plots/bleedsim/studentt/'
    prefix = 'bleedsim_studentt'
    locations, tissue_mask, true_rates, true_counts, Reads = generate_data(bleeding='t')

    #### Load real data
    base_path = '../data/zebrafish/'
    plot_path = '../plots/zebrafish/'
    prefix = 'zebrafish_A1'
    locations = np.load(f'{base_path}{prefix}_locations.npy')
    tissue_mask = np.load(f'{base_path}{prefix}_tissue_mask.npy')
    Reads = np.load(f'{base_path}{prefix}_reads.npy')
    true_rates, true_counts = None, None

    import os
    if not os.path.exists(base_path):
        os.makedirs(base_path)
    n_genes = Reads.shape[1]

    # Get the distance from each spot to each other spot, unnormed
    spot_distances = locations[None] - locations[:,None]

    # Build the 8 basis functions
    basis_idxs, basis_mask = build_basis_indices(locations)

    #### Filter down to a subset of genes and spots, to speed things up when debugging
    # Filter down to every 3rd row and column
    # subsample_mask = np.all(locations % 2 == 0, axis=1)
    # locations, tissue_mask, Reads = locations[subsample_mask] // 2, tissue_mask[subsample_mask], Reads[subsample_mask]
    
    # Filter down to only a subset genes, most importantly BRAF (gene 58)
    # Reads = Reads[:,[58] + list(range(2))]

    #### Choose the local weight hyperparameter
    best_local_weight, delta_local_weight, lw_losses, lw_grid, (best_basis_init, best_Rates_init) = select_local_weight(Reads, tissue_mask, basis_idxs, basis_mask)
    local_weight = delta_local_weight
    loss_curve_coefs = np.polyfit(lw_grid, lw_losses, 2)
    best_local_quadratic = -loss_curve_coefs[1]/(2*loss_curve_coefs[0])
    print(f'Best by quadratic fit: {best_local_quadratic}')
    plt.plot(lw_grid, np.array([lw_grid**2, lw_grid]).T.dot(loss_curve_coefs[:2])+loss_curve_coefs[2], color='lightblue')
    plt.scatter(lw_grid, lw_losses, color='blue')
    plt.axvline(best_local_quadratic, color='red')
    plt.savefig(f'{plot_path}loss-curve.pdf', bbox_inches='tight')
    plt.close()

    # Select the weight with the best held-out performance
    local_weight = best_local_weight

    #### Fit the model to the data
    global_rates, fit_Rates, basis_functions, Weights, basis_init, Rates_init = decontaminate_spots(Reads, tissue_mask, basis_idxs, basis_mask,
                                                                                                local_weight=local_weight, max_steps=10)
    # fit_Counts = np.load(f'{base_path}zebrafish_A1_reads_fixed.npy')
    # basis_functions = np.load(f'{base_path}zebrafish_A1_bleed_basis.npy')
    # Weights = np.load(f'{base_path}zebrafish_A1_bleed_weights.npy')
    # fit_Rates = np.load(f'{base_path}zebrafish_A1_spot_rates.npy')
    # global_rates = np.load(f'{base_path}zebrafish_A1_global_rates.npy')

    #### Quickly estimate the counts as just rounded versions of the rates
    # fit_Counts = np.round(fit_Rates)
    fit_Counts = np.round(fit_Rates / fit_Rates.sum(axis=0, keepdims=True) * Reads.sum(axis=0, keepdims=True))
    np.save(f'{base_path}{prefix}_reads_fixed.npy', fit_Counts)
    np.save(f'{base_path}{prefix}_bleed_basis.npy', basis_functions)
    np.save(f'{base_path}{prefix}_bleed_weights.npy', Weights)
    np.save(f'{base_path}{prefix}_spot_rates.npy', fit_Rates)
    np.save(f'{base_path}{prefix}_global_rates.npy', global_rates)

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
        # plot_basis_functions(basis_functions)
        # plot_bleed_vectors(locations, tissue_mask, fit_Rates, Weights)
        n_plot_cols = 4
        have_truth = true_rates is not None
        if not os.path.exists(f'{plot_path}/debleed'):
            os.makedirs(f'{plot_path}/debleed')
        for i in range(min(n_genes,3)):
            print(f'Gene {i+1}/{min(n_genes,3)}')
            fig, axarr = plt.subplots(3+int(have_truth), 2, figsize=(5*2,5*(3+int(have_truth))), sharex=True, sharey=True)
            
            im = axarr[0,0].imshow(imshow_matrix(Reads[:,i], locations, fill=~have_truth))
            plt.colorbar(im, ax=axarr[0,0])
            axarr[0,0].set_title('Bleed counts')

            im = axarr[0,1].imshow(imshow_matrix(Reads[:,i], locations, fill=~have_truth), vmax=15)
            plt.colorbar(im, ax=axarr[0,1])
            axarr[0,1].set_title('Bleed counts up to 15')

            im = axarr[1,0].imshow(imshow_matrix(Reads[:,i]*tissue_mask, locations, fill=~have_truth))
            plt.colorbar(im, ax=axarr[1,0])
            axarr[1,0].set_title('Clipped counts')

            im = axarr[1,1].imshow(imshow_matrix(Reads[:,i]*tissue_mask, locations, fill=~have_truth), vmax=15)
            plt.colorbar(im, ax=axarr[1,1])
            axarr[1,1].set_title('Clipped counts up to 15')

            im = axarr[2,0].imshow(imshow_matrix(fit_Counts[:,i], locations, fill=~have_truth))
            plt.colorbar(im, ax=axarr[2,0])
            axarr[2,0].set_title('Denoised counts')

            im = axarr[2,1].imshow(imshow_matrix(fit_Counts[:,i], locations, fill=~have_truth), vmax=15)
            plt.colorbar(im, ax=axarr[2,1])
            axarr[2,1].set_title('Denoised counts up to 15')

            if have_truth:
                im = axarr[3,0].imshow(imshow_matrix(true_counts[:,i], locations, fill=~have_truth))
                plt.colorbar(im, ax=axarr[3,0])
                axarr[3,0].set_title('True counts')

                im = axarr[3,1].imshow(imshow_matrix(true_counts[:,i], locations, fill=~have_truth), vmax=15)
                plt.colorbar(im, ax=axarr[3,1])
                axarr[3,1].set_title('True counts up to 15')


            plt.tight_layout()
            plt.savefig(f'{plot_path}/debleed/{i}.pdf', bbox_inches='tight')
            plt.close()