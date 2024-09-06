
# from multiprocessing import Pool

# Define a function to perform the projection for a single permutation
def project_permutation(perm):
    import numpy as np
    import Fabricofcognition as FC
    fcu = FC.utilities()
    perm = np.array(perm, ndmin=2)
    return fcu.project_new_data(perm)

def generate_null(job):
    job = int(job)
    ind = int(job-1)

    rsn = f'{int(job):02d}'
    # from shiny import App, render, ui, reactive
    # from shinywidgets import output_widget, render_widget
    # import seaborn as sns
    # import matplotlib.pyplot as plt
    # import matplotlib as mpl
    # sys.path.insert(1, '/homes_unix/agillig/Projects/FabricOfCognition/code')
    import Fabricofcognition as FC
    # import plotly.graph_objects as go
    import numpy as np
    import pandas as pd
    import os
    from nilearn import image
    import time
    from tqdm.auto import tqdm
    # from sklearn.neighbors import KernelDensity
    # from sklearn.cluster import AgglomerativeClustering
    # from scipy.spatial import Delaunay
    # import  scipy.spatial.distance as dist
    from brainsmash.mapgen.base import Base
    from scipy.spatial.distance import pdist, squareform

    project_dir = '/homes_unix/agillig/Projects/FabricOfCognition'
    BCS_terms_file = project_dir + '/fabricogcognition_valentina/BCS_3D.csv'
    df = pd.read_csv(BCS_terms_file, sep = ',')

    BCSterms = df['Functions']

    BCScoords = np.asarray(df[['X', 'Y', 'Z']])

    fcu = FC.utilities()
    fcu.analysis_dir = '/analysis/mean_RSNs'


    


    parcels_dir = project_dir + '/analysis/mean_RSNs/parcellations'

    # filestr = [f'rsn-{r:02d}'for r in df['rsn'].unique()]
    rsn_str = [d[-2:] for d in os.listdir(parcels_dir) if d.startswith('rsn-')]
    rsn_str.sort()

    if np.isin(rsn, rsn_str) == False:
        print(f'rsn {rsn} is not to be analysed, exiting')
        exit()


    n_rsn = len(rsn_str)

    dirs = [os.path.join(parcels_dir, r) for r in os.listdir(parcels_dir)]

    parcellations_f = [os.path.join(dirs[i], os.listdir(dirs[i])[0]) for i in range(len(dirs))]
    parcellations = np.stack([np.loadtxt(p, delimiter=',') for p in parcellations_f]).T
    # keep only the parcellations corresponding to the RSNs of interest
    parcellations = parcellations[:,rsn_str.index(rsn)]
    # compute distance matrix
    ordr = fcu.order_parcels()
    parcels_dir = '/homes_unix/agillig/Projects/FabricOfCognition/fabricogcognition_valentina/BCS_repo/ROIs2mm'

    dist_matrix_file = os.path.join(project_dir, 'analysis/mean_RSNs/null_parcellations/distance_matrix.txt')
    
    if os.path.exists(dist_matrix_file) == False:
        # compute simple euclidean distance between parcels (needed for the next step)
        centers_parcels = []
        from scipy import ndimage
        for prcl in ordr:

            img = image.load_img(os.path.join(parcels_dir, str(prcl + '.nii.gz')))
            centers_parcels.append(ndimage.center_of_mass(img.get_fdata()))
        centers_parcels = np.stack(centers_parcels)

        # pairwise distances
        
        dist_parcels = squareform(pdist(centers_parcels, metric='euclidean'))

        # save to a file
        os.makedirs(os.path.dirname(dist_matrix_file), exist_ok=True)
        np.savetxt(dist_matrix_file, dist_parcels)
        # else:
        #     dist_parcels = np.loadtxt(dist_matrix_file, delimiter=',')

    null_projections = {}
    null_dir = project_dir + '/analysis/mean_RSNs/projections_null'
    null_dir_parcels = project_dir + '/analysis/mean_RSNs/null_parcellations'


    


    os.makedirs(null_dir, exist_ok=True)
    os.makedirs(null_dir_parcels, exist_ok=True)


    save_dir = null_dir + f'/rsn-{rsn}'
    save_dir_parcels = null_dir_parcels + f'/rsn-{rsn}'
    os.makedirs(save_dir, exist_ok=True)
    save_file = os.path.join(save_dir, f'rsn-{rsn}_null_projections.csv')
    # save_file_parcels = os.path.join(save_dir_parcels, f'rsn-{rsn}_null_parcellations.csv')

    # if os.path.exists(save_file):
    #     continue
    print(f'processing rsn {rsn}')
    


    # if os.path.exists(save_file):
    #     exit()
    

    n_perm = 1000
    n_batches = 10
    permuted = []
    # instantiate class and generate 1000 surrogates
    gen = Base(parcellations, dist_matrix_file)  # note: can pass numpy arrays as well as filenames
    for batch in range(n_batches):
        projections = []
        start_time = time.time()
        print(f'batch {batch+1}')
        print(f'generating {n_perm} surrogates')
        tmp_perm = gen(n=n_perm)
        # permuted.append(tmp_perm)
        # print(tmp_perm[0][:3])
        end_time = time.time()
        elapsed_time = end_time - start_time
        print(f'elapsed time: {elapsed_time:.2f} s')

        save_file_parcels = os.path.join(save_dir_parcels, f'rsn-{rsn}_null_parcellations_batch-{batch+1:02d}_of_{n_batches}.csv')
        print('saving surrogate maps')
        os.makedirs(os.path.dirname(save_file_parcels), exist_ok=True)
        np.savetxt(save_file_parcels, tmp_perm)
        # save each batch in a separate file
        # print('projecting null data')
        # save_file = os.path.join(save_dir, f'rsn-{rsn}_null_projections_batch-{batch+1:02d}_of_{n_batches}.csv')
        # # for perm in tqdm(tmp_perm):
        # #     perm = np.array(perm, ndmin=2)
        # #     projections.append(fcu.project_new_data(perm))
        # start_time = time.time()
        # with Pool() as pool:
        #     # Apply the projection function to each permutation in parallel
        #     projections = pool.map(project_permutation, tmp_perm)
        # projections = np.vstack(projections)
        # end_time = time.time()
        # elapsed_time = end_time - start_time
        # print(f'elapsed time: {elapsed_time:.2f} s')
        # print('saving null projections')
        # np.savetxt(save_file, projections)
    # permuted = np.vstack(permuted)
    # print(f'permuted shape: {permuted.shape}')
    # print(f'size of array (MB): {permuted.nbytes / 1e6}')
        # Create a pool of worker processes
    # with Pool() as pool:
        # Apply the projection function to each permutation in parallel
        # projections = pool.map(project_permutation, range(n_perm)) 
        # AttributeError: Can't pickle local object 'project_null.<locals>.project_permutation'
    # projections = fcu.project_new_data(permuted)
    # projections = np.stack(projections).squeeze()
    # print('saving null projections')
    # np.savetxt(save_file, projections)