
import func_toolbox as ftools
from nilearn import image
import numpy as np
import os
import pandas as pd
import time

from brainsmash.mapgen.base import Base
from scipy.spatial.distance import pdist, squareform


def generate_null(job, project_dir, n_perm=1000, n_batches=10, overwrite=False):
    job = int(job)
    ind = int(job-1)

    rsn = f'{int(job):02d}'
    fcu = ftools.Utilities()
    fcu.analysis_dir = '/analysis/mean_RSNs'

    np.random.seed(15011999)

    
    parc_atlas = 'aicha-aal3'

    parcels_dir = project_dir + f'/analysis/mean_RSNs_{parc_atlas}/parcellations'


    dirs = [os.path.join(parcels_dir, r) for r in os.listdir(parcels_dir)]

    parcellations_f = [os.path.join(dirs[i], os.listdir(dirs[i])[0]) for i in range(len(dirs))]
    parcellations = np.stack([np.loadtxt(p, delimiter=',') for p in parcellations_f]).T

    # load the parcellations corresponding to the RSNs of interest
    parcellations = np.loadtxt(parcellations_f[ind], delimiter=',').T

    # compute distance matrix (indicates euclidean distance between brain parcels)
    dist_matrix_file = os.path.join(project_dir, 'analysis/null_parcellations/distance_matrix.txt')
    os.makedirs(os.path.dirname(dist_matrix_file), exist_ok=True)
    
    atlas_file = f'{project_dir}/data/parcellation_atlases/{parc_atlas}/parcels_{parc_atlas}.nii.gz'
    atlas = image.load_img(atlas_file)
    atlas_data = atlas.get_fdata().copy()
    n_parcels = int(np.max(atlas_data.ravel()))

    if os.path.exists(dist_matrix_file) == False and overwrite == False:
        # compute simple euclidean distance between parcels (needed for the next step)
        centers_parcels = []
        from scipy import ndimage
        for prcl in range(1, n_parcels+1):
            mask = atlas_data == prcl
            tmp_data = np.where(mask, 1, 0)
            centers_parcels.append(ndimage.center_of_mass(tmp_data))
        centers_parcels = np.stack(centers_parcels)

        # pairwise distances
        dist_parcels = squareform(pdist(centers_parcels, metric='euclidean'))

        # save to a file
        os.makedirs(os.path.dirname(dist_matrix_file), exist_ok=True)
        # warning: add a filelock to prevent bugs from parallel processing
        np.savetxt(dist_matrix_file, dist_parcels)


    null_dir_parcels = project_dir + f'/analysis/mean_RSNs_{parc_atlas}/null_parcellations'
    # os.makedirs(null_dir, exist_ok=True)
    os.makedirs(null_dir_parcels, exist_ok=True)
    save_dir_parcels = null_dir_parcels + f'/rsn-{rsn}'
    os.makedirs(save_dir_parcels, exist_ok=True)



    print(f'processing rsn {rsn}')

    
    np.random.seed(15011999)

    # instantiate brainsmash class and generate 1000 surrogates
    gen = Base(parcellations, dist_matrix_file)  # note: for dist matrix, can pass numpy arrays as well as filenames
    for batch in range(n_batches):
        start_time = time.time()
        print(f'batch {batch+1}')
        print(f'generating {n_perm} surrogates')
        tmp_perm = gen(n=n_perm)
        end_time = time.time()
        elapsed_time = end_time - start_time
        print(f'elapsed time: {elapsed_time:.2f} s')

        save_file_parcels = os.path.join(save_dir_parcels, f'rsn-{rsn}_null_parcellations_batch-{batch+1:02d}_of_{n_batches}.csv')
        print(f'saving surrogate maps to {save_file_parcels}')
        np.savetxt(save_file_parcels, tmp_perm)