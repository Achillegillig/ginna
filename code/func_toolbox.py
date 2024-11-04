
import numpy as np
import os
from pathlib import Path
import requests
import shutil
import zipfile
class Utilities:
    
    def __init__(self) -> None:
        # self.user = os.getlogin()
        # if self.user == 'root':
        #     self.user = 'achillegillig'

        # init dirs
        # directory of parcels
        self._project_dir = '/homes_unix/agillig/Projects/FabricOfCognition'
        self._analysis_dir = self.project_dir + '/analysis'
        # self._projections_dir = self.analysis_dir + '/projections'
        # self.parcellations_dir = self.analysis_dir + '/parcellations'

        self.RSNs_filelist_dir = '/homes_unix/agillig/RSN_MRISHARE_filelist'

        self.n_terms = None
        

    @property
    def project_dir(self):
        return self._project_dir

    @project_dir.setter
    def project_dir(self, value):
        self._project_dir = value
        self.analysis_dir = self._analysis_dir

    @property
    def analysis_dir(self):
        return self._analysis_dir

    @analysis_dir.setter
    def analysis_dir(self, value):
        self._analysis_dir = self.project_dir + value
        # self._projections_dir = self._analysis_dir + '/projections'

    @property
    def projections_dir(self):
        self._projections_dir = self.analysis_dir + '/projections'
        return self._projections_dir

    @property
    def parcellations_dir(self):
        self._parcellations_dir = self.analysis_dir + '/parcellations'
        return self._parcellations_dir

    @property
    def parcel_dir(self):
        return self._parcel_dir
    
    @parcel_dir.setter
    def parcel_dir(self, value):
        self._parcel_dir = value

    def set_project_dir(self, project_dir):
        self._project_dir = project_dir
        self._analysis_dir = self._project_dir + '/analysis'

    def set_atlas(self, atlas):
        self.atlas = atlas
    
    def set_atlas_name(self, atlas_name):
        self.atlas_name = atlas_name

    def compute_correlations(self, input_vector, database):
        import numpy as np
        # input_vector: 2D array containing the data of the map to analyze; must have same dimensions as database (410)
        # use parcellate to transform an input nifti image into its parcellation vector
        self.input_vector = input_vector
        # database: file containing the parcellated neurosynth maps
        self.database = database

        self.n_terms = self.database.shape[0]

        self.correlations = [np.corrcoef(self.input_vector, self.database.iloc[j,:])[1,0] for j in range(self.n_terms)]

        return self.correlations
    

    def create_parcellation_atlas(self):
        import numpy as np
        from nilearn.image import get_data, new_img_like, load_img
        from os.path import join, isfile

        atlas_file = self.parcellations_dir + '/parcellation_atlas.nii.gz'

        if isfile(atlas_file):
            print('loading existing parcellation atlas')
            atlas_img = load_img(atlas_file)
        else:
            print('creating parcellation atlas')
            self.parcel_list = [s + '.nii.gz' for s in self.order_parcels()]

            atlas_array = np.zeros((91,109,91), dtype='int32')
            atlas_shape = atlas_array.shape
            atlas_vec = atlas_array.reshape(-1,1)
            
            for i, parcel in enumerate(self.parcel_list):
                parcel_file = join(self.parcel_dir, parcel)
                # print(parcel_list[prcl])
                parcel_data = get_data(parcel_file).astype('int').reshape(-1,1)
                atlas_vec[parcel_data == 1] = int(i+1)

            atlas_array = atlas_vec.reshape(atlas_shape)
            atlas_img = new_img_like(parcel_file, atlas_array)
            atlas_img.to_filename(atlas_file)

        return atlas_img

    def compute_correlation_null(self, rsn, n_perm = 1000, n_batches = 10, overwrite = False):
        print(f'processing rsn {rsn}')

        if type(rsn) == int:
            rsn = f'{rsn:02d}'

        null_dir = self._analysis_dir + '/null_correlations'
        save_dir = null_dir + f'/rsn-{rsn}'
        os.makedirs(save_dir, exist_ok=True)
        save_file = os.path.join(save_dir, f'rsn-{rsn}_null_correlations.csv')    

        if os.path.isfile(save_file) and overwrite == False:
            print('file already exists; no overwriting - loading existing data')
            return np.loadtxt(save_file)
        
        data = []
        for batch in range(1,n_batches+1):
            null_prcl_dir = self._analysis_dir + f'/mean_RSNs_{self.atlas_name}/null_parcellations'
            null_parcellations_file = null_prcl_dir + f'/rsn-{rsn}/rsn-{rsn}_null_parcellations_batch-{batch:02d}_of_{n_batches}.csv'
            temp_data = np.loadtxt(null_parcellations_file, delimiter = ' ')
            data.append(temp_data)
        data = np.concatenate(data, axis = 0)
        # print(f'shape of data: {data.shape}')

        index = int(rsn) - 1

        # Get the target series
        corr_temp = []
        n_terms = self.dataset_parcellated.shape[0]
        for j in range(n_terms):
            
            term_single = self.dataset_parcellated.iloc[j,:].values
            # Compute the Pearson correlation coefficient for each column
            corr_temp.append([np.corrcoef(data.T[:, i], term_single)[0,1] for i in range(n_perm * n_batches)])

            # corr_temp = []
            # for j, (term, file) in enumerate(zip(BCSterms, BCS_maps_files)): 
            #     img_comparison = image.index_img(concat_ds, j)
            #     data_comparison = img_comparison.get_fdata().copy()
            #     corr_temp.append(np.corrcoef(data.flatten(), data_comparison.flatten())[1,0])
        corr_temp = np.array(corr_temp)
        # spatial_correlation_null[rsn] = corr_temp
        # save to file
        print('saving file')
        np.savetxt(save_file, corr_temp) 
        return corr_temp
    

    def compute_pvalues(self, observed_correlations, null_distribution):
        import numpy as np 
        # nulldistribution: 2D array containing the null distribution of the correlations, shape (n_terms, n_perms)
        null_distribution = null_distribution

        corrected_null_distribution = np.max(null_distribution, axis=0)
        if type(observed_correlations) == list:
            observed_correlations = np.array(observed_correlations)

        if self.n_terms is None:
            self.n_terms = observed_correlations.shape[0]

        # data_parcellated = np.loadtxt(parcellation_file, delimiter = ',')
        pvalues = []
        pvalues_corr = []
        observed_corr_list = []
        # null_distribution = np.loadtxt(null_distribution_file, delimiter = ' ')
        tmp_ind_obs = []
        for t in range(self.n_terms): 
            
            observed_correlation = observed_correlations[t]
            # print(f'observed correlation: {observed_corr}')
            # observed_corr_list.append(observed_corr)
            null_corr= []

            # without correction for multiple comparisons
            null_distribution_term = null_distribution[t,:]
            distr_sorted = np.sort(null_distribution_term)

            n_val = len(distr_sorted) + 1
            ind_obs = n_val - np.searchsorted(distr_sorted, observed_correlation) #+1 # +1 because python indices start at 0 
            # indice observé: rank of the observed correlation in the null distribution
            # tmp_ind_obs.append(ind_obs)
            p_val = ind_obs / n_val
            pvalues.append(p_val)


             # with CORRECTION FOR MULTIPLE COMPARISONS
            corrected_distr_sorted = np.sort(corrected_null_distribution)
            n_val = len(corrected_distr_sorted) + 1
            ind_obs_corr = n_val - np.searchsorted(corrected_distr_sorted, observed_correlation) #+1 # +1 because python indices start at 0
            # print(ind_obs)
            # tmp_ind_obs.append(ind_obs)
            p_val_corr = ind_obs_corr / n_val
            pvalues_corr.append(p_val_corr)

        pvalues = np.asarray(pvalues)
        pvalues_corr = np.asarray(pvalues_corr)
           
        return pvalues, pvalues_corr


    def generate_null_distribution(self, input_vector, out_dir, n_perm_per_batch, n_batches, atlas_file):
        # use brainsmash to generate surrogate data that preserves spatial autocorrelation (Burt, 2020)
        # https://github.com/murraylab/brainsmash
        import numpy as np
        import pandas as pd
        import os
        from nilearn import image
        import time
        from brainsmash.mapgen.base import Base
        from scipy.spatial.distance import pdist, squareform
        from scipy import ndimage

        self.input_vector = input_vector
        self.out_dir = out_dir
        self.atlas_file = atlas_file
        self.n_perm_per_batch = n_perm_per_batch
        self.n_batches = n_batches 

        self.dist_matrix_file = os.path.join(self.out_dir, 'distance_matrix.txt')
        os.makedirs(os.path.dirname(self.dist_matrix_file), exist_ok=True)
    
        self.atlas = image.load_img(self.atlas_file)
        self.atlas_data = self.atlas.get_fdata().copy()

        n_parcels = int(np.max(self.atlas_data.ravel()))

        if os.path.exists(self.dist_matrix_file) == False:
            print('computing distance matrix')
            # compute simple euclidean distance between parcels (needed for the next step)
            centers_parcels = []
            
            for prcl in range(1, n_parcels+1):
                mask = self.atlas_data == prcl
                tmp_data = np.where(mask, 1, 0)
                centers_parcels.append(ndimage.center_of_mass(tmp_data))
            centers_parcels = np.stack(centers_parcels)

            # pairwise distances       
            self.dist_parcels = squareform(pdist(centers_parcels, metric='euclidean'))

            # save to a file
            # warning: add a filelock to prevent bugs from parallel processing
            np.savetxt(self.dist_matrix_file, self.dist_parcels)
        else:
            print('using existing distance matrix')
        
        print('generating surrogates')
        

        np.random.seed(15011999)
        # n_perm = 1000
        # n_batches = 10
        permuted = []
        # instantiate class and generate 1000 surrogates
        self.filelist = []
        gen = Base(self.input_vector, self.dist_matrix_file)  # note: can pass numpy arrays as well as filenames
        for batch in range(self.n_batches):
            projections = []
            start_time = time.time()

            print(f'batch {batch+1}')
            print(f'generating {self.n_perm_per_batch} surrogates')
            self.tmp_perm = gen(n=self.n_perm_per_batch)

            end_time = time.time()
            elapsed_time = end_time - start_time
            print(f'elapsed time: {elapsed_time:.2f} s')

            self.save_file_parcels = os.path.join(self.out_dir, f'surrogates_batch-{batch+1:02d}_of_{self.n_batches}.csv')
            self.filelist.append(self.save_file_parcels)
            print('saving surrogate maps')
            os.makedirs(os.path.dirname(self.save_file_parcels), exist_ok=True)
            np.savetxt(self.save_file_parcels, self.tmp_perm)

        return self.filelist





    def parcellate(self, img, **kwargs):
        import numpy as np
        import nilearn
        import os
        from os import listdir
        from os.path import isfile, join
        from nilearn import masking
        from nilearn.image import load_img
        from nilearn.maskers import NiftiLabelsMasker

        self.atlas = kwargs.get('atlas', 'bcs')

        if self.atlas == 'bcs':
            parcel_list = [s + '.nii.gz' for s in self.order_parcels()]

            n_parcels = len(parcel_list)

            # extract each parcels' data and average
            self._atlas = self.create_parcellation_atlas()
        else:
            self._atlas = self.atlas

        masker = NiftiLabelsMasker(labels_img=self._atlas, strategy='mean')

        avg_data = masker.fit_transform(img)
    
        return avg_data
    
    def parcellate_neurosynth_dataset(self, dataset, terms, atlas_file, out_file):
        from nilearn.image import index_img
        import numpy as np
        import pandas as pd
        import os
        # dataset: a 4D nifti file containing all the maps
        self.dataset = dataset
        self.terms = terms
        self.atlas_file = atlas_file
        self.out_file = out_file

        # CREATE & retrieve parcellated 506 meta analytic maps
        self.n_terms = self.dataset.shape[3]

        self.parcellated = []
        if os.path.isfile(self.out_file) == False:
            for t in range(self.n_terms):
                self.temp_img = index_img(self.dataset, t)
                self.parcellated.append(self.parcellate(self.temp_img, atlas=self.atlas_file))

            self.parcellated = np.array(self.parcellated).squeeze()
            
            self.parcels = [i for i in range(1, self.parcellated.shape[1] +1)]
            df = pd.DataFrame(self.parcellated, index=self.terms, columns = self.parcels)
            df.to_csv(self.out_file)
        
        df_parcellated = pd.read_csv(self.out_file, sep = ',', index_col = 0)
        return df_parcellated

   
    def retrieve_neurosynth_files(self, terms_maps_dir):
        import os
        self.terms_maps_dir = terms_maps_dir
        self.terms_maps_files = [os.path.join(self.terms_maps_dir, f) for f in os.listdir(self.terms_maps_dir) if f.endswith('.nii.gz')]
        self.terms_maps_files.sort()

        self.terms = [os.path.basename(f)[:-12] for f in self.terms_maps_files]

        return self.terms, self.terms_maps_files
    
    def concatenate_maps(self, maps, savefile):
        # # # Create A 4D volume with all 506 maps of the model
        from nilearn.image import concat_imgs, load_img
        import os 
        self.savefile = savefile
        self.maps = maps
        if os.path.isfile(self.savefile) == False:
            concat_imgs(self.maps).to_filename(self.savefile)
        self.concat_ds = load_img(self.savefile)

        return self.concat_ds

    
def fetch_neurosynth_data(out_dir):

    repo_url = 'https://github.com/vale-pak/BCS'
    files = {'dataset': '2017_dataset.zip',
            'terms': 'BCS_3D.csv'}
    
    for key, file in files.items():
        download_files_from_github(repo_url, file, f'{out_dir}/{key}')

        if file.endswith('.zip') == False:
            continue

        print(f'Unzipping {file}')
        with zipfile.ZipFile(f'{out_dir}/{key}/{file}', 'r') as zip_ref:
            zip_ref.extractall(f'{out_dir}/{key}')

        macosx_dir = f'{out_dir}/{key}/__MACOSX'
        if os.path.exists(macosx_dir):
            shutil.rmtree(macosx_dir)
        # os.remove(f'{out_dir}/{key}/{file}')

        # Move files from subfolder to parent folder
        subfolder = f'{out_dir}/{key}/Originals_2017_dataset'
        for root, dirs, files in os.walk(subfolder):
            for file in files:
                file_path = os.path.join(root, file)
                shutil.move(file_path, Path(subfolder).parent)
        # Remove empty dir
        if os.path.exists(subfolder):
            os.rmdir(subfolder)

def download_files_from_github(repo_url, file_paths, local_dir, branch='main'):
    """
    Download files from a specified GitHub repository directory.

    Parameters:
    - repo_url: URL to the GitHub repository (e.g., 'https://github.com/user/repo').
    - file_paths: List of file paths within the repository to download.
    - local_dir: Local directory to save the downloaded files.
    - branch: Branch name to download from (default is 'main').
    """
    # Ensure the local directory exists
    os.makedirs(local_dir, exist_ok=True)

    file_paths = [file_paths] if isinstance(file_paths, str) else file_paths

    for file_path in file_paths:
        # Construct the URL to the raw content
        raw_url = f"{repo_url.replace('github.com', 'raw.githubusercontent.com')}/{branch}/{file_path}"
        
        # Fetch the file
        response = requests.get(raw_url)
        if response.status_code == 200:
            # Save the file locally
            local_file_path = os.path.join(local_dir, os.path.basename(file_path))
            with open(local_file_path, 'wb') as file:
                file.write(response.content)
            print(f"Downloaded {file_path} to {local_file_path}")
        else:
            print(f"Failed to download {file_path} from {raw_url}")
