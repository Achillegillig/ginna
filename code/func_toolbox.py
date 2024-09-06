
class Utilities:
    
    def __init__(self) -> None:
        import os
        import numpy as np
        # self.user = os.getlogin()
        # if self.user == 'root':
        #     self.user = 'achillegillig'

        # init dirs
        # directory of parcels
        self._project_dir = '/homes_unix/agillig/Projects/FabricOfCognition'
        self._analysis_dir = self.project_dir + '/analysis'
        self._projections_dir = self.analysis_dir + '/projections'
        # self.parcellations_dir = self.analysis_dir + '/parcellations'

        self.RSNs_filelist_dir = '/homes_unix/agillig/RSN_MRISHARE_filelist'
        

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

    def compute_correlations_null(self, savefile, nullfiles, database, n_perm_per_batch, n_batches):
        
        import os
        import numpy as np

        self.savefile = savefile
        self.nullfiles = nullfiles
        self.database = database
        self.n_perm_per_batch = n_perm_per_batch
        self.n_batches = n_batches

        if os.path.isfile(self.savefile):
            print('file already exists; no overwriting')
            self.corr_temp = np.loadtxt(self.savefile)
            return self.corr_temp
        
        self.data = []
        for i, batch in enumerate(range(1,self.n_batches+1)):
            null_parcellations_file = self.nullfiles[i]
            self.temp_data = np.loadtxt(null_parcellations_file, delimiter = ' ')
            self.data.append(self.temp_data)
        self.data = np.concatenate(self.data, axis = 0)
        # print(f'shape of data: {data.shape}')
        self.n_terms = self.database.shape[0]
        # Get the target series

        self.corr_temp = []
        for j in range(self.n_terms):
            
            self.term_single = self.database.iloc[j,:].values
            # Compute the Pearson correlation coefficient for each column
            self.corr_temp.append([np.corrcoef(self.data.T[:, i], self.term_single)[0,1] for i in range(self.n_perm_per_batch * self.n_batches)])
        self.corr_temp = np.array(self.corr_temp)

        print('saving file')
        np.savetxt(self.savefile, self.corr_temp) 

        return self.corr_temp
    

    def compute_pvalues(self, observed_correlations, null_distribution):
        import numpy as np 
        self.observed_correlations = observed_correlations
        # nulldistribution: 2D array containing the null distribution of the correlations, shape (n_terms, n_perms)
        self.null_distribution = null_distribution

        self.corrected_null_distribution = np.max(self.null_distribution, axis=0)

        self.n_terms = self.observed_correlations.shape[0]

        # data_parcellated = np.loadtxt(parcellation_file, delimiter = ',')
        self.pvalues = []
        self.pvalues_corr = []
        observed_corr_list = []
        # null_distribution = np.loadtxt(null_distribution_file, delimiter = ' ')
        tmp_ind_obs = []
        for t in range(self.n_terms): 
            
            self.observed_correlation = self.observed_correlations[t]
            # print(f'observed correlation: {observed_corr}')
            # observed_corr_list.append(observed_corr)
            null_corr= []

            # without correction for multiple comparisons
            self.null_distribution_term = self.null_distribution[t,:]
            self.distr_sorted = np.sort(self.null_distribution_term)

            self.n_val = len(self.distr_sorted) + 1
            self.ind_obs = self.n_val - np.searchsorted(self.distr_sorted, self.observed_correlation) #+1 # +1 because python indices start at 0 
            # indice observé: rank of the observed correlation in the null distribution
            # tmp_ind_obs.append(ind_obs)
            self.p_val = self.ind_obs / self.n_val
            self.pvalues.append(self.p_val)


             # with CORRECTION FOR MULTIPLE COMPARISONS
            self.corrected_distr_sorted = np.sort(self.corrected_null_distribution)
            self.n_val = len(self.corrected_distr_sorted) + 1
            self.ind_obs_corr = self.n_val - np.searchsorted(self.corrected_distr_sorted, self.observed_correlation) #+1 # +1 because python indices start at 0
            # print(ind_obs)
            # tmp_ind_obs.append(ind_obs)
            self.p_val_corr = self.ind_obs_corr / self.n_val
            self.pvalues_corr.append(self.p_val_corr)

        self.pvalues = np.asarray(self.pvalues)
        self.pvalues_corr = np.asarray(self.pvalues_corr)
           
        return self.pvalues, self.pvalues_corr


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

    

    