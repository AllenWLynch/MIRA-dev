from ._model_optimizers import fit_models, score
import pickle
import numpy as np
from scipy.sparse import isspmatrix
from mira.adata_interface import rp_model as rpi
from mira.adata_interface import core as adi


class GeneModel:
    '''
    Gene-level RP model object.
    '''

    def __init__(self,*,
        gene, 
    ):
        self.gene = gene

    @property
    def prefix(self):
        return self.gene
    
    def fit(self, features):
        self._params = fit_models(**features)
        return self

    def predict(self, features):
        model_scores, model_logrates = score(features)
        return model_logrates

    def score(self, features):
        model_scores, model_logrates = score(**self._params, **features)
        return model_scores

    def predict_and_score(self, features):
        return score(**self._params, **features)

    def _get_save_data(self):
        return self._params

    def _load_save_data(self, data):
        self._params = data


    @staticmethod
    def _select_informative_samples(expression, n_bins = 20, n_samples = 1500, seed = 2556):
        '''
        Bin based on contribution to overall expression, then take stratified sample to get most informative cells.
        '''
        np.random.seed(seed)

        expression = np.ravel(expression)
        assert(np.all(expression >= 0))

        expression = np.log1p(expression)
        expression += np.mean(expression)

        sort_order = np.argsort(-expression)

        cummulative_counts = np.cumsum(expression[sort_order])
        counts_per_bin = expression.sum()/(n_bins - 1)
        
        samples_per_bin = n_samples//n_bins
        bin_num = cummulative_counts//counts_per_bin
        
        differential = 0
        informative_samples = []
        samples_taken = 0
        for _bin, _count in zip(*np.unique(bin_num, return_counts = True)):
            
            if _bin == n_bins - 1:
                take_samples = n_samples - samples_taken
            else:
                take_samples = samples_per_bin + differential

            if _count < take_samples:
                informative_samples.append(
                    sort_order[bin_num == _bin]
                )
                differential = take_samples - _count
                samples_taken += _count

            else:
                differential = 0
                samples_taken += take_samples
                informative_samples.append(
                    np.random.choice(sort_order[bin_num == _bin], size = take_samples, replace = False)
                )

        return np.concatenate(informative_samples)


    @staticmethod
    def _prob_ISD(hits_matrix,*, correction_vector,
        upstream_weights, downstream_weights, 
        promoter_weights, upstream_idx, promoter_idx, downstream_idx,
        upstream_distances, downstream_distances, read_depth, 
        softmax_denom, gene_expr, NITE_features, params, bn_eps):

        assert(isspmatrix(hits_matrix))
        assert(len(hits_matrix.shape) == 2)
        num_factors = hits_matrix.shape[0]

        def tile(x):
            x = np.expand_dims(x, -1)
            return np.tile(x, num_factors+1).transpose((0,2,1))

        def delete_regions(weights, region_mask):
            
            num_regions = len(region_mask)
            hits = 1 - hits_matrix[:, region_mask].toarray().astype(int) #1, factors, regions
            hits = np.vstack([np.ones((1, num_regions)), hits])
            hits = hits[np.newaxis, :, :].astype(int)

            return np.multiply(weights, hits)

        upstream_weights = delete_regions(tile(upstream_weights), upstream_idx) #cells, factors, regions
        promoter_weights = delete_regions(tile(promoter_weights), promoter_idx)
        downstream_weights = delete_regions(tile(downstream_weights), downstream_idx)

        read_depth = read_depth[:, np.newaxis]
        softmax_denom = softmax_denom[:, np.newaxis]

        upstream_distances = upstream_distances[np.newaxis, np.newaxis, :]
        downstream_distances = downstream_distances[np.newaxis,np.newaxis, :]
        expression = gene_expr[:, np.newaxis]

        def RP(weights, distances, d):
            return (weights * np.power(0.5, distances/(1e3 * d))).sum(-1)

        f_Z = params['a'][0] * RP(upstream_weights, upstream_distances, params['distance'][0]) \
        + params['a'][1] * RP(downstream_weights, downstream_distances, params['distance'][1]) \
        + params['a'][2] * promoter_weights.sum(-1) # cells, factors

        original_data = f_Z[:,0]
        sorted_first_col = np.sort(original_data).reshape(-1)
        quantiles = np.argsort(f_Z, axis = 0).argsort(0)

        f_Z = sorted_first_col[quantiles]
        f_Z[:,0] = original_data

        #f_Z = (f_Z - f_Z[:,0].mean(0,keepdims = True))/np.sqrt(f_Z[:, 0].var(0, keepdims = True) + bn_eps)
        f_Z = (f_Z - params['bn_mean'])/np.sqrt(params['bn_var'] + bn_eps)

        indep_rate = np.exp(params['gamma'] * f_Z + params['bias'] + \
                correction_vector[:, np.newaxis])
                
        compositional_rate = indep_rate/softmax_denom

        mu = np.exp(read_depth) * compositional_rate

        p = mu / (mu + params['theta'])

        logp_data = nbinom(params['theta'], 1 - p).logpmf(expression)
        logp_summary = logp_data.sum(0)
        return logp_summary[0] - logp_summary[1:]#, f_Z, expression, logp_data


    def probabilistic_isd(self, features, hits_matrix, n_samples = 1500, n_bins = 20):
        
        np.random.seed(2556)
        N = len(features['gene_expr'])
        informative_samples = self._select_informative_samples(features['gene_expr'], 
            n_bins = n_bins, n_samples = n_samples)
        
        
        for k in 'gene_expr,correction_vector,upstream_weights,downstream_weights,promoter_weights,softmax_denom,read_depth,NITE_features'.split(','):
            features[k] = features[k][informative_samples]

        samples_mask = np.zeros(N)
        samples_mask[informative_samples] = 1
        samples_mask = samples_mask.astype(bool)
        
        return self._prob_ISD(
            hits_matrix, **features, 
            params = self._get_normalized_params(), 
            bn_eps= self.bn.eps
        ), samples_mask


    def _get_RP_model_coordinates(self, bin_size = 50,
        decay_periods = 20, promoter_width = 3000, *,
        gene_chrom, gene_start, gene_end, gene_strand):

        assert(isinstance(promoter_width, int) and promoter_width > 0)
        assert(isinstance(decay_periods, int) and decay_periods > 0)
        assert(isinstance(bin_size, int) and bin_size > 0)
    
        upstream, downstream = 1e3*self._params['upstream_decay'], 1e3*self._params['downstream_decay']

        left_decay, right_decay, start_pos = upstream, downstream, gene_start

        if gene_strand == '-':
            left_decay, right_decay, start_pos = downstream, upstream, gene_end
        
        left_extent = int(decay_periods*left_decay)
        left_x = np.linspace(1, left_extent, left_extent//bin_size).astype(int)
        left_y = 0.5**(left_x / left_decay)

        right_extent = int(decay_periods*right_decay)
        right_x = np.linspace(0, right_extent, right_extent//bin_size).astype(int)
        right_y = 0.5**(right_x / right_decay)


        left_x = -left_x[::-1] - promoter_width//2 + start_pos
        right_x = right_x + promoter_width//2 + start_pos
        promoter_x = [-promoter_width//2 + start_pos]
        promoter_y = 1.

        x = np.concatenate([left_x, promoter_x, right_x])
        y = np.concatenate([left_y[::-1], promoter_y, right_y])

        return x, y


    @adi.wraps_modelfunc(fetch_TSS_from_adata, 
        fill_kwargs = ['gene_chrom','gene_start','gene_end','gene_strand'])
    def write_bedgraph(self, bin_size = 50,
        decay_periods = 20, promoter_width = 3000,*, save_name,
        gene_chrom, gene_start, gene_end, gene_strand):
        '''
        Write bedgraph of RP model coverage. Useful for visualization with 
        Bedtools.

        Parameters
        ----------

        adata : anndata.AnnData
            AnnData object with TSS data annotated by `mira.tl.get_distance_to_TSS`.
        save_name : str
            Path to saved bedgraph file.
        scale_height : boolean, default = False
            Write RP model tails proportional in height to their respective
            multiplicative coeffecient. Useful for evaluating not only the distance
            of predicted regulatory influence, but the weighted importance of regions 
            in terms of predicting expression.
        decay_periods : int>0, default = 10
            Number of decay periods to write.
        promoter_width : int>0, default = 0
            Width of flat region at promoter of gene in base pairs (bp). MIRA default is 3000 bp.

        Returns
        -------

        None

        '''
        coord, value = self._get_RP_model_coordinates( bin_size = bin_size,
            decay_periods = decay_periods, promoter_width = promoter_width,
            gene_chrom = gene_chrom, gene_start = gene_start, 
            gene_end = gene_end, gene_strand = gene_strand)

        with open(save_name, 'w') as f:
            for start, end, val in zip(coord[:-1], coord[1:], value):
                print(gene_chrom, start, end, val, sep = '\t', end = '\n', file = f)



    @adi.wraps_modelfunc(rpi.fetch_get_influential_local_peaks, rpi.return_peaks_by_idx,
        fill_kwargs=['peak_idx','tss_distance'])
    def get_influential_local_peaks(self, peak_idx, tss_distance, decay_periods = 5):
        '''
        Returns the `.var` field of the adata, but subset for only peaks within 
        the local chromatin neighborhood of a gene. The local chromatin neighborhood
        is defined by the decay distance parameter for that gene's RP model.

        Parameters
        ----------

        adata : anndata.AnnData
            AnnData object with ATAC features and TSS annotations.
        decay_periods : int > 0, default = 5
            Return peaks that are within `decay_periods*upstream_decay_distance` upstream
            of gene and `decay_periods*downstream_decay_distance` downstream of gene,
            where upstream and downstream decay distances are given by the parameters
            of the RP model.

        Returns
        -------

        pd.DataFrame : 
            
            subset from `adata.var` to include only features/peaks within
            the gene's local chromatin neighborhood. This function adds two columns:

            `distance_to_TSS` : int
                Distance, in base pairs, from the gene's TSS
            `is_upstream` : boolean
                If peak is upstream or downstream of gene

        '''

        assert isinstance(decay_periods, (int, float)) and decay_periods > 0

        downstream_mask = (tss_distance >= 0) \
                & (tss_distance < (decay_periods * 1e3 * self._params['downstream_decay']))

        upstream_mask = (tss_distance < 0) \
                & (np.abs(tss_distance) < (decay_periods * 1e3 * self._params['upstream_decay']))

        combined_mask = upstream_mask | downstream_mask

        return peak_idx[combined_mask], tss_distance[combined_mask]