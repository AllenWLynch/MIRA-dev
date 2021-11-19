
import anndata
import numpy as np
import pandas as pd
import logging
import mira.adata_interface.core as adi
logger = logging.getLogger(__name__)


def wide_view():
    from IPython.core.display import display, HTML
    display(HTML("<style>.container { width:100% !important; }</style>"))
    

def make_joint_representation(
    adata1, adata2,
    adata1_key = 'X_umap_features',
    adata2_key = 'X_umap_features',
    key_added = 'X_joint_umap_features'
):

    obs_1, obs_2 = adata1.obs_names.values, adata2.obs_names.values

    shared_cells = np.intersect1d(obs_1, obs_2)

    num_shared_cells = len(shared_cells)
    if num_shared_cells == 0:
        raise ValueError('No cells/obs are shared between these two datasets. Make sure .obs_names is formatted identically between datasets.')

    if num_shared_cells < len(obs_1) or num_shared_cells < len(obs_2):
        logger.warn('Some cells are not shared between views. Returned adatas will be subset copies')

    total_cells = num_shared_cells + len(obs_1) + len(obs_2) - 2*num_shared_cells

    logger.info('{} out of {} cells shared between datasets ({}%).'.format(
        str(num_shared_cells), str(total_cells), str(int(num_shared_cells/total_cells * 100))
    ))

    adata1 = adata1[shared_cells].copy()
    adata2 = adata2[shared_cells].copy()

    joint_representation = np.hstack([
        adata1.obsm[adata1_key], adata2.obsm[adata2_key]
    ])

    adata1.obsm[key_added] = joint_representation
    adata2.obsm[key_added] = joint_representation
    
    logger.info('Key added to obsm: {}'.format(key_added))

    return adata1, adata2


def mask_non_expressed_factors(atac_adata,*, expressed_genes, factor_type = 'motifs'):
    
    metadata, _ = adi.get_factor_meta(None, atac_adata, 
        factor_type = factor_type, mask_factors = False)
    
    assert(isinstance(expressed_genes, (list, np.ndarray, pd.Index)))

    factor_mask = [
        factor['parsed_name'] in expressed_genes
        for factor in metadata
    ]

    adi.add_factor_mask(None, atac_adata, factor_mask, factor_type = factor_type)
    
    logger.info('Found {} factors in expression data.'.format(str(np.array(factor_mask).sum())))


def get_factor_meta(atac_adata, factor_type = 'motifs', mask_factors = False):
    return adi.get_factor_meta(None, atac_adata, factor_type = factor_type, 
            mask_factors = mask_factors)[0]