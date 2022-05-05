import warnings

# Disable warning

warnings.filterwarnings("ignore", category=DeprecationWarning)
warnings.filterwarnings("ignore", category=UserWarning)

from src_pipelines import run_king_clfs_features, run_king_clfs, run_king_cluster, run_king_freqitem

PIPELINE_NAME = 'king'

def loop_king_once():
    base_dir = ''

    # TODO step1
    print('generate features')
    run_king_clfs_features.get_clf_dat(base_dir, PIPELINE_NAME)

    # # #TODO step2
    print('generating classifiers')
    run_king_clfs.run_clfs(base_dir, thread_count=4)

    # #TODO step 3
    print('predicting tweets')
    read_fn = ''
    df_pred = run_king_cluster.run_prediction(base_dir, read_fn)

    print('clustering')
    # TODO step 4 clustering
    run_king_cluster.run_clustering_hdbscan(base_dir, df_pred)

    # TODO step 5 generate frequent items
    run_king_freqitem.gen_freqitem(base_dir)
    return


if __name__ == '__main__':
    # debug()
    loop_king_once()
    pass
