
import itertools
import pickle 
from pathlib import Path
from fge import Dataset, ModelBuilder, TreeBuilder
from tqdm import tqdm
import argparse

def experiment(seed, ds, data_folder, exps, infos=False):
    model_kwargs = {
        'titanic': dict(eta=0.1, max_depth=8, subsample=1.0, seed=seed, num_rounds=100),
        'adult': dict(eta=0.3, max_depth=8, subsample=1.0, seed=seed, num_rounds=200),
        'california': dict(eta=0.3, max_depth=8, subsample=1.0, seed=seed, num_rounds=200),
        'boston': dict(eta=0.1, max_depth=8, subsample=1.0, seed=seed, num_rounds=200)
    }
    dataset = Dataset(dataset_name=ds, data_folder=data_folder, seed=seed)
    model_builder = ModelBuilder()
    results = model_builder.train(dataset, **model_kwargs[ds])
    performance = results['score']
    model = results['model']
    tree_builder = TreeBuilder(model, dataset, original_score=performance)
    siv = tree_builder.shap_interaction_values(group_id=None)
    trees = {}
    build_infos = {}
    logger = tqdm(desc=f'Processing {ds}', total=len(exps))
    for e in exps:
        exp_name = '_'.join(map(lambda x: str(x), e))
        score_method, n_select_scores, n_select_gain, nodes_to_run_method, filter_method = e
        tree = tree_builder.build(
            score_method=score_method, 
            shap_interactions=siv, 
            n_select_scores=n_select_scores,  # select nodes_to_run & keys to filter 
            n_select_gain=n_select_gain, 
            max_iter=None,
            nodes_to_run_method=nodes_to_run_method,  # random / sort / full
            filter_method=filter_method,  # random / sort / prob
            rt_only_best=True,
            verbose=False
        )
        trees[exp_name] = tree
        if infos:
            build_infos[exp_name] = tree_builder.infos
        logger.update(1)
        logger.set_postfix_str(f'{exp_name}')

    if infos:
        res = {
            'dataset': dataset, 'model': model,'siv': siv, 'performance': performance,'trees': trees, 
            'explainer': tree_builder.explainer, 'build_infos': build_infos
        }
    else:
        res = {
            'dataset': dataset,'model': model,'siv': siv, 'trees': trees,
            'explainer': tree_builder.explainer
        }
    logger.close()
    return res

def main(infos=False):
    seed = 8
    datasets = ['titanic', 'adult', 'boston', 'california']

    score_method_list = ['abs', 'abs_interaction', 'ratio']
    n_select_scores_list = [3, 5]
    n_select_gain_list = [3, 5]
    nodes_to_run_method_list = ['random', 'sort', 'full']
    filter_method_list = ['random','sort', 'prob']
    exps = list(itertools.product(
        score_method_list, 
        n_select_scores_list,
        n_select_gain_list, 
        nodes_to_run_method_list, 
        filter_method_list
    ))

    data_folder = './data'
    cache_folder = Path('./cache').resolve()
    if infos:
        cache_folder = cache_folder / 'infos'
    if not cache_folder.exists():
        cache_folder.mkdir(parents=True)
    for ds in datasets:
        print(f'Running Experiment in {ds}')
        res = experiment(seed, ds, data_folder, exps, infos=infos)
        with (cache_folder / f'{ds}.pickle').open('wb') as file:
            pickle.dump(res, file)

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--infos', action='store_true')
    args = parser.parse_args()
    main(infos=args.infos)