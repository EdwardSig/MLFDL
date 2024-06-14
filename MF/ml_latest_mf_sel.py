import json

import numpy as np
import torch

import global_config
from dataloader import BaseImplicitBCELossDataLoader, MLExplicitDataLoader
from evaluate import ExplicitTestManager
from models import BiasFactMFExplicit
from train import ExplicitTrainManager
from utils import merge_dict, _show_me_a_list_func, _mean_merge_dict_func, show_me_all_the_fucking_result

DEVICE: torch.device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

MODEL_CONFIG: dict = {
    'env_num': 2,
    'factor_num': 48,
    'cdea_hidden': 64,
    'drop_rate': 0.3,
    'reg_only_embed': True,
    'reg_env_embed': False
}

TRAIN_CONFIG: dict = {
    "batch_size": 512,
    "epochs": 300,
    "cluster_interval": 5,
    "evaluate_interval": 10,
    "lr": 0.005,
    "invariant_coe": 3.2135,
    "env_aware_coe": 6,
    "env_coe": 8,
    "L2_coe": 4,
    "L1_coe": 0.5,
    "alpha": 2,
    "use_class_re_weight": False,
    "use_recommend_re_weight": False,
    "test_begin_epoch": 0,
    "begin_cluster_epoch": None,
    "stop_cluster_epoch": None
}

EVALUATE_CONFIG: dict = {
    'top_k_list': [20, 50],
    'test_batch_size': 1024,
    'eval_k': 50,
    'eval_metric': 'recall'
}

RANDOM_SEED_LIST = [123456789]

DATASET_PATH = 'ml-latest-mf-sel/'
METRIC_LIST = ['ndcg', 'recall', 'precision']


def main(
        device: torch.device,
        model_config: dict,
        train_config: dict,
        evaluate_config: dict,
        data_loader: BaseImplicitBCELossDataLoader,
        random_seed: int,
        silent: bool = False,
        auto: bool = False,
):
    torch.manual_seed(random_seed)
    torch.cuda.manual_seed(random_seed)
    torch.cuda.manual_seed_all(random_seed)
    np.random.seed(random_seed)

    model: BiasFactMFExplicit = BiasFactMFExplicit(
        user_num=data_loader.user_num,
        item_num=data_loader.item_num,
        env_num=model_config['env_num'],
        factor_num=model_config['factor_num'],
        drop_rate=model_config['drop_rate'],
        reg_only_embed=model_config['reg_only_embed'],
        reg_env_embed=model_config['reg_env_embed'])
    model = model.to(device)

    train_tensor: torch.LongTensor = torch.LongTensor(
        data_loader.train_data_np).to(device)
    test_tensor: torch.LongTensor = torch.LongTensor(
        data_loader.test_data_np).to(device)

    print(train_tensor.shape)
    assert train_tensor.shape[1] == 3

    test_tensor_data = {
        'users_tensor': test_tensor[:, 0],
        'items_tensor': test_tensor[:, 1],
    }

    evaluator: ExplicitTestManager = ExplicitTestManager(
        model=model,
        data_loader=data_loader,
        test_tensor_data=test_tensor_data,
        test_batch_size=evaluate_config['test_batch_size'],
        top_k_list=evaluate_config['top_k_list'],
        use_item_pool=False)

    train_manager: ExplicitTrainManager = ExplicitTrainManager(
        model=model,
        evaluator=evaluator,
        training_data=train_tensor,
        env_label=torch.LongTensor(data_loader.train_df['sel_label'].tolist()),
        device=device,
        batch_size=train_config['batch_size'],
        epochs=train_config['epochs'],
        cluster_interval=train_config['cluster_interval'],
        evaluate_interval=train_config['evaluate_interval'],
        lr=train_config['lr'],
        invariant_coe=train_config['invariant_coe'],
        env_aware_coe=train_config['env_aware_coe'],
        env_coe=train_config['env_coe'],
        L2_coe=train_config['L2_coe'],
        L1_coe=train_config['L1_coe'],
        alpha=train_config['alpha'],
        use_class_re_weight=train_config['use_class_re_weight'],
        test_begin_epoch=train_config['test_begin_epoch'],
        begin_cluster_epoch=train_config['begin_cluster_epoch'],
        stop_cluster_epoch=train_config['stop_cluster_epoch'],
        use_recommend_re_weight=train_config['use_recommend_re_weight'])

    train_tuple, test_tuple, cluster_tuple = train_manager.train(silent=silent,
                                                                 auto=auto)

    test_result_list = test_tuple[0]
    result_merged_by_metric: dict = merge_dict(test_result_list,
                                               _show_me_a_list_func)

    eval_k: int = evaluate_config['eval_k']
    eval_metric: str = evaluate_config['eval_metric']

    stand_result: np.array = np.array(
        merge_dict(result_merged_by_metric[eval_metric],
                   _show_me_a_list_func)[eval_k])

    best_performance = np.max(stand_result)
    best_indexes: list = np.where(stand_result == best_performance)[0].tolist()
    fucking_result: dict = show_me_all_the_fucking_result(
        result_merged_by_metric, METRIC_LIST, evaluate_config['top_k_list'],
        best_indexes[0])

    if not auto:
        print('Best {}@{}:'.format(eval_metric, eval_k), best_performance,
              best_indexes)

    return best_performance, best_indexes, fucking_result


if __name__ == '__main__':
    loader: MLExplicitDataLoader = MLExplicitDataLoader(
        dataset_path=global_config.CODE_OCEAN_DATASET_PATH + DATASET_PATH,
        file_name=['train.csv', 'test.csv'],
        device=DEVICE,
        has_item_pool_file=False)

    best_metric_perform_list: list = []
    all_metric_results_list: list = []

    for seed in RANDOM_SEED_LIST:
        print('Begin seed:', seed)
        best_perform, _, all_metric_result = main(
            device=DEVICE,
            model_config=MODEL_CONFIG,
            train_config=TRAIN_CONFIG,
            evaluate_config=EVALUATE_CONFIG,
            data_loader=loader,
            random_seed=seed)

        best_metric_perform_list.append(best_perform)
        all_metric_results_list.append(all_metric_result)

        merged_all_metric: dict = merge_dict(all_metric_results_list,
                                             _mean_merge_dict_func)

        merged_all_metric_str: str = json.dumps(merged_all_metric, indent=4)
        print(merged_all_metric_str)

        print('Best perform mean:', np.mean(best_metric_perform_list))
        print('Best perform var:', np.var(best_metric_perform_list))
        print('Best perform std:', np.std(best_metric_perform_list))
        print('Random seed list:', RANDOM_SEED_LIST)
        print('=' * 50)
