import itertools
import json
import math
import sys
import time

from recbole.model.loss import EmbLoss
from torch import nn
import torch
import numpy as np
from colorama import Fore, Back, Style

from dataloader import ImplicitBCELossDataLoaderStaticPopularity
from evaluate import ImplicitTestManager, ExplicitTestManager
from models import GeneralDebiasImplicitRecommender, BasicRecommender, GeneralDebiasExplicitRecommender, \
    BasicExplicitRecommender, MFExplicit
from utils import mini_batch, merge_dict, _mean_merge_dict_func, transfer_loss_dict_to_line_str, _show_me_a_list_func


class ImplicitTrainManager:
    def __init__(
            self, model: GeneralDebiasImplicitRecommender, evaluator: ImplicitTestManager,
            device: torch.device,
            training_data: torch.Tensor, env_label: torch.Tensor, batch_size: int,
            epochs: int, cluster_interval: int, evaluate_interval: int, lr: float,
            invariant_coe: float, env_aware_coe: float, env_coe: float, L2_coe: float, L1_coe: float,
            alpha: float = None, use_class_re_weight: bool = False, test_begin_epoch: int = 0,
            begin_cluster_epoch: int = None, stop_cluster_epoch: int = None, cluster_use_random_sort: bool = True,
            use_recommend_re_weight: bool = True
    ):
        self.model: GeneralDebiasImplicitRecommender = model
        self.evaluator: ImplicitTestManager = evaluator
        self.envs_num: int = self.model.env_num
        self.device: torch.device = device
        self.users_tensor: torch.Tensor = training_data[:, 0]
        self.items_tensor: torch.Tensor = training_data[:, 1]
        self.scores_tensor: torch.Tensor = training_data[:, 2].float()
        self.user_positive_interaction = evaluator.data_loader.user_positive_interaction
        self.envs: torch.LongTensor = env_label
        self.envs = self.envs.to(device)
        self.cluster_interval: int = cluster_interval
        self.evaluate_interval: int = evaluate_interval
        self.batch_size: int = batch_size
        self.epochs: int = epochs
        self.optimizer: torch.optim.Adam = torch.optim.Adam(model.parameters(), lr=lr)
        self.recommend_loss_type = nn.BCELoss
        self.cluster_distance_func = nn.BCELoss(reduction='none')
        self.env_loss_type = nn.NLLLoss

        self.invariant_coe = invariant_coe
        self.env_aware_coe = env_aware_coe
        self.env_coe = env_coe
        self.L2_coe = L2_coe
        self.L1_coe = L1_coe

        self.epoch_cnt: int = 0

        self.batch_num = math.ceil(training_data.shape[0] / batch_size)

        self.each_env_count = dict()

        if alpha is None:
            self.alpha = 0.
            self.update_alpha = True
        else:
            self.alpha = alpha
            self.update_alpha = False

        self.use_class_re_weight: bool = use_class_re_weight
        self.use_recommend_re_weight: bool = use_recommend_re_weight
        self.sample_weights: torch.Tensor = torch.Tensor(np.zeros(training_data.shape[0])).to(device)
        self.class_weights: torch.Tensor = torch.Tensor(np.zeros(self.envs_num)).to(device)

        self.test_begin_epoch: int = test_begin_epoch

        self.begin_cluster_epoch: int = begin_cluster_epoch
        self.stop_cluster_epoch: int = stop_cluster_epoch

        self.eps_random_tensor: torch.Tensor = self._init_eps().to(self.device)

        self.cluster_use_random_sort: bool = cluster_use_random_sort

        self.const_env_tensor_list: list = []

        # 为每个batch创建相关的purchase_vec
        item_num = evaluator.data_loader.item_num
        self.batch_cdae_list = []
        for i in range(self.batch_num):
            batch_users_tensor = self.users_tensor[i * batch_size:(i + 1) * batch_size]
            batch_items_tensor = self.items_tensor[i * batch_size:(i + 1) * batch_size]
            batch_scores_tensor = self.scores_tensor[i * batch_size:(i + 1) * batch_size]
            batch_envs_tensor = self.envs[i * batch_size:(i + 1) * batch_size]

            users_unique_tensor = batch_users_tensor.unique()
            purchase_tensor = torch.zeros(size=(len(users_unique_tensor), item_num))
            purchase_tensor = purchase_tensor.to(self.device)
            cdae_envs_id = torch.zeros(size=(len(users_unique_tensor), item_num), dtype=torch.long)
            cdae_envs_id = cdae_envs_id.to(self.device)
            cdae_envs_weight = torch.zeros(size=(len(users_unique_tensor), item_num))
            cdae_envs_weight = cdae_envs_weight.to(self.device)

            for pidx, uid_tensor in enumerate(users_unique_tensor):
                uid = uid_tensor.item()
                user_idxs = (batch_users_tensor == uid).nonzero().squeeze()
                item_ids = batch_items_tensor[user_idxs]
                purchase_tensor[pidx, item_ids] = batch_scores_tensor[user_idxs]
                cdae_envs_id[pidx, item_ids] = batch_envs_tensor[user_idxs]
                cdae_envs_weight[pidx, item_ids] = 1
            self.batch_cdae_list.append((users_unique_tensor, cdae_envs_id, cdae_envs_weight, purchase_tensor))

        for env in range(self.envs_num):
            envs_tensor: torch.Tensor = torch.LongTensor(np.full(training_data.shape[0], env, dtype=int))
            envs_tensor = envs_tensor.to(self.device)
            self.const_env_tensor_list.append(envs_tensor)

    def _init_eps(self):
        base_eps = 1e-10
        eps_list: list = [base_eps * (1e-1 ** idx) for idx in range(self.envs_num)]
        temp: torch.Tensor = torch.Tensor(eps_list)
        eps_random_tensor: torch.Tensor = torch.Tensor(list(itertools.permutations(temp)))

        return eps_random_tensor

    def train_a_batch(
            self,
            batch_users_tensor: torch.Tensor,
            batch_items_tensor: torch.Tensor,
            batch_scores_tensor: torch.Tensor,
            batch_envs_tensor: torch.Tensor,
            batch_sample_weights: torch.Tensor,
            alpha,
            batch_index: int
    ) -> dict:

        # 获取不变兴趣特征预测的分数, 可变特征预测的分数, 环境分类器预测的分类结构
        out1, env_aware_score, env_outputs = self.model(
            batch_users_tensor, batch_items_tensor, batch_envs_tensor,
            self.batch_cdae_list[batch_index][0],
            self.batch_cdae_list[batch_index][1],
            self.batch_cdae_list[batch_index][2],
            self.batch_cdae_list[batch_index][3], alpha
        )
        # out2, env_aware_score, env_outputs = self.model(
        #     batch_users_tensor, batch_items_tensor, batch_envs_tensor,
        #     self.batch_cdae_list[batch_index][0],
        #     self.batch_cdae_list[batch_index][1],
        #     self.batch_cdae_list[batch_index][2],
        #     self.batch_cdae_list[batch_index][3] * out1, alpha
        # )

        if self.use_class_re_weight:
            env_loss = self.env_loss_type(reduction='none')
        else:
            env_loss = self.env_loss_type()

        if self.use_recommend_re_weight:
            recommend_loss = self.recommend_loss_type(reduction='none')
        else:
            recommend_loss = self.recommend_loss_type()

        cdae_loss: torch.Tensor = 5 * torch.sum(recommend_loss(out1, self.batch_cdae_list[batch_index][3]))
        env_aware_loss: torch.Tensor = recommend_loss(env_aware_score, batch_scores_tensor)
        envs_loss: torch.Tensor = env_loss(env_outputs, batch_envs_tensor)

        if self.use_class_re_weight:
            envs_loss = torch.mean(envs_loss * batch_sample_weights)

        if self.use_recommend_re_weight:
            cdae_loss = torch.mean(cdae_loss * batch_sample_weights)
            env_aware_loss = torch.mean(env_aware_loss * batch_sample_weights)

        L2_reg: torch.Tensor = self.model.get_L2_reg(batch_users_tensor, batch_items_tensor, batch_envs_tensor)
        L1_reg: torch.Tensor = self.model.get_L1_reg(batch_users_tensor, batch_items_tensor, batch_envs_tensor)

        """
        loss: torch.Tensor = invariant_loss * self.invariant_coe + env_aware_loss * self.env_aware_coe \
                             + envs_loss * self.env_coe + L2_reg * self.L2_coe + L1_reg * self.L1_coe
        """

        loss: torch.Tensor = cdae_loss + env_aware_loss * self.env_aware_coe \
                             + envs_loss * self.env_coe + L2_reg * self.L2_coe + L1_reg * self.L1_coe

        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

        loss_dict: dict = {
            'invariant_loss': float(cdae_loss),
            'env_aware_loss': float(env_aware_loss),
            'envs_loss': float(envs_loss),
            'L2_reg': float(L2_reg),
            'L1_reg': float(L1_reg),
            'loss': float(loss),
        }
        return loss_dict

    def cluster_a_batch(
            self,
            batch_users_tensor: torch.Tensor,
            batch_items_tensor: torch.Tensor,
            batch_scores_tensor: torch.Tensor,
    ) -> torch.Tensor:
        # 此时应该是eval()\
        distances_list: list = []
        for env_idx in range(self.envs_num):
            # const_env_tensor_list存放的是环境的标签，标签数据包含训练集中所有的数据
            envs_tensor: torch.Tensor = self.const_env_tensor_list[env_idx][0:batch_users_tensor.shape[0]]
            # print('envs_tensor:', envs_tensor.shape, envs_tensor)

            # 返回一个环境识别的得分
            cluster_pred: torch.Tensor = self.model.cluster_predict(batch_users_tensor, batch_items_tensor, envs_tensor)
            # print('cluster_pred:', cluster_pred)

            # 计算当前的环境的评分和真实评分之间的距离
            distances: torch.Tensor = self.cluster_distance_func(cluster_pred, batch_scores_tensor)
            # print('distances:', distances)

            distances = distances.reshape(-1, 1)
            # print('distances reshape:', distances)
            distances_list.append(distances)

        # [samples_num, envs_num]
        each_envs_distances: torch.Tensor = torch.cat(distances_list, dim=1)
        # print('each_envs_distances:', each_envs_distances)
        # [samples_num]
        if self.cluster_use_random_sort:
            sort_random_index: np.array = \
                np.random.randint(0, self.eps_random_tensor.shape[0], each_envs_distances.shape[0])
            # random_eps可能代表随机扰动
            random_eps: torch.Tensor = self.eps_random_tensor[sort_random_index]
            each_envs_distances = each_envs_distances + random_eps
        # print('pes_each_envs_distances:', each_envs_distances)
        # print('random_eps:', random_eps)
        new_envs: torch.Tensor = torch.argmin(each_envs_distances, dim=1)
        # print('new_envs:', new_envs)

        return new_envs

    def train_a_epoch(self) -> dict:
        self.model.train()
        loss_dicts_list: list = []

        # batch_scores_tensor/scores_tensor表示训练集中用户评分，评分包含0和1
        for (batch_index, (
                batch_users_tensor, batch_items_tensor, batch_scores_tensor, batch_envs_tensor, batch_sample_weights
        )) \
                in enumerate(mini_batch(self.batch_size, self.users_tensor,
                                        self.items_tensor, self.scores_tensor, self.envs, self.sample_weights)):

            if self.update_alpha:
                p = float(batch_index + (self.epoch_cnt + 1) * self.batch_num) / float((self.epoch_cnt + 1)
                                                                                       * self.batch_num)
                self.alpha = 2. / (1. + np.exp(-10. * p)) - 1.

            loss_dict: dict = self.train_a_batch(
                batch_users_tensor=batch_users_tensor,
                batch_items_tensor=batch_items_tensor,
                batch_scores_tensor=batch_scores_tensor,
                batch_envs_tensor=batch_envs_tensor,
                batch_sample_weights=batch_sample_weights,
                alpha=self.alpha,
                batch_index=batch_index
            )
            loss_dicts_list.append(loss_dict)

        self.epoch_cnt += 1

        mean_loss_dict: dict = merge_dict(loss_dicts_list, _mean_merge_dict_func)

        return mean_loss_dict

    def cluster(self) -> int:
        """
        转换环境标签
        """
        self.model.eval()

        new_env_tensors_list: list = []

        for (batch_index, (batch_users_tensor, batch_items_tensor, batch_scores_tensor)) \
                in enumerate(mini_batch(self.batch_size, self.users_tensor, self.items_tensor, self.scores_tensor)):
            new_env_tensor: torch.Tensor = self.cluster_a_batch(
                batch_users_tensor=batch_users_tensor,
                batch_items_tensor=batch_items_tensor,
                batch_scores_tensor=batch_scores_tensor
            )

            # print(new_env_tensor.shape)
            new_env_tensors_list.append(new_env_tensor)

        all_new_env_tensors: torch.Tensor = torch.cat(new_env_tensors_list, dim=0)
        # print()
        # print(all_new_env_tensors.shape)
        envs_diff: torch.Tensor = (self.envs - all_new_env_tensors) != 0
        diff_num: int = int(torch.sum(envs_diff))
        self.envs = all_new_env_tensors
        return diff_num

    def update_each_env_count(self):
        result_dict: dict = {}
        for env in range(self.envs_num):
            cnt = torch.sum(self.envs == env)
            result_dict[env] = cnt
        self.each_env_count.update(result_dict)

    def stat_envs(self) -> dict:
        """
        看当前数据集中属于某个环境的数据有多少个
        @return:
        """
        result: dict = dict()
        class_rate_np: np.array = np.zeros(self.envs_num)
        for env in range(self.envs_num):
            cnt: int = int(torch.sum(self.envs == env))
            result[env] = cnt
            class_rate_np[env] = min(cnt + 1, self.scores_tensor.shape[0] - 1)

        class_rate_np = class_rate_np / self.scores_tensor.shape[0]
        self.class_weights = torch.Tensor(class_rate_np).to(self.device)
        self.sample_weights = self.class_weights[self.envs]  # 采样权重，其权重数值由class_rate_np得来

        return result

    def train(self, silent: bool = False, auto: bool = False):
        """

        @param silent:
        @param auto:
        @return: (epoch中各损失值, 当前是第几个epoch), (当前测试指标, 当前是第几个epoch), (聚类后环境标签改变的数据有多少个, 每个环境有多少个数据, 当前是第几个epoch)
        """
        print(Fore.GREEN)
        print('=' * 30, 'train started!!!', '=' * 30)
        print(Style.RESET_ALL)

        test_result_list: list = []
        test_epoch_list: list = []

        cluster_diff_num_list: list = []
        cluster_epoch_list: list = []
        envs_cnt_list: list = []

        loss_result_list: list = []
        train_epoch_index_list: list = []

        temp_eval_result: dict = self.evaluator.evaluate()
        test_result_list.append(temp_eval_result)
        test_epoch_list.append(self.epoch_cnt)

        # self.stat_envs()

        if not silent and not auto:
            print(Fore.BLUE)
            print('test at epoch:', self.epoch_cnt)
            print(transfer_loss_dict_to_line_str(temp_eval_result))

        while self.epoch_cnt < self.epochs:
            # 训练数据, 在该方法中存在模型训练的信息
            temp_loss_dict = self.train_a_epoch()
            train_epoch_index_list.append(self.epoch_cnt)
            loss_result_list.append(temp_loss_dict)

            if not silent and not auto:
                print(Fore.GREEN)
                print('train epoch:', self.epoch_cnt)
                print(transfer_loss_dict_to_line_str(temp_loss_dict))

            if (self.epoch_cnt % self.evaluate_interval) == 0 and self.epoch_cnt >= self.test_begin_epoch:
                temp_eval_result: dict = self.evaluator.evaluate()
                test_result_list.append(temp_eval_result)
                test_epoch_list.append(self.epoch_cnt)

                if not silent and not auto:
                    print(Fore.BLUE)
                    print('test at epoch:', self.epoch_cnt)
                    print(transfer_loss_dict_to_line_str(temp_eval_result))
            # sys.exit()

            # if (self.epoch_cnt % self.cluster_interval) == 0:

            #     if (self.begin_cluster_epoch is None or self.begin_cluster_epoch <= self.epoch_cnt) \
            #             and (self.stop_cluster_epoch is None or self.stop_cluster_epoch > self.epoch_cnt):
            #         # 通过聚类改变数据集中的环境分布，并返回聚类类环境标签改变的数量
            #         diff_num: int = self.cluster()
            #         cluster_diff_num_list.append(diff_num)
            #     else:
            #         diff_num: int = 0
            #         cluster_diff_num_list.append(diff_num)

            #     envs_cnt: dict = self.stat_envs()

            #     cluster_epoch_list.append(self.epoch_cnt)
            #     envs_cnt_list.append(envs_cnt)

            #     if not silent and not auto:
            #         print(Fore.CYAN)
            #         print('cluster at epoch:', self.epoch_cnt)
            #         print('diff num:', diff_num)
            #         print(transfer_loss_dict_to_line_str(envs_cnt))
            #         print(Style.RESET_ALL)

        print('=' * 30, 'train finished!!!', '=' * 30)
        return (loss_result_list, train_epoch_index_list), \
            (test_result_list, test_epoch_list), \
            (cluster_diff_num_list, envs_cnt_list, cluster_epoch_list)


class ExplicitTrainManager:
    def __init__(
            self, model: GeneralDebiasExplicitRecommender, evaluator: ExplicitTestManager,
            device: torch.device,
            training_data: torch.Tensor, env_label: torch.Tensor, batch_size: int,
            epochs: int, cluster_interval: int,  evaluate_interval: int, lr: float,
            invariant_coe: float, env_aware_coe: float, env_coe: float, L2_coe: float, L1_coe: float,
            alpha: float = None, use_class_re_weight: bool = False, test_begin_epoch: int = 0,
            begin_cluster_epoch: int = None, stop_cluster_epoch: int = None, cluster_use_random_sort: bool = True,
            use_recommend_re_weight: bool = True
    ):
        self.model: GeneralDebiasExplicitRecommender = model
        self.evaluator: ExplicitTestManager = evaluator
        self.envs_num: int = self.model.env_num
        self.device: torch.device = device
        self.users_tensor: torch.Tensor = training_data[:, 0]
        self.items_tensor: torch.Tensor = training_data[:, 1]
        self.scores_tensor: torch.Tensor = training_data[:, 2].float()
        # TODO: 修改
        # self.user_positive_interaction = evaluator.data_loader.user_positive_interaction
        self.envs: torch.LongTensor = env_label
        self.envs = self.envs.to(device)
        self.cluster_interval: int = cluster_interval
        self.evaluate_interval: int = evaluate_interval
        self.batch_size: int = batch_size
        self.epochs: int = epochs
        self.optimizer: torch.optim.Adam = torch.optim.Adam(model.parameters(), lr=lr)
        self.recommend_loss_type = nn.MSELoss
        # self.cluster_distance_func = nn.MSELoss(reduction='none')
        self.env_loss_type = nn.NLLLoss

        self.invariant_coe = invariant_coe
        self.env_aware_coe = env_aware_coe
        self.env_coe = env_coe
        self.L2_coe = L2_coe
        self.L1_coe = L1_coe

        self.epoch_cnt: int = 0

        self.batch_num = math.ceil(training_data.shape[0] / batch_size)

        self.each_env_count = dict()

        if alpha is None:
            self.alpha = 0.
            self.update_alpha = True
        else:
            self.alpha = alpha
            self.update_alpha = False

        self.use_class_re_weight: bool = use_class_re_weight
        self.use_recommend_re_weight: bool = use_recommend_re_weight
        self.sample_weights: torch.Tensor = torch.Tensor(np.zeros(training_data.shape[0])).to(device)
        self.class_weights: torch.Tensor = torch.Tensor(np.zeros(self.envs_num)).to(device)

        self.test_begin_epoch: int = test_begin_epoch

        self.begin_cluster_epoch: int = begin_cluster_epoch
        self.stop_cluster_epoch: int = stop_cluster_epoch

        self.eps_random_tensor: torch.Tensor = self._init_eps().to(self.device)

        self.cluster_use_random_sort: bool = cluster_use_random_sort

        # self.const_env_tensor_list: list = []
        #
        # for env in range(self.envs_num):
        #     envs_tensor: torch.Tensor = torch.LongTensor(np.full(training_data.shape[0], env, dtype=int))
        #     envs_tensor = envs_tensor.to(self.device)
        #     self.const_env_tensor_list.append(envs_tensor)

    def _init_eps(self):
        base_eps = 1e-10
        eps_list: list = [base_eps * (1e-1 ** idx) for idx in range(self.envs_num)]
        temp: torch.Tensor = torch.Tensor(eps_list)
        eps_random_tensor: torch.Tensor = torch.Tensor(list(itertools.permutations(temp)))

        return eps_random_tensor

    def train_a_batch(
            self,
            batch_users_tensor: torch.Tensor,
            batch_items_tensor: torch.Tensor,
            batch_scores_tensor: torch.Tensor,
            batch_envs_tensor: torch.Tensor,
            batch_sample_weights: torch.Tensor,
            alpha
    ) -> dict:

        # print('embed_env_GMF:', self.model.embed_env_GMF.weight)
        # print('batch_envs_tensor:', batch_envs_tensor)

        # print()
        mf_score, env_aware_score, env_outputs = self.model(
            batch_users_tensor, batch_items_tensor, batch_envs_tensor, alpha
        )

        if self.use_class_re_weight:
            env_loss = self.env_loss_type(reduction='none')
        else:
            env_loss = self.env_loss_type()

        if self.use_recommend_re_weight:
            recommend_loss = self.recommend_loss_type(reduction='none')
        else:
            recommend_loss = self.recommend_loss_type()

        mf_loss: torch.Tensor = recommend_loss(mf_score, batch_scores_tensor)
        env_aware_loss: torch.Tensor = recommend_loss(env_aware_score, batch_scores_tensor)

        # print(invariant_loss, env_aware_loss, batch_sample_weights, sep='\n')

        envs_loss: torch.Tensor = env_loss(env_outputs, batch_envs_tensor)

        if self.use_class_re_weight:
            envs_loss = torch.mean(envs_loss * batch_sample_weights)

        if self.use_recommend_re_weight:
            invariant_loss = torch.mean(mf_loss * batch_sample_weights)
            env_aware_loss = torch.mean(env_aware_loss * batch_sample_weights)

        L2_reg: torch.Tensor = self.model.get_L2_reg(batch_users_tensor, batch_items_tensor, batch_envs_tensor)
        L1_reg: torch.Tensor = self.model.get_L1_reg(batch_users_tensor, batch_items_tensor, batch_envs_tensor)

        """
        loss: torch.Tensor = invariant_loss * self.invariant_coe + env_aware_loss * self.env_aware_coe \
                             + envs_loss * self.env_coe + L2_reg * self.L2_coe + L1_reg * self.L1_coe
        """

        loss: torch.Tensor = mf_loss * self.invariant_coe + env_aware_loss * self.env_aware_coe \
                             + envs_loss * self.env_coe + L2_reg * self.L2_coe + L1_reg * self.L1_coe

        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

        loss_dict: dict = {
            'invariant_loss': float(mf_loss),
            'env_aware_loss': float(env_aware_loss),
            'envs_loss': float(envs_loss),
            'L2_reg': float(L2_reg),
            'L1_reg': float(L1_reg),
            'loss': float(loss),
        }
        return loss_dict

    def cluster_a_batch(
            self,
            batch_users_tensor: torch.Tensor,
            batch_items_tensor: torch.Tensor,
            batch_scores_tensor: torch.Tensor,
    ) -> torch.Tensor:
        # 此时应该是eval()\
        distances_list: list = []
        for env_idx in range(self.envs_num):
            envs_tensor: torch.Tensor = self.const_env_tensor_list[env_idx][0:batch_users_tensor.shape[0]]
            # print('envs_tensor:', envs_tensor.shape, envs_tensor)
            cluster_pred: torch.Tensor = self.model.cluster_predict(batch_users_tensor, batch_items_tensor, envs_tensor)
            # print('cluster_pred:', cluster_pred)
            distances: torch.Tensor = self.cluster_distance_func(cluster_pred, batch_scores_tensor)
            # print('distances:', distances)
            distances = distances.reshape(-1, 1)
            # print('distances reshape:', distances)
            distances_list.append(distances)

        # [samples_num, envs_num]
        each_envs_distances: torch.Tensor = torch.cat(distances_list, dim=1)
        # print('each_envs_distances:', each_envs_distances)
        # [samples_num]
        if self.cluster_use_random_sort:
            sort_random_index: np.array = \
                np.random.randint(0, self.eps_random_tensor.shape[0], each_envs_distances.shape[0])
            random_eps: torch.Tensor = self.eps_random_tensor[sort_random_index]
            each_envs_distances = each_envs_distances + random_eps
        # print('pes_each_envs_distances:', each_envs_distances)
        # print('random_eps:', random_eps)
        new_envs: torch.Tensor = torch.argmin(each_envs_distances, dim=1)
        # print('new_envs:', new_envs)

        return new_envs

    def train_a_epoch(self) -> dict:
        self.model.train()
        loss_dicts_list: list = []

        for (batch_index, (
                batch_users_tensor, batch_items_tensor, batch_scores_tensor, batch_envs_tensor, batch_sample_weights
        )) \
                in enumerate(mini_batch(self.batch_size, self.users_tensor,
                                        self.items_tensor, self.scores_tensor, self.envs, self.sample_weights)):

            if self.update_alpha:
                p = float(batch_index + (self.epoch_cnt + 1) * self.batch_num) / float((self.epoch_cnt + 1)
                                                                                       * self.batch_num)
                self.alpha = 2. / (1. + np.exp(-10. * p)) - 1.

            loss_dict: dict = self.train_a_batch(
                batch_users_tensor=batch_users_tensor,
                batch_items_tensor=batch_items_tensor,
                batch_scores_tensor=batch_scores_tensor,
                batch_envs_tensor=batch_envs_tensor,
                batch_sample_weights=batch_sample_weights,
                alpha=self.alpha
            )
            loss_dicts_list.append(loss_dict)

        self.epoch_cnt += 1

        mean_loss_dict: dict = merge_dict(loss_dicts_list, _mean_merge_dict_func)

        return mean_loss_dict

    def cluster(self) -> int:
        self.model.eval()

        new_env_tensors_list: list = []

        for (batch_index, (batch_users_tensor, batch_items_tensor, batch_scores_tensor)) \
                in enumerate(mini_batch(self.batch_size, self.users_tensor, self.items_tensor, self.scores_tensor)):

            new_env_tensor: torch.Tensor = self.cluster_a_batch(
                batch_users_tensor=batch_users_tensor,
                batch_items_tensor=batch_items_tensor,
                batch_scores_tensor=batch_scores_tensor
            )

            # print(new_env_tensor.shape)

            new_env_tensors_list.append(new_env_tensor)

        all_new_env_tensors: torch.Tensor = torch.cat(new_env_tensors_list, dim=0)
        # print()
        # print(all_new_env_tensors.shape)
        envs_diff: torch.Tensor = (self.envs - all_new_env_tensors) != 0
        diff_num: int = int(torch.sum(envs_diff))
        self.envs = all_new_env_tensors
        return diff_num

    def update_each_env_count(self):
        result_dict: dict = {}
        for env in range(self.envs_num):
            cnt = torch.sum(self.envs == env)
            result_dict[env] = cnt
        self.each_env_count.update(result_dict)

    def stat_envs(self) -> dict:
        result: dict = dict()
        class_rate_np: np.array = np.zeros(self.envs_num)
        for env in range(self.envs_num):
            cnt: int = int(torch.sum(self.envs == env))
            result[env] = cnt
            class_rate_np[env] = min(cnt + 1, self.scores_tensor.shape[0] - 1)

        class_rate_np = class_rate_np / self.scores_tensor.shape[0]
        self.class_weights = torch.Tensor(class_rate_np).to(self.device)
        self.sample_weights = self.class_weights[self.envs]

        return result

    def train(self, silent: bool = False, auto: bool = False):
        """

        @param silent:
        @param auto:
        @return: (epoch中各损失值, 当前是第几个epoch), (当前测试指标, 当前是第几个epoch), (聚类后环境标签改变的数据有多少个, 每个环境有多少个数据, 当前是第几个epoch)
        """
        print(Fore.GREEN)
        print('=' * 30, 'train started!!!', '=' * 30)
        print(Style.RESET_ALL)

        test_result_list: list = []
        test_epoch_list: list = []

        cluster_diff_num_list: list = []
        cluster_epoch_list: list = []
        envs_cnt_list: list = []

        loss_result_list: list = []
        train_epoch_index_list: list = []

        temp_eval_result: dict = self.evaluator.evaluate()
        test_result_list.append(temp_eval_result)
        test_epoch_list.append(self.epoch_cnt)

        # self.stat_envs()

        if not silent and not auto:
            print(Fore.BLUE)
            print('test at epoch:', self.epoch_cnt)
            print(transfer_loss_dict_to_line_str(temp_eval_result))

        while self.epoch_cnt < self.epochs:
            # 训练数据, 在该方法中存在模型训练的信息
            temp_loss_dict = self.train_a_epoch()
            train_epoch_index_list.append(self.epoch_cnt)
            loss_result_list.append(temp_loss_dict)

            if not silent and not auto:
                print(Fore.GREEN)
                print('train epoch:', self.epoch_cnt)
                print(transfer_loss_dict_to_line_str(temp_loss_dict))

            if (self.epoch_cnt % self.evaluate_interval) == 0 and self.epoch_cnt >= self.test_begin_epoch:
                temp_eval_result: dict = self.evaluator.evaluate()
                test_result_list.append(temp_eval_result)
                test_epoch_list.append(self.epoch_cnt)

                if not silent and not auto:
                    print(Fore.BLUE)
                    print('test at epoch:', self.epoch_cnt)
                    print(transfer_loss_dict_to_line_str(temp_eval_result))

        print('=' * 30, 'train finished!!!', '=' * 30)
        return (loss_result_list, train_epoch_index_list), \
            (test_result_list, test_epoch_list), \
            (cluster_diff_num_list, envs_cnt_list, cluster_epoch_list)


class ExplicitTrainManagerNone:
    def __init__(
            self, model: MFExplicit, evaluator: ExplicitTestManager,
            device: torch.device,
            training_data: torch.Tensor, batch_size: int,
            epochs: int, cluster_interval: int,  evaluate_interval: int, lr: float,
            invariant_coe: float, env_aware_coe: float, env_coe: float, L2_coe: float, L1_coe: float,
            alpha: float = None, use_class_re_weight: bool = False, test_begin_epoch: int = 0,
            begin_cluster_epoch: int = None, stop_cluster_epoch: int = None, cluster_use_random_sort: bool = True,
            use_recommend_re_weight: bool = True
    ):
        self.model: MFExplicit = model
        self.evaluator: ExplicitTestManager = evaluator
        # self.envs_num: int = self.model.env_num
        self.device: torch.device = device
        self.users_tensor: torch.Tensor = training_data[:, 0]
        self.items_tensor: torch.Tensor = training_data[:, 1]
        self.scores_tensor: torch.Tensor = training_data[:, 2].float()
        # self.envs: torch.LongTensor = torch.LongTensor(np.random.randint(0, self.envs_num, training_data.shape[0]))
        # self.envs = self.envs.to(device)
        # self.cluster_interval: int = cluster_interval
        self.evaluate_interval: int = evaluate_interval
        self.batch_size: int = batch_size
        self.epochs: int = epochs
        self.optimizer: torch.optim.Adam = torch.optim.Adam(model.parameters(), lr=lr)
        self.recommend_loss_type = nn.MSELoss
        self.cluster_distance_func = nn.MSELoss(reduction='none')
        # self.env_loss_type = nn.NLLLoss

        self.invariant_coe = invariant_coe
        # self.env_aware_coe = env_aware_coe
        # self.env_coe = env_coe
        self.L2_coe = L2_coe
        self.L1_coe = L1_coe

        self.epoch_cnt: int = 0

        self.batch_num = math.ceil(training_data.shape[0] / batch_size)

        self.each_env_count = dict()

        if alpha is None:
            self.alpha = 0.
            self.update_alpha = True
        else:
            self.alpha = alpha
            self.update_alpha = False

        # self.use_class_re_weight: bool = use_class_re_weight
        # self.use_recommend_re_weight: bool = use_recommend_re_weight
        # self.sample_weights: torch.Tensor = torch.Tensor(np.zeros(training_data.shape[0])).to(device)
        # self.class_weights: torch.Tensor = torch.Tensor(np.zeros(self.envs_num)).to(device)

        self.test_begin_epoch: int = test_begin_epoch

    def _init_eps(self):
        base_eps = 1e-10
        eps_list: list = [base_eps * (1e-1 ** idx) for idx in range(self.envs_num)]
        temp: torch.Tensor = torch.Tensor(eps_list)
        eps_random_tensor: torch.Tensor = torch.Tensor(list(itertools.permutations(temp)))

        return eps_random_tensor

    def train_a_batch(
            self,
            batch_users_tensor: torch.Tensor,
            batch_items_tensor: torch.Tensor,
            batch_scores_tensor: torch.Tensor,
            alpha
    ) -> dict:

        # print('embed_env_GMF:', self.model.embed_env_GMF.weight)
        # print('batch_envs_tensor:', batch_envs_tensor)

        # print()
        predict_score = self.model(
            batch_users_tensor, batch_items_tensor)

        # if self.use_recommend_re_weight:
        #     recommend_loss = self.recommend_loss_type(reduction='none')
        # else:
        recommend_loss = self.recommend_loss_type()

        score_loss: torch.Tensor = recommend_loss(predict_score, batch_scores_tensor)

        L2_reg: torch.Tensor = self.model.get_L2_reg(batch_users_tensor, batch_items_tensor)
        L1_reg: torch.Tensor = self.model.get_L1_reg(batch_users_tensor, batch_items_tensor)

        """
        loss: torch.Tensor = invariant_loss * self.invariant_coe + env_aware_loss * self.env_aware_coe \
                             + envs_loss * self.env_coe + L2_reg * self.L2_coe + L1_reg * self.L1_coe
        """

        loss: torch.Tensor = score_loss * self.invariant_coe + L2_reg * self.L2_coe + L1_reg * self.L1_coe

        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

        loss_dict: dict = {
            'score_loss': float(score_loss),
            'L2_reg': float(L2_reg),
            'L1_reg': float(L1_reg),
            'loss': float(loss),
        }
        return loss_dict

    def cluster_a_batch(
            self,
            batch_users_tensor: torch.Tensor,
            batch_items_tensor: torch.Tensor,
            batch_scores_tensor: torch.Tensor,
    ) -> torch.Tensor:
        # 此时应该是eval()\
        distances_list: list = []
        for env_idx in range(self.envs_num):
            envs_tensor: torch.Tensor = self.const_env_tensor_list[env_idx][0:batch_users_tensor.shape[0]]
            # print('envs_tensor:', envs_tensor.shape, envs_tensor)
            cluster_pred: torch.Tensor = self.model.cluster_predict(batch_users_tensor, batch_items_tensor, envs_tensor)
            # print('cluster_pred:', cluster_pred)
            distances: torch.Tensor = self.cluster_distance_func(cluster_pred, batch_scores_tensor)
            # print('distances:', distances)
            distances = distances.reshape(-1, 1)
            # print('distances reshape:', distances)
            distances_list.append(distances)

        # [samples_num, envs_num]
        each_envs_distances: torch.Tensor = torch.cat(distances_list, dim=1)
        # print('each_envs_distances:', each_envs_distances)
        # [samples_num]
        if self.cluster_use_random_sort:
            sort_random_index: np.array = \
                np.random.randint(0, self.eps_random_tensor.shape[0], each_envs_distances.shape[0])
            random_eps: torch.Tensor = self.eps_random_tensor[sort_random_index]
            each_envs_distances = each_envs_distances + random_eps
        # print('pes_each_envs_distances:', each_envs_distances)
        # print('random_eps:', random_eps)
        new_envs: torch.Tensor = torch.argmin(each_envs_distances, dim=1)
        # print('new_envs:', new_envs)

        return new_envs

    def train_a_epoch(self) -> dict:
        self.model.train()
        loss_dicts_list: list = []

        for (batch_index, (batch_users_tensor, batch_items_tensor, batch_scores_tensor)) \
                in enumerate(mini_batch(self.batch_size, self.users_tensor, self.items_tensor, self.scores_tensor)):

            if self.update_alpha:
                p = float(batch_index + (self.epoch_cnt + 1) * self.batch_num) / float((self.epoch_cnt + 1)
                                                                                       * self.batch_num)
                self.alpha = 2. / (1. + np.exp(-10. * p)) - 1.

            loss_dict: dict = self.train_a_batch(
                batch_users_tensor=batch_users_tensor,
                batch_items_tensor=batch_items_tensor,
                batch_scores_tensor=batch_scores_tensor,
                alpha=self.alpha
            )
            loss_dicts_list.append(loss_dict)

        self.epoch_cnt += 1

        mean_loss_dict: dict = merge_dict(loss_dicts_list, _mean_merge_dict_func)

        return mean_loss_dict

    def cluster(self) -> int:
        self.model.eval()

        new_env_tensors_list: list = []

        for (batch_index, (batch_users_tensor, batch_items_tensor, batch_scores_tensor)) \
                in enumerate(mini_batch(self.batch_size, self.users_tensor, self.items_tensor, self.scores_tensor)):

            new_env_tensor: torch.Tensor = self.cluster_a_batch(
                batch_users_tensor=batch_users_tensor,
                batch_items_tensor=batch_items_tensor,
                batch_scores_tensor=batch_scores_tensor
            )

            # print(new_env_tensor.shape)

            new_env_tensors_list.append(new_env_tensor)

        all_new_env_tensors: torch.Tensor = torch.cat(new_env_tensors_list, dim=0)
        # print()
        # print(all_new_env_tensors.shape)
        envs_diff: torch.Tensor = (self.envs - all_new_env_tensors) != 0
        diff_num: int = int(torch.sum(envs_diff))
        self.envs = all_new_env_tensors
        return diff_num

    def update_each_env_count(self):
        result_dict: dict = {}
        for env in range(self.envs_num):
            cnt = torch.sum(self.envs == env)
            result_dict[env] = cnt
        self.each_env_count.update(result_dict)

    def stat_envs(self) -> dict:
        result: dict = dict()
        class_rate_np: np.array = np.zeros(self.envs_num)
        for env in range(self.envs_num):
            cnt: int = int(torch.sum(self.envs == env))
            result[env] = cnt
            class_rate_np[env] = min(cnt + 1, self.scores_tensor.shape[0] - 1)

        class_rate_np = class_rate_np / self.scores_tensor.shape[0]
        self.class_weights = torch.Tensor(class_rate_np).to(self.device)
        self.sample_weights = self.class_weights[self.envs]

        return result

    def train(self, silent: bool = False, auto: bool = False):
        test_result_list: list = []
        test_epoch_list: list = []

        cluster_diff_num_list: list = []
        cluster_epoch_list: list = []
        envs_cnt_list: list = []

        loss_result_list: list = []
        train_epoch_index_list: list = []

        temp_eval_result: dict = self.evaluator.evaluate()
        test_result_list.append(temp_eval_result)
        test_epoch_list.append(self.epoch_cnt)

        # self.stat_envs()

        if not silent and not auto:
            print('test at epoch:', self.epoch_cnt)
            print(transfer_loss_dict_to_line_str(temp_eval_result))

        while self.epoch_cnt < self.epochs:
            temp_loss_dict = self.train_a_epoch()
            train_epoch_index_list.append(self.epoch_cnt)
            loss_result_list.append(temp_loss_dict)
            if not silent and not auto:
                print('train epoch:', self.epoch_cnt)
                print(transfer_loss_dict_to_line_str(temp_loss_dict))

            if (self.epoch_cnt % self.evaluate_interval) == 0 and self.epoch_cnt >= self.test_begin_epoch:
                temp_eval_result: dict = self.evaluator.evaluate()
                test_result_list.append(temp_eval_result)
                test_epoch_list.append(self.epoch_cnt)

                if not silent and not auto:
                    print('test at epoch:', self.epoch_cnt)
                    print(transfer_loss_dict_to_line_str(temp_eval_result))

        return (loss_result_list, train_epoch_index_list), \
               (test_result_list, test_epoch_list), \
               (cluster_diff_num_list, envs_cnt_list, cluster_epoch_list)


class PDAExplicitTrainManager:
    def __init__(
            self, model: GeneralDebiasExplicitRecommender, evaluator: ExplicitTestManager,
            device: torch.device, training_data: torch.Tensor, batch_size: int,
            epochs: int, evaluate_interval: int, lr: float,
            invariant_coe: float, L2_coe: float, L1_coe: float,
            alpha: float = None, use_class_re_weight: bool = False, test_begin_epoch: int = 0,
            use_recommend_re_weight: bool = True
    ):
        self.model: GeneralDebiasExplicitRecommender = model
        self.evaluator: ExplicitTestManager = evaluator
        self.envs_num: int = self.model.env_num
        self.device: torch.device = device
        self.users_tensor: torch.Tensor = training_data[:, 0]
        self.items_tensor: torch.Tensor = training_data[:, 1]
        self.scores_tensor: torch.Tensor = training_data[:, 2].float()
        self.evaluate_interval: int = evaluate_interval
        self.batch_size: int = batch_size
        self.epochs: int = epochs
        self.optimizer: torch.optim.Adam = torch.optim.Adam(model.parameters(), lr=lr)
        self.recommend_loss_type = nn.MSELoss
        self.reg_loss = EmbLoss()
        # self.cluster_distance_func = nn.MSELoss(reduction='none')

        self.invariant_coe = invariant_coe
        self.L2_coe = L2_coe
        self.L1_coe = L1_coe

        self.epoch_cnt: int = 0

        self.batch_num = math.ceil(training_data.shape[0] / batch_size)

        self.each_env_count = dict()

        if alpha is None:
            self.alpha = 0.
            self.update_alpha = True
        else:
            self.alpha = alpha
            self.update_alpha = False

        self.use_class_re_weight: bool = use_class_re_weight
        self.use_recommend_re_weight: bool = use_recommend_re_weight
        self.sample_weights: torch.Tensor = torch.Tensor(np.zeros(training_data.shape[0])).to(device)

        self.test_begin_epoch: int = test_begin_epoch

        self.eps_random_tensor: torch.Tensor = self._init_eps().to(self.device)

        # self.const_env_tensor_list: list = []
        #
        # for env in range(self.envs_num):
        #     envs_tensor: torch.Tensor = torch.LongTensor(np.full(training_data.shape[0], env, dtype=int))
        #     envs_tensor = envs_tensor.to(self.device)
        #     self.const_env_tensor_list.append(envs_tensor)

    def _init_eps(self):
        base_eps = 1e-10
        eps_list: list = [base_eps * (1e-1 ** idx) for idx in range(self.envs_num)]
        temp: torch.Tensor = torch.Tensor(eps_list)
        eps_random_tensor: torch.Tensor = torch.Tensor(list(itertools.permutations(temp)))

        return eps_random_tensor

    def train_a_batch(
            self,
            batch_users_tensor: torch.Tensor,
            batch_items_tensor: torch.Tensor,
            batch_scores_tensor: torch.Tensor,
    ) -> dict:

        # print('embed_env_GMF:', self.model.embed_env_GMF.weight)
        # print('batch_envs_tensor:', batch_envs_tensor)

        # print()
        pos_score = self.model(batch_users_tensor, batch_items_tensor)

        recommend_loss = self.recommend_loss_type()

        mf_loss: torch.Tensor = recommend_loss(pos_score, batch_scores_tensor)

        # print(invariant_loss, env_aware_loss, batch_sample_weights, sep='\n')
        reg_loss = self.model.reg_loss_func(batch_users_tensor, batch_items_tensor)

        """
        loss: torch.Tensor = invariant_loss * self.invariant_coe + env_aware_loss * self.env_aware_coe \
                             + envs_loss * self.env_coe + L2_reg * self.L2_coe + L1_reg * self.L1_coe
        """

        loss: torch.Tensor = mf_loss + reg_loss

        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

        loss_dict: dict = {
            'invariant_loss': float(mf_loss),
            'loss': float(loss),
        }
        return loss_dict

    def train_a_epoch(self) -> dict:
        self.model.train()
        loss_dicts_list: list = []

        for (batch_index, (
                batch_users_tensor, batch_items_tensor, batch_scores_tensor)) \
                in enumerate(mini_batch(self.batch_size, self.users_tensor, self.items_tensor, self.scores_tensor)):

            if self.update_alpha:
                p = float(batch_index + (self.epoch_cnt + 1) * self.batch_num) / float((self.epoch_cnt + 1)
                                                                                       * self.batch_num)
                self.alpha = 2. / (1. + np.exp(-10. * p)) - 1.

            loss_dict: dict = self.train_a_batch(
                batch_users_tensor=batch_users_tensor,
                batch_items_tensor=batch_items_tensor,
                batch_scores_tensor=batch_scores_tensor)
            loss_dicts_list.append(loss_dict)

        self.epoch_cnt += 1

        mean_loss_dict: dict = merge_dict(loss_dicts_list, _mean_merge_dict_func)

        return mean_loss_dict

    def train(self, silent: bool = False, auto: bool = False):
        """

        @param silent:
        @param auto:
        @return: (epoch中各损失值, 当前是第几个epoch), (当前测试指标, 当前是第几个epoch), (聚类后环境标签改变的数据有多少个, 每个环境有多少个数据, 当前是第几个epoch)
        """
        print(Fore.GREEN)
        print('=' * 30, 'train started!!!', '=' * 30)
        print(Style.RESET_ALL)

        test_result_list: list = []
        test_epoch_list: list = []

        cluster_diff_num_list: list = []
        cluster_epoch_list: list = []
        envs_cnt_list: list = []

        loss_result_list: list = []
        train_epoch_index_list: list = []

        temp_eval_result: dict = self.evaluator.evaluate()
        test_result_list.append(temp_eval_result)
        test_epoch_list.append(self.epoch_cnt)

        # self.stat_envs()

        if not silent and not auto:
            print(Fore.BLUE)
            print('test at epoch:', self.epoch_cnt)
            print(transfer_loss_dict_to_line_str(temp_eval_result))

        while self.epoch_cnt < self.epochs:
            # 训练数据, 在该方法中存在模型训练的信息
            temp_loss_dict = self.train_a_epoch()
            train_epoch_index_list.append(self.epoch_cnt)
            loss_result_list.append(temp_loss_dict)

            if not silent and not auto:
                print(Fore.GREEN)
                print('train epoch:', self.epoch_cnt)
                print(transfer_loss_dict_to_line_str(temp_loss_dict))

            if (self.epoch_cnt % self.evaluate_interval) == 0 and self.epoch_cnt >= self.test_begin_epoch:
                temp_eval_result: dict = self.evaluator.evaluate()
                test_result_list.append(temp_eval_result)
                test_epoch_list.append(self.epoch_cnt)

                if not silent and not auto:
                    print(Fore.BLUE)
                    print('test at epoch:', self.epoch_cnt)
                    print(transfer_loss_dict_to_line_str(temp_eval_result))

        print('=' * 30, 'train finished!!!', '=' * 30)
        return (loss_result_list, train_epoch_index_list), \
            (test_result_list, test_epoch_list), \
            (cluster_diff_num_list, envs_cnt_list, cluster_epoch_list)


class MACRExplicitTrainManager:
    def __init__(
            self, model: GeneralDebiasExplicitRecommender, evaluator: ExplicitTestManager,
            device: torch.device, training_data: torch.Tensor, batch_size: int,
            epochs: int, evaluate_interval: int, lr: float,
            invariant_coe: float, L2_coe: float, L1_coe: float,
            alpha: float = None, use_class_re_weight: bool = False, test_begin_epoch: int = 0,
            use_recommend_re_weight: bool = True
    ):
        self.model: GeneralDebiasExplicitRecommender = model
        self.evaluator: ExplicitTestManager = evaluator
        self.envs_num: int = self.model.env_num
        self.device: torch.device = device
        self.users_tensor: torch.Tensor = training_data[:, 0]
        self.items_tensor: torch.Tensor = training_data[:, 1]
        self.scores_tensor: torch.Tensor = training_data[:, 2].float()
        self.evaluate_interval: int = evaluate_interval
        self.batch_size: int = batch_size
        self.epochs: int = epochs
        self.optimizer: torch.optim.Adam = torch.optim.Adam(model.parameters(), lr=lr)
        self.recommend_loss_type = nn.BCELoss
        # self.cluster_distance_func = nn.MSELoss(reduction='none')

        self.invariant_coe = invariant_coe
        self.L2_coe = L2_coe
        self.L1_coe = L1_coe

        self.epoch_cnt: int = 0

        self.batch_num = math.ceil(training_data.shape[0] / batch_size)

        self.each_env_count = dict()

        if alpha is None:
            self.alpha = 0.
            self.update_alpha = True
        else:
            self.alpha = alpha
            self.update_alpha = False

        self.use_class_re_weight: bool = use_class_re_weight
        self.use_recommend_re_weight: bool = use_recommend_re_weight
        self.sample_weights: torch.Tensor = torch.Tensor(np.zeros(training_data.shape[0])).to(device)

        self.test_begin_epoch: int = test_begin_epoch

        self.eps_random_tensor: torch.Tensor = self._init_eps().to(self.device)

        # self.const_env_tensor_list: list = []
        #
        # for env in range(self.envs_num):
        #     envs_tensor: torch.Tensor = torch.LongTensor(np.full(training_data.shape[0], env, dtype=int))
        #     envs_tensor = envs_tensor.to(self.device)
        #     self.const_env_tensor_list.append(envs_tensor)

    def _init_eps(self):
        base_eps = 1e-10
        eps_list: list = [base_eps * (1e-1 ** idx) for idx in range(self.envs_num)]
        temp: torch.Tensor = torch.Tensor(eps_list)
        eps_random_tensor: torch.Tensor = torch.Tensor(list(itertools.permutations(temp)))

        return eps_random_tensor

    def train_a_batch(
            self,
            batch_users_tensor: torch.Tensor,
            batch_items_tensor: torch.Tensor,
            batch_scores_tensor: torch.Tensor,
    ) -> dict:

        # print('embed_env_GMF:', self.model.embed_env_GMF.weight)
        # print('batch_envs_tensor:', batch_envs_tensor)

        # print()
        yk, yui, yu, yi = self.model(batch_users_tensor, batch_items_tensor)

        recommend_loss = self.recommend_loss_type()

        loss_o: torch.Tensor = recommend_loss(yui, batch_scores_tensor)
        loss_i: torch.Tensor = recommend_loss(yi, batch_scores_tensor)
        loss_u: torch.Tensor = recommend_loss(yu, batch_scores_tensor)

        """
        loss: torch.Tensor = invariant_loss * self.invariant_coe + env_aware_loss * self.env_aware_coe \
                             + envs_loss * self.env_coe + L2_reg * self.L2_coe + L1_reg * self.L1_coe
        """

        loss: torch.Tensor = loss_o + self.model.item_loss_weight * loss_i + self.model.user_loss_weight * loss_u

        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

        loss_dict: dict = {
            'loss_o': float(loss_o),
            'loss_i': float(loss_i),
            'loss_u': float(loss_u),
            'loss': float(loss),
        }
        return loss_dict

    def train_a_epoch(self) -> dict:
        self.model.train()
        loss_dicts_list: list = []

        for (batch_index, (
                batch_users_tensor, batch_items_tensor, batch_scores_tensor)) \
                in enumerate(mini_batch(self.batch_size, self.users_tensor, self.items_tensor, self.scores_tensor)):

            if self.update_alpha:
                p = float(batch_index + (self.epoch_cnt + 1) * self.batch_num) / float((self.epoch_cnt + 1)
                                                                                       * self.batch_num)
                self.alpha = 2. / (1. + np.exp(-10. * p)) - 1.

            loss_dict: dict = self.train_a_batch(
                batch_users_tensor=batch_users_tensor,
                batch_items_tensor=batch_items_tensor,
                batch_scores_tensor=batch_scores_tensor)
            loss_dicts_list.append(loss_dict)

        self.epoch_cnt += 1

        mean_loss_dict: dict = merge_dict(loss_dicts_list, _mean_merge_dict_func)

        return mean_loss_dict

    def train(self, silent: bool = False, auto: bool = False):
        """

        @param silent:
        @param auto:
        @return: (epoch中各损失值, 当前是第几个epoch), (当前测试指标, 当前是第几个epoch), (聚类后环境标签改变的数据有多少个, 每个环境有多少个数据, 当前是第几个epoch)
        """
        print(Fore.GREEN)
        print('=' * 30, 'train started!!!', '=' * 30)
        print(Style.RESET_ALL)

        test_result_list: list = []
        test_epoch_list: list = []

        cluster_diff_num_list: list = []
        cluster_epoch_list: list = []
        envs_cnt_list: list = []

        loss_result_list: list = []
        train_epoch_index_list: list = []

        temp_eval_result: dict = self.evaluator.evaluate()
        test_result_list.append(temp_eval_result)
        test_epoch_list.append(self.epoch_cnt)

        # self.stat_envs()

        if not silent and not auto:
            print(Fore.BLUE)
            print('test at epoch:', self.epoch_cnt)
            print(transfer_loss_dict_to_line_str(temp_eval_result))

        while self.epoch_cnt < self.epochs:
            # 训练数据, 在该方法中存在模型训练的信息
            temp_loss_dict = self.train_a_epoch()
            train_epoch_index_list.append(self.epoch_cnt)
            loss_result_list.append(temp_loss_dict)

            if not silent and not auto:
                print(Fore.GREEN)
                print('train epoch:', self.epoch_cnt)
                print(transfer_loss_dict_to_line_str(temp_loss_dict))

            if (self.epoch_cnt % self.evaluate_interval) == 0 and self.epoch_cnt >= self.test_begin_epoch:
                temp_eval_result: dict = self.evaluator.evaluate()
                test_result_list.append(temp_eval_result)
                test_epoch_list.append(self.epoch_cnt)

                if not silent and not auto:
                    print(Fore.BLUE)
                    print('test at epoch:', self.epoch_cnt)
                    print(transfer_loss_dict_to_line_str(temp_eval_result))

        print('=' * 30, 'train finished!!!', '=' * 30)
        return (loss_result_list, train_epoch_index_list), \
            (test_result_list, test_epoch_list), \
            (cluster_diff_num_list, envs_cnt_list, cluster_epoch_list)


class DICEExplicitTrainManager:
    def __init__(
            self, model: GeneralDebiasExplicitRecommender, evaluator: ExplicitTestManager,
            device: torch.device, training_data: torch.Tensor, batch_size: int,
            epochs: int, evaluate_interval: int, lr: float,
            invariant_coe: float, L2_coe: float, L1_coe: float,
            alpha: float = None, use_class_re_weight: bool = False, test_begin_epoch: int = 0,
            use_recommend_re_weight: bool = True
    ):
        self.model: GeneralDebiasExplicitRecommender = model
        self.evaluator: ExplicitTestManager = evaluator
        self.envs_num: int = self.model.env_num
        self.device: torch.device = device
        self.users_tensor: torch.Tensor = training_data[:, 0]
        self.items_tensor: torch.Tensor = training_data[:, 1]
        self.scores_tensor: torch.Tensor = training_data[:, 2].float()
        self.neg_items_tensor: torch.Tensor = training_data[:, 4]
        self.mask_tensor: torch.Tensor = training_data[:, 5].bool()
        self.evaluate_interval: int = evaluate_interval
        self.batch_size: int = batch_size
        self.epochs: int = epochs
        self.optimizer: torch.optim.Adam = torch.optim.Adam(model.parameters(), lr=lr)
        self.recommend_loss_type = nn.BCELoss
        # self.cluster_distance_func = nn.MSELoss(reduction='none')

        self.invariant_coe = invariant_coe
        self.L2_coe = L2_coe
        self.L1_coe = L1_coe

        self.epoch_cnt: int = 0

        self.batch_num = math.ceil(training_data.shape[0] / batch_size)

        self.each_env_count = dict()

        if alpha is None:
            self.alpha = 0.
            self.update_alpha = True
        else:
            self.alpha = alpha
            self.update_alpha = False

        self.use_class_re_weight: bool = use_class_re_weight
        self.use_recommend_re_weight: bool = use_recommend_re_weight
        self.sample_weights: torch.Tensor = torch.Tensor(np.zeros(training_data.shape[0])).to(device)

        self.test_begin_epoch: int = test_begin_epoch

        self.eps_random_tensor: torch.Tensor = self._init_eps().to(self.device)

        # self.const_env_tensor_list: list = []
        #
        # for env in range(self.envs_num):
        #     envs_tensor: torch.Tensor = torch.LongTensor(np.full(training_data.shape[0], env, dtype=int))
        #     envs_tensor = envs_tensor.to(self.device)
        #     self.const_env_tensor_list.append(envs_tensor)

    def _init_eps(self):
        base_eps = 1e-10
        eps_list: list = [base_eps * (1e-1 ** idx) for idx in range(self.envs_num)]
        temp: torch.Tensor = torch.Tensor(eps_list)
        eps_random_tensor: torch.Tensor = torch.Tensor(list(itertools.permutations(temp)))

        return eps_random_tensor

    def train_a_batch(
            self,
            batch_users_tensor: torch.Tensor,
            batch_items_tensor: torch.Tensor,
            batch_scores_tensor: torch.Tensor,
            batch_neg_items_tensor: torch.Tensor,
            batch_mask_tensor: torch.Tensor
    ) -> dict:

        # print('embed_env_GMF:', self.model.embed_env_GMF.weight)
        # print('batch_envs_tensor:', batch_envs_tensor)

        # print()
        loss = self.model.calculate_loss(batch_users_tensor, batch_items_tensor, batch_neg_items_tensor, batch_mask_tensor)
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

        loss_dict: dict = {
            'loss': float(loss),
        }
        return loss_dict

    def train_a_epoch(self) -> dict:
        self.model.train()
        loss_dicts_list: list = []

        for (batch_index, (batch_users_tensor, batch_items_tensor, batch_scores_tensor, batch_neg_items_tensor,
                           batch_mask_tensor)) \
                in enumerate(mini_batch(self.batch_size, self.users_tensor, self.items_tensor, self.scores_tensor,
                                        self.neg_items_tensor, self.mask_tensor)):

            if self.update_alpha:
                p = float(batch_index + (self.epoch_cnt + 1) * self.batch_num) / float((self.epoch_cnt + 1)
                                                                                       * self.batch_num)
                self.alpha = 2. / (1. + np.exp(-10. * p)) - 1.

            loss_dict: dict = self.train_a_batch(
                batch_users_tensor=batch_users_tensor,
                batch_items_tensor=batch_items_tensor,
                batch_scores_tensor=batch_scores_tensor,
                batch_neg_items_tensor=batch_neg_items_tensor,
                batch_mask_tensor=batch_mask_tensor
            )
            loss_dicts_list.append(loss_dict)

        self.epoch_cnt += 1

        mean_loss_dict: dict = merge_dict(loss_dicts_list, _mean_merge_dict_func)

        return mean_loss_dict

    def train(self, silent: bool = False, auto: bool = False):
        """

        @param silent:
        @param auto:
        @return: (epoch中各损失值, 当前是第几个epoch), (当前测试指标, 当前是第几个epoch), (聚类后环境标签改变的数据有多少个, 每个环境有多少个数据, 当前是第几个epoch)
        """
        print(Fore.GREEN)
        print('=' * 30, 'train started!!!', '=' * 30)
        print(Style.RESET_ALL)

        test_result_list: list = []
        test_epoch_list: list = []

        cluster_diff_num_list: list = []
        cluster_epoch_list: list = []
        envs_cnt_list: list = []

        loss_result_list: list = []
        train_epoch_index_list: list = []

        temp_eval_result: dict = self.evaluator.evaluate()
        test_result_list.append(temp_eval_result)
        test_epoch_list.append(self.epoch_cnt)

        # self.stat_envs()

        if not silent and not auto:
            print(Fore.BLUE)
            print('test at epoch:', self.epoch_cnt)
            print(transfer_loss_dict_to_line_str(temp_eval_result))

        while self.epoch_cnt < self.epochs:
            # 训练数据, 在该方法中存在模型训练的信息
            temp_loss_dict = self.train_a_epoch()
            train_epoch_index_list.append(self.epoch_cnt)
            loss_result_list.append(temp_loss_dict)

            if not silent and not auto:
                print(Fore.GREEN)
                print('train epoch:', self.epoch_cnt)
                print(transfer_loss_dict_to_line_str(temp_loss_dict))

            if (self.epoch_cnt % self.evaluate_interval) == 0 and self.epoch_cnt >= self.test_begin_epoch:
                temp_eval_result: dict = self.evaluator.evaluate()
                test_result_list.append(temp_eval_result)
                test_epoch_list.append(self.epoch_cnt)

                if not silent and not auto:
                    print(Fore.BLUE)
                    print('test at epoch:', self.epoch_cnt)
                    print(transfer_loss_dict_to_line_str(temp_eval_result))

        print('=' * 30, 'train finished!!!', '=' * 30)
        return (loss_result_list, train_epoch_index_list), \
            (test_result_list, test_epoch_list), \
            (cluster_diff_num_list, envs_cnt_list, cluster_epoch_list)


class BasicExplicitTrainManager:
    def __init__(
            self, model: BasicExplicitRecommender, evaluator: ExplicitTestManager,
            device: torch.device,
            training_data: torch.Tensor, batch_size: int,
            epochs: int, evaluate_interval: int, lr: float,
            L2_coe: float, L1_coe: float, test_begin_epoch: int = 0
    ):
        self.model: BasicExplicitRecommender = model
        self.evaluator: ExplicitTestManager = evaluator

        self.device: torch.device = device

        self.users_tensor: torch.Tensor = training_data[:, 0].to(device)
        self.items_tensor: torch.Tensor = training_data[:, 1].to(device)
        self.scores_tensor: torch.Tensor = training_data[:, 2].float().to(device)

        print(self.scores_tensor)

        self.evaluate_interval: int = evaluate_interval
        self.batch_size: int = batch_size
        self.epochs: int = epochs
        self.optimizer: torch.optim.Adam = torch.optim.Adam(model.parameters(), lr=lr)

        self.L2_coe = L2_coe
        self.L1_coe = L1_coe

        self.epoch_cnt: int = 0

        self.batch_num = math.ceil(training_data.shape[0] / batch_size)

        self.test_begin_epoch: int = test_begin_epoch

    def train_a_batch(
            self,
            batch_users_tensor: torch.Tensor,
            batch_items_tensor: torch.Tensor,
            batch_scores_tensor: torch.Tensor,
            *args
    ) -> dict:

        # print()

        score_loss: torch.Tensor = self.model(batch_users_tensor, batch_items_tensor, batch_scores_tensor)
        L2_reg: torch.Tensor = self.model.get_L2_reg(batch_users_tensor, batch_items_tensor)
        L1_reg: torch.Tensor = self.model.get_L1_reg(batch_users_tensor, batch_items_tensor)

        loss: torch.Tensor = score_loss + L2_reg * self.L2_coe + L1_reg * self.L1_coe

        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

        loss_dict: dict = {
            'score_loss': float(score_loss),
            'L2_reg': float(L2_reg),
            'L1_reg': float(L1_reg),
            'loss': float(loss),
        }
        return loss_dict

    def train_a_epoch(self) -> dict:
        self.model.train()
        loss_dicts_list: list = []

        for (batch_index, (batch_users_tensor, batch_items_tensor, batch_scores_tensor)) \
                in enumerate(mini_batch(self.batch_size, self.users_tensor,
                                        self.items_tensor, self.scores_tensor)):

            loss_dict: dict = self.train_a_batch(
                batch_users_tensor=batch_users_tensor,
                batch_items_tensor=batch_items_tensor,
                batch_scores_tensor=batch_scores_tensor
            )
            loss_dicts_list.append(loss_dict)

        self.epoch_cnt += 1

        mean_loss_dict: dict = merge_dict(loss_dicts_list, _mean_merge_dict_func)

        return mean_loss_dict

    def train(self, silent: bool = False, auto: bool = False):
        test_result_list: list = []
        test_epoch_list: list = []

        loss_result_list: list = []
        train_epoch_index_list: list = []

        temp_eval_result: dict = self.evaluator.evaluate()
        test_result_list.append(temp_eval_result)
        test_epoch_list.append(self.epoch_cnt)

        if not silent and not auto:
            print('test at epoch:', self.epoch_cnt)
            print(transfer_loss_dict_to_line_str(temp_eval_result))

        while self.epoch_cnt < self.epochs:
            temp_loss_dict = self.train_a_epoch()
            train_epoch_index_list.append(self.epoch_cnt)
            loss_result_list.append(temp_loss_dict)
            if not silent and not auto:
                print('train epoch:', self.epoch_cnt)
                print(transfer_loss_dict_to_line_str(temp_loss_dict))

            if (self.epoch_cnt % self.evaluate_interval) == 0 and self.epoch_cnt >= self.test_begin_epoch:
                temp_eval_result: dict = self.evaluator.evaluate()
                test_result_list.append(temp_eval_result)
                test_epoch_list.append(self.epoch_cnt)

                if not silent and not auto:
                    print('test at epoch:', self.epoch_cnt)
                    print(transfer_loss_dict_to_line_str(temp_eval_result))

        return (loss_result_list, train_epoch_index_list), \
               (test_result_list, test_epoch_list)


class BasicUniformExplicitTrainManager(BasicExplicitTrainManager):
    def __init__(
            self, model: BasicExplicitRecommender,
            evaluator: ExplicitTestManager,
            device: torch.device,
            training_data: torch.Tensor, uniform_data: torch.Tensor, batch_size: int,
            epochs: int, evaluate_interval: int, lr: float,
            L2_coe: float, L1_coe: float, test_begin_epoch: int = 0
    ):
        super(BasicUniformExplicitTrainManager, self).__init__(
            model=model, evaluator=evaluator, device=device, training_data=training_data,
            batch_size=batch_size, epochs=epochs, evaluate_interval=evaluate_interval, lr=lr,
            L2_coe=L2_coe, L1_coe=L1_coe, test_begin_epoch=test_begin_epoch
        )

        self.uniform_user: torch.Tensor = uniform_data[:, 0].to(self.device).long()
        self.uniform_item: torch.Tensor = uniform_data[:, 1].to(self.device).long()
        self.uniform_score: torch.Tensor = uniform_data[:, 2].to(self.device).float()

