import math
import os
import pandas as pd
import numpy as np
from datetime import datetime
import sys
import torch
from tqdm import tqdm

from utils import analyse_interaction_from_text, analyse_user_interacted_set


class BaseImplicitDataLoader:
    def __init__(self, dataset_path: str):
        self.dataset_path = dataset_path

    def user_mask_items(self, user_id: int) -> set:
        raise NotImplementedError

    def user_highlight_items(self, user_id: int) -> set:
        raise NotImplementedError

    @property
    def all_test_users_by_sorted_tensor(self) -> torch.Tensor:
        raise NotImplementedError

    @property
    def all_test_users_by_sorted_list(self) -> list:
        raise NotImplementedError

    def get_user_ground_truth(self, user_id: int) -> set:
        raise NotImplementedError

    @property
    def get_sorted_all_test_users_ground_truth(self) -> list:
        raise NotImplementedError

    @property
    def train_data_len(self) -> int:
        raise NotImplementedError

    @property
    def test_data_len(self) -> int:
        raise NotImplementedError

    @property
    def user_num(self) -> int:
        raise NotImplementedError

    @property
    def item_num(self) -> int:
        raise NotImplementedError

    @property
    def test_data_df(self) -> pd.DataFrame:
        raise NotImplementedError

    @property
    def test_data_np(self) -> np.array:
        raise NotImplementedError


class BaseImplicitBCELossDataLoader(BaseImplicitDataLoader):
    def __init__(self, dataset_path: str):
        super(BaseImplicitBCELossDataLoader, self).__init__(dataset_path)

    def user_mask_items(self, user_id: int) -> set:
        raise NotImplementedError

    def user_highlight_items(self, user_id: int) -> set:
        raise NotImplementedError

    @property
    def all_test_users_by_sorted_tensor(self) -> torch.Tensor:
        raise NotImplementedError

    @property
    def all_test_users_by_sorted_list(self) -> list:
        raise NotImplementedError

    def get_user_ground_truth(self, user_id: int) -> set:
        raise NotImplementedError

    @property
    def get_sorted_all_test_users_ground_truth(self) -> list:
        raise NotImplementedError

    @property
    def train_data_len(self) -> int:
        raise NotImplementedError

    @property
    def test_data_len(self) -> int:
        raise NotImplementedError

    @property
    def user_num(self) -> int:
        raise NotImplementedError

    @property
    def item_num(self) -> int:
        raise NotImplementedError

    @property
    def test_data_df(self) -> pd.DataFrame:
        raise NotImplementedError

    @property
    def train_data_df(self) -> pd.DataFrame:
        raise NotImplementedError

    @property
    def test_data_np(self) -> np.array:
        raise NotImplementedError

    @property
    def train_data_np(self) -> np.array:
        raise NotImplementedError

# Yahoo, Implicit
class YahooImplicitBCELossDataLoader(BaseImplicitBCELossDataLoader):
    def __init__(self, dataset_path: str, device: torch.device, has_item_pool_file: bool = False):
        super(YahooImplicitBCELossDataLoader, self).__init__(dataset_path)
        # 训练集和测试集文件路径
        self.train_data_path: str = self.dataset_path + '/train.csv'
        self.test_data_path: str = self.dataset_path + '/test.csv'

        # DataFrame格式的训练集和测试集的数据
        self.train_df: pd.DataFrame = pd.read_csv(self.train_data_path)  # [0: 100000]
        self.test_df: pd.DataFrame = pd.read_csv(self.test_data_path)

        # numpy格式的训练集和测试集的数据
        self._train_data: np.array = self.train_df.values.astype(np.int64)
        self._test_data: np.array = self.test_df.values.astype(np.int64)

        self.user_positive_interaction = []     # 用户正样本交互
        self.user_list: list = []       # 用户ID列表
        self.item_list: list = []       # 项目ID列表

        self._user_num = 0      # 用户的数量
        self._item_num = 0      # 项目的数量

        self.test_user_list: list = []      # 测试集中用户ID列表
        self.test_item_list: list = []      # 测试集中项目ID列表
        self.ground_truth: list = []        # TODO

        self.has_item_pool: bool = has_item_pool_file   # TODO

        with open(self.train_data_path, 'r') as inp:
            inp.readline()
            lines: list = inp.readlines()

            print('Begin analyze raw train file')
            # 获取训练集中的用户打分数据
            # 将每行打分数据变成一个元组pairs，
            # 获取训练集中用户的ID列表user_list, 
            # 获取测试集中项目ID列表item_list
            pairs, self.user_list, self.item_list = analyse_interaction_from_text(lines, has_value=True)

            # 获取训练集中用户正反馈的打分数据
            positive_pairs: list = list(filter(lambda pair: pair[2] > 0, pairs))

            # 获取一个列表，这个列表中的元素是集合，集合存放的是相关用户正反馈的项目
            user_positive_interaction: list = analyse_user_interacted_set(positive_pairs)
            self.user_positive_interaction = user_positive_interaction

            self._train_pairs: list = pairs     

            inp.close()

        with open(self.test_data_path, 'r') as inp:
            inp.readline()
            lines: list = inp.readlines()
            print('Begin analyze raw test file')
            # 获取测试集中的用户打分数据
            # 将每行打分数据变成一个元组pairs，
            # 获取测试集中用户的ID列表test_user_list, 
            # 获取测试集中项目ID列表test_item_list
            pairs, self.test_user_list, self.test_item_list = analyse_interaction_from_text(lines)
            # print(self.test_user_list)
            # 获取一个列表，这个列表中的元素是集合，集合存放的是测试集中相关用户正反馈的项目
            self.ground_truth: list = analyse_user_interacted_set(pairs)
            inp.close()

        # TODO: item池？干嘛用的？
        if self.has_item_pool:
            self.item_pool_path: str = self.dataset_path + '/test_item_pool.csv'
            with open(self.item_pool_path, 'r') as inp:
                inp.readline()
                lines: list = inp.readlines()
                print('Begin analyze item pool file')
                pairs, _, _ = analyse_interaction_from_text(lines)

                self.item_pool: list = analyse_user_interacted_set(pairs)
                inp.close()

        self._user_num = max(self.user_list + self.test_user_list) + 1      # 总的用户数
        self._item_num = max(self.item_list + self.test_item_list) + 1      # 总的项目数

        self.users_tensor : torch.LongTensor = torch.LongTensor(self.user_list)
        self.users_tensor = self.users_tensor.to(device)
        self.sorted_positive_interaction = [self.user_mask_items(user_id) for user_id in self.user_list]
        self.test_users_tensor: torch.LongTensor = torch.LongTensor(self.test_user_list)
        self.test_users_tensor = self.test_users_tensor.to(device)
        self.sorted_ground_truth: list = [self.get_user_ground_truth(user_id) for user_id in self.test_user_list]

    def user_mask_items(self, user_id: int) -> set:
        """获取用户ID为user_id的用户在训练集中曾今打分过的项目

        Args:
            user_id (int): 用户ID

        Returns:
            set: 物品集合
        """
        return self.user_positive_interaction[user_id]
    
    @property
    def all_train_users_by_sorted_tensor(self) -> torch.Tensor:
        """
        注意，这个tensor没有指明device
        :return:
        """
        return self.users_tensor

    @property
    def all_train_users_by_sorted_list(self) -> list:
        return self.user_list

    def user_highlight_items(self, user_id: int) -> set:
        if not self.has_item_pool:
            raise NotImplementedError('Not has item pool!')
        return self.item_pool[user_id]

    @property
    def all_test_users_by_sorted_tensor(self) -> torch.Tensor:
        """
        注意，这个tensor没有指明device
        :return:
        """
        return self.test_users_tensor

    @property
    def all_test_users_by_sorted_list(self) -> list:
        return self.test_user_list

    def get_user_ground_truth(self, user_id: int) -> set:
        """获取用户ID为user_id的用户，在测试集中真实的物品选择列表

        Args:
            user_id (int): 用户ID

        Returns:
            set: _description_
        """
        return self.ground_truth[user_id]

    @property
    def get_sorted_all_test_users_ground_truth(self) -> list:
        """获取当前数据集所有用户在测试集中真实的物品选择列表

        Returns:
            list: _description_
        """
        return self.sorted_ground_truth

    @property
    def get_sorted_all_train_users_positive_interaction(self) -> list:
        """获取当前数据集所有用户在训练集中真实的物品选择列表

        Returns:
            list: _description_
        """
        return self.sorted_positive_interaction
    
    @property
    def train_data_len(self) -> int:
        return self.train_df.shape[0]

    @property
    def test_data_len(self) -> int:
        return self.test_df.shape[0]

    @property
    def user_num(self) -> int:
        return self._user_num

    @property
    def item_num(self) -> int:
        return self._item_num

    @property
    def test_data_df(self) -> pd.DataFrame:
        return self.test_df

    @property
    def train_data_df(self) -> pd.DataFrame:
        return self.train_df

    @property
    def test_data_np(self) -> np.array:
        return self._test_data

    @property
    def train_data_np(self) -> np.array:
        return self._train_data


class YahooUniformImplicitBCELossDataLoader(YahooImplicitBCELossDataLoader):
    def __init__(self, dataset_path: str, device: torch.device, has_item_pool_file: bool = False):
        super(YahooUniformImplicitBCELossDataLoader, self).__init__(
            dataset_path,
            device,
            has_item_pool_file
        )
        self.uniform_data_path: str = self.dataset_path + '/uniform_train.csv'
        self.uniform_df: pd.DataFrame = pd.read_csv(self.uniform_data_path)
        self._uniform_data: np.array = self.uniform_df.values.astype(np.int64)

    @property
    def uniform_data_np(self) -> np.array:
        return self._uniform_data

    @property
    def uniform_data_len(self) -> int:
        return self.uniform_df.shape[0]


class ImplicitBCELossDataLoaderStaticPopularity(YahooImplicitBCELossDataLoader):
    def __init__(self, dataset_path: str, device: torch.device, has_item_pool_file: bool = False):
        super(ImplicitBCELossDataLoaderStaticPopularity, self).__init__(
            dataset_path,
            device,
            has_item_pool_file
        )

        self.user_inter_cnt_np: np.array = np.zeros(self.user_num).astype(np.int64)
        self.item_inter_cnt_np: np.array = np.zeros(self.item_num).astype(np.int64)

        for pair in self._train_pairs:
            uid, iid = pair[0], pair[1]
            self.user_inter_cnt_np[uid] += 1
            self.item_inter_cnt_np[iid] += 1

        self.max_user_inter_cnt = self.user_inter_cnt_np.max()
        self.min_user_inter_cnt = self.user_inter_cnt_np.min()
        self.max_item_inter_cnt = self.item_inter_cnt_np.max()
        self.min_item_inter_cnt = self.item_inter_cnt_np.min()

        self.user_inter_cnt_normalize_np: np.array \
            = (self.user_inter_cnt_np - self.min_user_inter_cnt) / (self.max_user_inter_cnt - self.min_user_inter_cnt)

        self.item_inter_cnt_normalize_np: np.array \
            = (self.item_inter_cnt_np - self.min_item_inter_cnt) / (self.max_item_inter_cnt - self.min_item_inter_cnt)

    def query_users_inter_cnt(self, users_id):
        return self.user_inter_cnt_np[users_id]

    def query_items_inter_cnt(self, items_id):
        return self.item_inter_cnt_np[items_id]

    def query_users_inter_cnt_normalize(self, users_id):
        return self.user_inter_cnt_normalize_np[users_id]

    def query_items_inter_cnt_normalize(self, items_id):
        return self.item_inter_cnt_normalize_np[items_id]

    def query_pairs_cnt_add(self, users_id, items_id):
        users_cnt: np.array = self.user_inter_cnt_np[users_id]
        items_cnt: np.array = self.item_inter_cnt_np[items_id]

        return users_cnt + items_cnt

    def query_pairs_cnt_normalize_multiply(self, users_id, items_id):
        users_cnt_normalize: np.array = self.user_inter_cnt_normalize_np[users_id]
        items_cnt_normalize: np.array = self.item_inter_cnt_normalize_np[items_id]

        return users_cnt_normalize * items_cnt_normalize


class BaseExplicitDataLoader:
    def __init__(self, dataset_path: str):
        self.dataset_path = dataset_path

    @property
    def all_test_pairs_np(self) -> np.array:
        raise NotImplementedError

    @property
    def all_test_scores_np(self) -> np.array:
        raise NotImplementedError

    @property
    def test_data_np(self) -> np.array:
        raise NotImplementedError

    @property
    def all_test_pairs_tensor(self) -> torch.Tensor:
        raise NotImplementedError

    @property
    def all_test_scores_tensor(self) -> torch.Tensor:
        raise NotImplementedError

    @property
    def test_data_tensor(self) -> torch.Tensor:
        raise NotImplementedError

    @property
    def all_train_pairs_np(self) -> np.array:
        raise NotImplementedError

    @property
    def all_train_scores_np(self) -> np.array:
        raise NotImplementedError

    @property
    def train_data_np(self) -> np.array:
        raise NotImplementedError

    @property
    def all_train_pairs_tensor(self) -> torch.Tensor:
        raise NotImplementedError

    @property
    def all_train_scores_tensor(self) -> torch.Tensor:
        raise NotImplementedError

    @property
    def train_data_tensor(self) -> torch.Tensor:
        raise NotImplementedError


    @property
    def train_data_len(self) -> int:
        raise NotImplementedError

    @property
    def test_data_len(self) -> int:
        raise NotImplementedError

    @property
    def user_num(self) -> int:
        raise NotImplementedError

    @property
    def item_num(self) -> int:
        raise NotImplementedError


class ExplicitDataLoader(BaseExplicitDataLoader):
    def __init__(self, dataset_path: str, device: torch.device):
        super(ExplicitDataLoader, self).__init__(dataset_path)
        self.device: torch.device = device
        self.train_data_path: str = self.dataset_path + '/train.csv'
        self.test_data_path: str = self.dataset_path + '/test.csv'

        self.train_df: pd.DataFrame = pd.read_csv(self.train_data_path)  # [0: 100000]
        self.test_df: pd.DataFrame = pd.read_csv(self.test_data_path)

        self._train_data: np.array = self.train_df.values.astype(np.int64)
        self._test_data: np.array = self.test_df.values.astype(np.int64)

        self._train_data_tensor: torch.Tensor = torch.LongTensor(self._train_data).to(self.device)
        self._test_data_tensor: torch.Tensor = torch.LongTensor(self._test_data).to(self.device)

        self.user_positive_interaction = []

        self._user_num = int(np.max(self._train_data[:, 0].reshape(-1))) + 1
        self._item_num = int(np.max(self._train_data[:, 1].reshape(-1))) + 1

        self._train_pairs: np.array = self._train_data[:, 0:2].astype(np.int64).reshape(-1, 2)
        self._test_pairs: np.array = self._test_data[:, 0:2].astype(np.int64).reshape(-1, 2)

        self._train_pairs_tensor: torch.Tensor = torch.LongTensor(self._train_pairs).to(self.device)
        self._test_pairs_tensor: torch.Tensor = torch.LongTensor(self._test_pairs).to(self.device)

        self._train_scores: np.array = self._train_data[:, 2].astype(np.float64).reshape(-1)
        self._test_scores: np.array = self._test_data[:, 2].astype(np.float64).reshape(-1)

        self._train_scores_tensor: torch.Tensor = torch.Tensor(self._train_scores).to(self.device)
        self._test_scores_tensor: torch.Tensor = torch.Tensor(self._test_scores).to(self.device)

    @property
    def all_test_pairs_np(self) -> np.array:
        return self._test_pairs

    @property
    def all_test_scores_np(self) -> np.array:
        return self._test_scores

    @property
    def test_data_np(self) -> np.array:
        return self._test_data

    @property
    def all_train_pairs_np(self) -> np.array:
        return self._train_pairs

    @property
    def all_train_scores_np(self) -> np.array:
        return self._train_scores

    @property
    def train_data_np(self) -> np.array:
        return self._train_data

    @property
    def train_data_len(self) -> int:
        return self._train_data.shape[0]

    @property
    def test_data_len(self) -> int:
        return self._test_data.shape[0]

    @property
    def user_num(self) -> int:
        return self._user_num

    @property
    def item_num(self) -> int:
        return self._item_num

    @property
    def all_test_pairs_tensor(self) -> torch.Tensor:
        return self._test_pairs_tensor

    @property
    def all_test_scores_tensor(self) -> torch.Tensor:
        return self._test_scores_tensor

    @property
    def test_data_tensor(self) -> torch.Tensor:
        return self._test_data_tensor

    @property
    def all_train_pairs_tensor(self) -> torch.Tensor:
        return self._train_pairs_tensor

    @property
    def all_train_scores_tensor(self) -> torch.Tensor:
        return self._train_scores_tensor

    @property
    def train_data_tensor(self) -> torch.Tensor:
        return self._train_data_tensor


class ExplicitUniformDataLoader(ExplicitDataLoader):
    def __init__(self, dataset_path: str, device: torch.device):
        super(ExplicitUniformDataLoader, self).__init__(dataset_path, device)
        self.uniform_data_path: str = self.dataset_path + '/uniform_train.csv'
        self.uniform_df: pd.DataFrame = pd.read_csv(self.uniform_data_path)
        self._uniform_data: np.array = self.uniform_df.values.astype(np.int64)

    @property
    def uniform_data_np(self) -> np.array:
        return self._uniform_data

    @property
    def uniform_data_len(self) -> int:
        return self.uniform_df.shape[0]

class MLImplicitBCELossDataLoader(BaseImplicitBCELossDataLoader):
    def __init__(self, dataset_path: str, file_name: tuple, device: torch.device, has_item_pool_file: bool = False):
        super(MLImplicitBCELossDataLoader, self).__init__(dataset_path)
        # 训练集和测试集文件路径
        self.train_data_path: str = os.path.join(self.dataset_path, file_name[0])
        self.test_data_path: str = os.path.join(self.dataset_path, file_name[1])

        # DataFrame格式的训练集和测试集的数据
        self.train_df: pd.DataFrame = pd.read_csv(self.train_data_path)  # [0: 100000]
        self.test_df: pd.DataFrame = pd.read_csv(self.test_data_path)

        # numpy格式的训练集和测试集的数据
        self._train_data: np.array = self.train_df.iloc[:, 0: 3].values.astype(np.int64)
        self._test_data: np.array = self.test_df.iloc[:, 0: 3].values.astype(np.int64)

        self.user_positive_interaction = []     # 用户正样本交互
        self.user_list: list = []       # 用户ID列表
        self.item_list: list = []       # 项目ID列表

        self._user_num = 0      # 用户的数量
        self._item_num = 0      # 项目的数量

        self.test_user_list: list = []      # 测试集中用户ID列表
        self.test_item_list: list = []      # 测试集中项目ID列表
        self.ground_truth: list = []        # TODO

        self.has_item_pool: bool = has_item_pool_file   # TODO

        with open(self.train_data_path, 'r') as inp:
            inp.readline()
            lines: list = inp.readlines()

            print('Begin analyze raw train file')
            # 获取训练集中的用户打分数据
            # 将每行打分数据变成一个元组pairs，
            # 获取训练集中用户的ID列表user_list, 
            # 获取测试集中项目ID列表item_list
            pairs, self.user_list, self.item_list = analyse_interaction_from_text(lines, has_value=True)

            # 获取训练集中用户正反馈的打分数据
            positive_pairs: list = list(filter(lambda pair: pair[2] > 0, pairs))

            # 获取一个列表，这个列表中的元素是集合，集合存放的是相关用户正反馈的项目
            user_positive_interaction: list = analyse_user_interacted_set(positive_pairs)
            self.user_positive_interaction = user_positive_interaction

            self._train_pairs: list = pairs     

            inp.close()

        with open(self.test_data_path, 'r') as inp:
            inp.readline()
            lines: list = inp.readlines()
            print('Begin analyze raw test file')
            # 获取测试集中的用户打分数据
            # 将每行打分数据变成一个元组pairs，
            # 获取测试集中用户的ID列表test_user_list, 
            # 获取测试集中项目ID列表test_item_list
            pairs, self.test_user_list, self.test_item_list = analyse_interaction_from_text(lines)
            # print(self.test_user_list)
            # 获取一个列表，这个列表中的元素是集合，集合存放的是测试集中相关用户正反馈的项目
            self.ground_truth: list = analyse_user_interacted_set(pairs)
            inp.close()

        # TODO: item池？干嘛用的？
        if self.has_item_pool:
            self.item_pool_path: str = self.dataset_path + '/test_item_pool.csv'
            with open(self.item_pool_path, 'r') as inp:
                inp.readline()
                lines: list = inp.readlines()
                print('Begin analyze item pool file')
                pairs, _, _ = analyse_interaction_from_text(lines)

                self.item_pool: list = analyse_user_interacted_set(pairs)
                inp.close()

        self._user_num = max(self.user_list + self.test_user_list) + 1      # 总的用户数
        self._item_num = max(self.item_list + self.test_item_list) + 1      # 总的项目数

        self.users_tensor : torch.LongTensor = torch.LongTensor(self.user_list)
        self.users_tensor = self.users_tensor.to(device)
        self.sorted_positive_interaction = [self.user_mask_items(user_id) for user_id in self.user_list]
        self.test_users_tensor: torch.LongTensor = torch.LongTensor(self.test_user_list)
        self.test_users_tensor = self.test_users_tensor.to(device)
        self.sorted_ground_truth: list = [self.get_user_ground_truth(user_id) for user_id in self.test_user_list]

    def user_mask_items(self, user_id: int) -> set:
        """获取用户ID为user_id的用户在训练集中曾今打分过的项目

        Args:
            user_id (int): 用户ID

        Returns:
            set: 物品集合
        """
        return self.user_positive_interaction[user_id]
    
    @property
    def all_train_users_by_sorted_tensor(self) -> torch.Tensor:
        """
        注意，这个tensor没有指明device
        :return:
        """
        return self.users_tensor

    @property
    def all_train_users_by_sorted_list(self) -> list:
        return self.user_list

    def user_highlight_items(self, user_id: int) -> set:
        if not self.has_item_pool:
            raise NotImplementedError('Not has item pool!')
        return self.item_pool[user_id]

    @property
    def all_test_users_by_sorted_tensor(self) -> torch.Tensor:
        """
        注意，这个tensor没有指明device
        :return:
        """
        return self.test_users_tensor

    @property
    def all_test_users_by_sorted_list(self) -> list:
        return self.test_user_list

    def get_user_ground_truth(self, user_id: int) -> set:
        """获取用户ID为user_id的用户，在测试集中真实的物品选择列表

        Args:
            user_id (int): 用户ID

        Returns:
            set: _description_
        """
        return self.ground_truth[user_id]

    @property
    def get_sorted_all_test_users_ground_truth(self) -> list:
        """获取当前数据集所有用户在测试集中真实的物品选择列表

        Returns:
            list: _description_
        """
        return self.sorted_ground_truth

    @property
    def get_sorted_all_train_users_positive_interaction(self) -> list:
        """获取当前数据集所有用户在训练集中真实的物品选择列表

        Returns:
            list: _description_
        """
        return self.sorted_positive_interaction
    
    @property
    def train_data_len(self) -> int:
        return self.train_df.shape[0]

    @property
    def test_data_len(self) -> int:
        return self.test_df.shape[0]

    @property
    def user_num(self) -> int:
        return self._user_num

    @property
    def item_num(self) -> int:
        return self._item_num

    @property
    def test_data_df(self) -> pd.DataFrame:
        return self.test_df

    @property
    def train_data_df(self) -> pd.DataFrame:
        return self.train_df

    @property
    def test_data_np(self) -> np.array:
        return self._test_data

    @property
    def train_data_np(self) -> np.array:
        return self._train_data


class MLImplicitBCELossDataLoaderThreeBiases(BaseImplicitBCELossDataLoader):
    def __init__(self, dataset_path: str, file_name: tuple, device: torch.device, has_item_pool_file: bool = False):
        super(MLImplicitBCELossDataLoaderThreeBiases, self).__init__(dataset_path)
        # 训练集和测试集文件路径
        self.train_data_path: str = os.path.join(self.dataset_path, file_name[0])
        self.test_data_path: str = os.path.join(self.dataset_path, file_name[1])

        # DataFrame格式的训练集和测试集的数据
        self.train_df: pd.DataFrame = pd.read_csv(self.train_data_path)  # [0: 100000]
        self.test_df: pd.DataFrame = pd.read_csv(self.test_data_path)

        # numpy格式的训练集和测试集的数据
        self._train_data: np.array = self.train_df.iloc[:, 0: 3].values.astype(np.int64)
        self._test_data: np.array = self.test_df.iloc[:, 0: 3].values.astype(np.int64)

        self.user_positive_interaction = []  # 用户正样本交互
        self.user_list: list = []  # 用户ID列表
        self.item_list: list = []  # 项目ID列表

        self._user_num = 0  # 用户的数量
        self._item_num = 0  # 项目的数量

        self.test_user_list: list = []  # 测试集中用户ID列表
        self.test_item_list: list = []  # 测试集中项目ID列表
        self.ground_truth: list = []  # TODO

        self.has_item_pool: bool = has_item_pool_file  # TODO

        with open(self.train_data_path, 'r') as inp:
            inp.readline()
            lines: list = inp.readlines()

            print('Begin analyze raw train file')
            # 获取训练集中的用户打分数据
            # 将每行打分数据变成一个元组pairs，
            # 获取训练集中用户的ID列表user_list,
            # 获取测试集中项目ID列表item_list
            pairs, self.user_list, self.item_list = analyse_interaction_from_text(lines, has_value=True)

            # 获取训练集中用户正反馈的打分数据
            positive_pairs: list = list(filter(lambda pair: pair[2] >= 0, pairs))

            # 获取一个列表，这个列表中的元素是集合，集合存放的是相关用户正反馈的项目
            user_positive_interaction: list = analyse_user_interacted_set(positive_pairs)
            self.user_positive_interaction = user_positive_interaction

            self._train_pairs: list = pairs

            inp.close()

        with open(self.test_data_path, 'r') as inp:
            inp.readline()
            lines: list = inp.readlines()
            print('Begin analyze raw test file')
            # 获取测试集中的用户打分数据
            # 将每行打分数据变成一个元组pairs，
            # 获取测试集中用户的ID列表test_user_list,
            # 获取测试集中项目ID列表test_item_list
            pairs, self.test_user_list, self.test_item_list = analyse_interaction_from_text(lines)
            # print(self.test_user_list)
            # 获取一个列表，这个列表中的元素是集合，集合存放的是测试集中相关用户正反馈的项目
            self.ground_truth: list = analyse_user_interacted_set(pairs)
            inp.close()

        # TODO: item池？干嘛用的？
        if self.has_item_pool:
            self.item_pool_path: str = self.dataset_path + '/test_item_pool.csv'
            with open(self.item_pool_path, 'r') as inp:
                inp.readline()
                lines: list = inp.readlines()
                print('Begin analyze item pool file')
                pairs, _, _ = analyse_interaction_from_text(lines)

                self.item_pool: list = analyse_user_interacted_set(pairs)
                inp.close()

        self._user_num = max(self.user_list + self.test_user_list) + 1  # 总的用户数
        self._item_num = max(self.item_list + self.test_item_list) + 1  # 总的项目数

        self.users_tensor: torch.LongTensor = torch.LongTensor(self.user_list)
        self.users_tensor = self.users_tensor.to(device)
        self.sorted_positive_interaction = [self.user_mask_items(user_id) for user_id in self.user_list]
        self.test_users_tensor: torch.LongTensor = torch.LongTensor(self.test_user_list)
        self.test_users_tensor = self.test_users_tensor.to(device)
        self.sorted_ground_truth: list = [self.get_user_ground_truth(user_id) for user_id in self.test_user_list]

    def user_mask_items(self, user_id: int) -> set:
        """获取用户ID为user_id的用户在训练集中曾今打分过的项目

        Args:
            user_id (int): 用户ID

        Returns:
            set: 物品集合
        """
        return self.user_positive_interaction[user_id]

    @property
    def all_train_users_by_sorted_tensor(self) -> torch.Tensor:
        """
        注意，这个tensor没有指明device
        :return:
        """
        return self.users_tensor

    @property
    def all_train_users_by_sorted_list(self) -> list:
        return self.user_list

    def user_highlight_items(self, user_id: int) -> set:
        if not self.has_item_pool:
            raise NotImplementedError('Not has item pool!')
        return self.item_pool[user_id]

    @property
    def all_test_users_by_sorted_tensor(self) -> torch.Tensor:
        """
        注意，这个tensor没有指明device
        :return:
        """
        return self.test_users_tensor

    @property
    def all_test_users_by_sorted_list(self) -> list:
        return self.test_user_list

    def get_user_ground_truth(self, user_id: int) -> set:
        """获取用户ID为user_id的用户，在测试集中真实的物品选择列表

        Args:
            user_id (int): 用户ID

        Returns:
            set: _description_
        """
        return self.ground_truth[user_id]

    @property
    def get_sorted_all_test_users_ground_truth(self) -> list:
        """获取当前数据集所有用户在测试集中真实的物品选择列表

        Returns:
            list: _description_
        """
        return self.sorted_ground_truth

    @property
    def get_sorted_all_train_users_positive_interaction(self) -> list:
        """获取当前数据集所有用户在训练集中真实的物品选择列表

        Returns:
            list: _description_
        """
        return self.sorted_positive_interaction

    @property
    def train_data_len(self) -> int:
        return self.train_df.shape[0]

    @property
    def test_data_len(self) -> int:
        return self.test_df.shape[0]

    @property
    def user_num(self) -> int:
        return self._user_num

    @property
    def item_num(self) -> int:
        return self._item_num

    @property
    def test_data_df(self) -> pd.DataFrame:
        return self.test_df

    @property
    def train_data_df(self) -> pd.DataFrame:
        return self.train_df

    @property
    def test_data_np(self) -> np.array:
        return self._test_data

    @property
    def train_data_np(self) -> np.array:
        return self._train_data


class MLExplicitDataLoader(BaseExplicitDataLoader):
    def __init__(self, dataset_path: str, file_name: tuple, device: torch.device, has_item_pool_file: bool = False):
        super(MLExplicitDataLoader, self).__init__(dataset_path)
        # 训练集和测试集文件路径
        self.train_data_path: str = os.path.join(self.dataset_path, file_name[0])
        self.test_data_path: str = os.path.join(self.dataset_path, file_name[1])

        # DataFrame格式的训练集和测试集的数据
        self.train_df: pd.DataFrame = pd.read_csv(self.train_data_path)  # [0: 100000]
        self.test_df: pd.DataFrame = pd.read_csv(self.test_data_path)

        # numpy格式的训练集和测试集的数据
        self._train_data: np.array = self.train_df.iloc[:, 0: 3].values.astype(np.int64)
        self._test_data: np.array = self.test_df.iloc[:, 0: 3].values.astype(np.int64)

        self.user_positive_interaction = []  # 用户正样本交互
        self.user_list: list = []  # 用户ID列表
        self.item_list: list = []  # 项目ID列表

        self._user_num = 0  # 用户的数量
        self._item_num = 0  # 项目的数量

        self.test_user_list: list = []  # 测试集中用户ID列表
        self.test_item_list: list = []  # 测试集中项目ID列表
        self.ground_truth: list = []  # TODO

        self.has_item_pool: bool = has_item_pool_file  # TODO

        with open(self.train_data_path, 'r') as inp:
            inp.readline()
            lines: list = inp.readlines()

            print('Begin analyze raw train file')
            # 获取训练集中的用户打分数据
            # 将每行打分数据变成一个元组pairs，
            # 获取训练集中用户的ID列表user_list,
            # 获取测试集中项目ID列表item_list
            pairs, self.user_list, self.item_list = analyse_interaction_from_text(lines, has_value=True)

            # 获取训练集中用户正反馈的打分数据
            positive_pairs: list = list(filter(lambda pair: pair[2] >= 0, pairs))

            # 获取一个列表，这个列表中的元素是集合，集合存放的是相关用户正反馈的项目
            user_positive_interaction: list = analyse_user_interacted_set(positive_pairs)
            self.user_positive_interaction = user_positive_interaction

            self._train_pairs: list = pairs

            inp.close()

        with open(self.test_data_path, 'r') as inp:
            inp.readline()
            lines: list = inp.readlines()
            print('Begin analyze raw test file')
            # 获取测试集中的用户打分数据
            # 将每行打分数据变成一个元组pairs，
            # 获取测试集中用户的ID列表test_user_list,
            # 获取测试集中项目ID列表test_item_list
            pairs, self.test_user_list, self.test_item_list = analyse_interaction_from_text(lines)
            # print(self.test_user_list)
            # 获取一个列表，这个列表中的元素是集合，集合存放的是测试集中相关用户正反馈的项目
            self.ground_truth: list = analyse_user_interacted_set(pairs)
            inp.close()

        # TODO: item池？干嘛用的？
        if self.has_item_pool:
            self.item_pool_path: str = self.dataset_path + '/test_item_pool.csv'
            with open(self.item_pool_path, 'r') as inp:
                inp.readline()
                lines: list = inp.readlines()
                print('Begin analyze item pool file')
                pairs, _, _ = analyse_interaction_from_text(lines)

                self.item_pool: list = analyse_user_interacted_set(pairs)
                inp.close()

        self._user_num = max(self.user_list + self.test_user_list) + 1  # 总的用户数
        self._item_num = max(self.item_list + self.test_item_list) + 1  # 总的项目数

        self.users_tensor: torch.LongTensor = torch.LongTensor(self.user_list)
        self.users_tensor = self.users_tensor.to(device)
        self.sorted_positive_interaction = [self.user_mask_items(user_id) for user_id in self.user_list]
        self.test_users_tensor: torch.LongTensor = torch.LongTensor(self.test_user_list)
        self.test_users_tensor = self.test_users_tensor.to(device)
        self.sorted_ground_truth: list = [self.get_user_ground_truth(user_id) for user_id in self.test_user_list]

    def user_mask_items(self, user_id: int) -> set:
        """获取用户ID为user_id的用户在训练集中曾今打分过的项目

        Args:
            user_id (int): 用户ID

        Returns:
            set: 物品集合
        """
        return self.user_positive_interaction[user_id]

    @property
    def all_train_users_by_sorted_tensor(self) -> torch.Tensor:
        """
        注意，这个tensor没有指明device
        :return:
        """
        return self.users_tensor

    @property
    def all_train_users_by_sorted_list(self) -> list:
        return self.user_list

    def user_highlight_items(self, user_id: int) -> set:
        if not self.has_item_pool:
            raise NotImplementedError('Not has item pool!')
        return self.item_pool[user_id]

    @property
    def all_test_users_by_sorted_tensor(self) -> torch.Tensor:
        """
        注意，这个tensor没有指明device
        :return:
        """
        return self.test_users_tensor

    @property
    def all_test_users_by_sorted_list(self) -> list:
        return self.test_user_list

    def get_user_ground_truth(self, user_id: int) -> set:
        """获取用户ID为user_id的用户，在测试集中真实的物品选择列表

        Args:
            user_id (int): 用户ID

        Returns:
            set: _description_
        """
        return self.ground_truth[user_id]

    @property
    def get_sorted_all_test_users_ground_truth(self) -> list:
        """获取当前数据集所有用户在测试集中真实的物品选择列表

        Returns:
            list: _description_
        """
        return self.sorted_ground_truth

    @property
    def get_sorted_all_train_users_positive_interaction(self) -> list:
        """获取当前数据集所有用户在训练集中真实的物品选择列表

        Returns:
            list: _description_
        """
        return self.sorted_positive_interaction

    @property
    def train_data_len(self) -> int:
        return self.train_df.shape[0]

    @property
    def test_data_len(self) -> int:
        return self.test_df.shape[0]

    @property
    def user_num(self) -> int:
        return self._user_num

    @property
    def item_num(self) -> int:
        return self._item_num

    @property
    def test_data_df(self) -> pd.DataFrame:
        return self.test_df

    @property
    def train_data_df(self) -> pd.DataFrame:
        return self.train_df

    @property
    def test_data_np(self) -> np.array:
        return self._test_data

    @property
    def train_data_np(self) -> np.array:
        return self._train_data


class DICEExplicitDataLoader(BaseExplicitDataLoader):
    def __init__(self, dataset_path: str, file_name: tuple, batch_size: int,
                 device: torch.device, has_item_pool_file: bool = False):
        super(DICEExplicitDataLoader, self).__init__(dataset_path)
        # 训练集和测试集文件路径
        self.train_data_path: str = os.path.join(self.dataset_path, file_name[0])
        self.test_data_path: str = os.path.join(self.dataset_path, file_name[1])

        # DataFrame格式的训练集和测试集的数据
        self.train_df: pd.DataFrame = pd.read_csv(self.train_data_path)  # [0: 100000]
        self.test_df: pd.DataFrame = pd.read_csv(self.test_data_path)

        # numpy格式的训练集和测试集的数据
        self._train_data: np.array = self.train_df.iloc[:, 0: 6].values.astype(np.int64)
        self._test_data: np.array = self.test_df.iloc[:, 0: 3].values.astype(np.int64)

        self.user_positive_interaction = []  # 用户正样本交互
        self.user_list: list = []  # 用户ID列表
        self.item_list: list = []  # 项目ID列表

        self._user_num = 0  # 用户的数量
        self._item_num = 0  # 项目的数量

        self.test_user_list: list = []  # 测试集中用户ID列表
        self.test_item_list: list = []  # 测试集中项目ID列表
        self.ground_truth: list = []  # TODO

        self.has_item_pool: bool = has_item_pool_file  # TODO

        with open(self.train_data_path, 'r') as inp:
            inp.readline()
            lines: list = inp.readlines()

            print('Begin analyze raw train file')
            # 获取训练集中的用户打分数据
            # 将每行打分数据变成一个元组pairs，
            # 获取训练集中用户的ID列表user_list,
            # 获取测试集中项目ID列表item_list
            pairs, self.user_list, self.item_list = analyse_interaction_from_text(lines, has_value=True)

            # 获取训练集中用户正反馈的打分数据
            positive_pairs: list = list(filter(lambda pair: pair[2] > 0, pairs))

            # 获取一个列表，这个列表中的元素是集合，集合存放的是相关用户正反馈的项目
            user_positive_interaction: list = analyse_user_interacted_set(positive_pairs)
            self.user_positive_interaction = user_positive_interaction

            self._train_pairs: list = pairs

            inp.close()

        with open(self.test_data_path, 'r') as inp:
            inp.readline()
            lines: list = inp.readlines()
            print('Begin analyze raw test file')
            # 获取测试集中的用户打分数据
            # 将每行打分数据变成一个元组pairs，
            # 获取测试集中用户的ID列表test_user_list,
            # 获取测试集中项目ID列表test_item_list
            pairs, self.test_user_list, self.test_item_list = analyse_interaction_from_text(lines)
            # print(self.test_user_list)
            # 获取一个列表，这个列表中的元素是集合，集合存放的是测试集中相关用户正反馈的项目
            self.ground_truth: list = analyse_user_interacted_set(pairs)
            inp.close()

        # TODO: item池？干嘛用的？
        if self.has_item_pool:
            self.item_pool_path: str = self.dataset_path + '/test_item_pool.csv'
            with open(self.item_pool_path, 'r') as inp:
                inp.readline()
                lines: list = inp.readlines()
                print('Begin analyze item pool file')
                pairs, _, _ = analyse_interaction_from_text(lines)

                self.item_pool: list = analyse_user_interacted_set(pairs)
                inp.close()

        self._user_num = max(self.user_list + self.test_user_list) + 1  # 总的用户数
        self._item_num = max(self.item_list + self.test_item_list) + 1  # 总的项目数

        self.users_tensor: torch.LongTensor = torch.LongTensor(self.user_list)
        self.users_tensor = self.users_tensor.to(device)
        self.sorted_positive_interaction = [self.user_mask_items(user_id) for user_id in self.user_list]
        self.test_users_tensor: torch.LongTensor = torch.LongTensor(self.test_user_list)
        self.test_users_tensor = self.test_users_tensor.to(device)
        self.sorted_ground_truth: list = [self.get_user_ground_truth(user_id) for user_id in self.test_user_list]

    def sample_neg_items_and_mask(self, batch_size):
        batch_num = math.ceil(self.train_df.shape[0] / batch_size)
        iid_list = np.arange(np.max(self.train_df['item_id']))
        self.train_df['neg_item_id'], self.train_df['mask'] = 0, True
        new_train_data = []

        # 计算流行度
        iid2pop = self.get_item_popularity()

        for i in range(0, batch_num, batch_size):
            batch_data = self.train_df[i*batch_size: (i+1)*batch_size]
            batch_data = pd.concat([batch_data, batch_data])

            # 负采样物品ID
            batch_item_list = np.sort(np.unique(batch_data['item_id']))
            mask = ~np.zeros(len(iid_list), dtype=np.bool)
            mask[(batch_item_list - 1)] = False
            neg_index_list = iid_list[mask]
            random_index_list = np.random.randint(0, len(neg_index_list), size=batch_data.shape[0])
            neg_item_list = neg_index_list[random_index_list] + 1
            batch_data['neg_item_id'] = neg_item_list

            # 对比正样本和负样本之间的流行度差异，如果正样本比负样本流行度高则为True，否则为False
            positive_item_list = np.array(batch_data['item_id'].values)
            print(batch_data)
            print(positive_item_list)
            pop_mask_list = np.array([iid2pop[i] for i in positive_item_list]) >= np.array([iid2pop[i] for i in neg_item_list])
            batch_data['mask'] = pop_mask_list

            new_train_data.extend(batch_data.values.tolist())

        return np.array(new_train_data, dtype=np.int64)


    def get_item_popularity(self):
        iid_count_matrix = np.array(list(self.train_df['item_id'].value_counts().items()))
        total_item_count = self.train_df.shape[0]
        iid_pop = (1.0 * iid_count_matrix[:, 1]) / total_item_count

        # 归一化
        iid_pop = (iid_pop - np.min(iid_pop)) / (np.max(iid_pop) - np.min(iid_pop))

        iid2pop = dict()

        # 将不存在的物品ID的流行度设置为0(初始化所有的物品ID)
        for iid in range(1, np.max(iid_count_matrix[:, 0])):
            iid2pop[int(iid)] = 0.0

        for i, iid in enumerate(iid_count_matrix[:, 0]):
            iid2pop[int(iid)] = float(iid_pop[i])

        return iid2pop

    def user_mask_items(self, user_id: int) -> set:
        """获取用户ID为user_id的用户在训练集中曾今打分过的项目

        Args:
            user_id (int): 用户ID

        Returns:
            set: 物品集合
        """
        return self.user_positive_interaction[user_id]

    @property
    def all_train_users_by_sorted_tensor(self) -> torch.Tensor:
        """
        注意，这个tensor没有指明device
        :return:
        """
        return self.users_tensor

    @property
    def all_train_users_by_sorted_list(self) -> list:
        return self.user_list

    def user_highlight_items(self, user_id: int) -> set:
        if not self.has_item_pool:
            raise NotImplementedError('Not has item pool!')
        return self.item_pool[user_id]

    @property
    def all_test_users_by_sorted_tensor(self) -> torch.Tensor:
        """
        注意，这个tensor没有指明device
        :return:
        """
        return self.test_users_tensor

    @property
    def all_test_users_by_sorted_list(self) -> list:
        return self.test_user_list

    def get_user_ground_truth(self, user_id: int) -> set:
        """获取用户ID为user_id的用户，在测试集中真实的物品选择列表

        Args:
            user_id (int): 用户ID

        Returns:
            set: _description_
        """
        return self.ground_truth[user_id]

    @property
    def get_sorted_all_test_users_ground_truth(self) -> list:
        """获取当前数据集所有用户在测试集中真实的物品选择列表

        Returns:
            list: _description_
        """
        return self.sorted_ground_truth

    @property
    def get_sorted_all_train_users_positive_interaction(self) -> list:
        """获取当前数据集所有用户在训练集中真实的物品选择列表

        Returns:
            list: _description_
        """
        return self.sorted_positive_interaction

    @property
    def train_data_len(self) -> int:
        return self.train_df.shape[0]

    @property
    def test_data_len(self) -> int:
        return self.test_df.shape[0]

    @property
    def user_num(self) -> int:
        return self._user_num

    @property
    def item_num(self) -> int:
        return self._item_num

    @property
    def test_data_df(self) -> pd.DataFrame:
        return self.test_df

    @property
    def train_data_df(self) -> pd.DataFrame:
        return self.train_df

    @property
    def test_data_np(self) -> np.array:
        return self._test_data

    @property
    def train_data_np(self) -> np.array:
        return self._train_data