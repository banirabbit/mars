import torch
import numpy as np
class NormalizedDCG:
    def __init__(self, top_k):
        self.top_k = top_k

    def calculate_dcg(self, relevant_scores):
        """
        计算DCG (Discounted Cumulative Gain)
        :param relevant_scores: 相关性得分列表 (长短相关性排序的向量)
        :return: DCG值
        """
        dcg = 0.0
        for i, score in enumerate(relevant_scores):
            # 注意log2的基准：i+2，i从0开始
            dcg += score / np.log2(i + 2)
        return dcg

    def calculate_ndcg(self, predicted_apis, truth_apis):
        """
        计算NDCG
        :param predicted_apis: 推荐的API列表（根据模型返回的排序）
        :param truth_apis: 真实的相关API列表
        :return: NDCG值
        """
        # 步骤1：根据top_k截取预测列表
        predicted_apis_top_k = predicted_apis[:self.top_k]
        
        # 确保真实的API列表不超过top_k的长度
        truth_apis_top_k = truth_apis[:self.top_k]

        # 步骤2：计算DCG（折损累积增益）
        # 将预测的API与真实API进行比对，构造相关性得分（1表示相关，0表示不相关）
        relevance_scores = [1 if api in truth_apis_top_k else 0 for api in predicted_apis_top_k]
        dcg = self.calculate_dcg(relevance_scores)

        # 步骤3：计算IDCG（理想DCG）
        # 对于理想的情况，正确的API排名靠前，构建理想的相关性得分
        ideal_relevance_scores = [1] * min(len(truth_apis_top_k), self.top_k)
        idcg = self.calculate_dcg(ideal_relevance_scores)
    
        # 步骤4：计算NDCG = DCG / IDCG
        ndcg = dcg / idcg if idcg > 0 else 0.0
        return ndcg