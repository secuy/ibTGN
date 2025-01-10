import numpy as np

def cosine_similarity(matrix1, matrix2):
    # 计算余弦相似度
    dot_product = np.dot(matrix1, matrix2.T)
    norm_matrix1 = np.linalg.norm(matrix1.reshape(1, -1), axis=1)
    norm_matrix2 = np.linalg.norm(matrix2, axis=1)
    similarity = dot_product / (norm_matrix1[:, None] * norm_matrix2)
    similarity = similarity.flatten()
    return similarity

def euclidean_distance(matrix1, matrix2):
    # 计算欧氏距离
    distance = np.sqrt(np.sum((matrix1 - matrix2) ** 2, axis=1))
    return distance


def find_nearest_fiber(target_fiber, other_species_matrix, k, distance_pattern):
    # k是计算距离前k小的纤维
    # distance_pattern是计算距离的模式, 0是欧式距离, 1是余弦相似度
    # 计算目标纤维与其他物种所有纤维的欧氏距离
    if distance_pattern == 0:
        distances = euclidean_distance(target_fiber, other_species_matrix)
        # 找到最小距离对应的索引
        # nearest_fiber_index = np.argmin(distances)
        indices_of_smallest_20 = np.argsort(distances)
        nearest_fiber_index = indices_of_smallest_20[:k]
    elif distance_pattern == 1:
        distances = cosine_similarity(target_fiber, other_species_matrix)
        # 找到最小距离对应的索引
        # nearest_fiber_index = np.argmin(distances)
        indices_of_smallest_20 = np.argsort(distances)[::-1]
        nearest_fiber_index = indices_of_smallest_20[:k]

    # 返回最近的纤维和距离
    nearest_fiber = other_species_matrix[nearest_fiber_index]
    min_distance = distances[nearest_fiber_index]

    return nearest_fiber, min_distance, nearest_fiber_index