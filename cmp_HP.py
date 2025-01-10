import os
import networkx as nx
import numpy as np
import scipy.sparse as sp
import torch
from dipy.io.image import load_nifti, load_nifti_data
from dipy.viz import actor, window
from sklearn.manifold import TSNE
import matplotlib.pyplot as plt
from torch.utils.data import DataLoader
from collections import defaultdict
from dipy.tracking.fbcmeasures import FBCMeasures
from dipy.denoise.enhancement_kernel import EnhancementKernel
import matplotlib.pyplot as plt

import args
from dataset import GraphDataset
import model
from utils.calc_code_euc import find_nearest_fiber
from utils.preprocessing import preprocess_graph, sparse_to_tuple, mask_test_edges
import test_args


def fiber_distance(fiber1, fiber2):
    dist = np.linalg.norm(fiber1.flatten() - fiber2.flatten())
    reverse_fiber2 = np.flipud(fiber2)
    dist_flip = np.linalg.norm(fiber1.flatten() - reverse_fiber2.flatten())
    return min(dist, dist_flip)

device = torch.device("cuda:1" if torch.cuda.is_available() else "cpu")
# 创建 Dataset
graph_dataset = GraphDataset(test_args)
# 使用 DataLoader 加载数据
dataloader = DataLoader(graph_dataset, batch_size=1, shuffle=False)

model = getattr(model, test_args.model)()
model.load_state_dict(torch.load('./trained_model/{}_model.pth'.format(test_args.model), map_location='cpu'))

model.eval()

feat_code = []
for batch_idx, (feature, adjacency_matrix) in enumerate(dataloader):
    feature = feature.numpy().reshape(-1, test_args.input_dim)
    adjacency_matrix = adjacency_matrix.numpy().reshape(adjacency_matrix.shape[1], -1)
    graph = defaultdict(dict)
    # 遍历邻接矩阵的每个元素，将非零元素加入 defaultdict
    for i in range(adjacency_matrix.shape[0]):
        for j in range(adjacency_matrix.shape[1]):
            weight = adjacency_matrix[i, j]
            if weight != 0:
                graph[i][j] = {'weight': weight}
    # Output the adjacency matrix size after the loop
    adj = nx.adjacency_matrix(nx.from_dict_of_dicts(graph), weight='weight')
    feature = sp.csr_matrix(feature).tolil()

    adj_norm = preprocess_graph(adj)

    feature = sparse_to_tuple(feature.tocoo())
    feature = torch.sparse.FloatTensor(torch.LongTensor(feature[0].T),
                                        torch.FloatTensor(feature[1]),
                                        torch.Size(feature[2]))

    adj_norm = torch.sparse.FloatTensor(torch.LongTensor(adj_norm[0].T),
                                        torch.FloatTensor(adj_norm[1]),
                                        torch.Size(adj_norm[2]))
    mean_times = 500
    if test_args.model == 'GAE':
        with torch.no_grad():
            A_pred, Z = model(feature, adj_norm)
        A_pred = A_pred.numpy()
        Z = Z.numpy()
    else:
        Zss = []
        with torch.no_grad():
            for times in range(mean_times):
                A_pred, Z = model(feature, adj_norm)
                Zss.append(Z.numpy())
        Zss = np.array(Zss)
        Z = np.mean(Zss, axis=0)

    feat_code.append(Z)

# plt.scatter(feat_code[0][:,13], feat_code[0][:,15],s=5,c='red')
# plt.scatter(feat_code[1][:,13], feat_code[1][:,15],s=5,c='blue')
# # 添加标题和标签
# plt.title('Simple Plot')
# plt.xlabel('X-axis')
# plt.ylabel('Y-axis')
#
# # 显示图形
# plt.show()


# 0 是 human转monkey, 1 是 monkey转human
changeModel = 0
human2monkey_idx = 21
monkey2human_idx = 262

if changeModel == 0:
    cmp_root_sub = "F:/njust/wuye/week08/vgae_pytorch-master/cmp_data_v2/sub-{}/".format(str(test_args.primate_start_num +
                                                                                             test_args.primate_sub_num - 1).zfill(3))
    for human_choose_fiber in range(human2monkey_idx, 1000):
        # 选择第1个物种的一条纤维（对比时这里修改）
        target_fiber_species1 = feat_code[0]

        # 距离前k近的纤维
        k = 10
        # 计算距离的模式0是欧氏距离，1是余弦相似度
        distance_pattern = 0
        # 找到最近的纤维和距离（对比时这里修改）
        nearest_fiber, min_distance, nearest_fiber_index = find_nearest_fiber(target_fiber_species1[human_choose_fiber],
                                                                              feat_code[1], k, distance_pattern)
        print(f"最近的纤维索引: {nearest_fiber_index}")
        print(f"最小欧氏距离: {min_distance}")

        scene = window.Scene()
        scene.SetBackground(1, 1, 1)

        tot_h = 0
        tot_p = 0
        primate_fiber = []
        for i in range(1000):
            h_fiber = np.load('F:/njust/wuye/week08/vgae_pytorch-master/cmp_data_v2/sub-021/cluser_{}.npy'.format(i))
            tot_h += h_fiber.shape[0]
            if i < 500:
                p_fiber = np.load(cmp_root_sub + 'cluser_{}.npy'.format(i))
                tot_p += p_fiber.shape[0]
                if i in nearest_fiber_index:
                    scene.add(actor.line(p_fiber, colors=(0, 1, 0)))
                    primate_fiber.append(p_fiber)
                else:
                    scene.add(actor.line(p_fiber, colors=(0, 0, 1)))
            if i == human_choose_fiber:
                scene.add(actor.line(h_fiber, colors=(0, 1, 0)))
                human_fiber = h_fiber
                print("人类的纤维尺寸:" + str(human_fiber.shape))
                print("人类的纤维索引:" + str(human_choose_fiber))
            else:
                scene.add(actor.line(h_fiber, colors=(1, 0, 0)))

        print(tot_h, tot_p)

        # hardi_fname = "F:/njust/wuye/week09/vgae_pytorch-master6/test02_data/sub-021/proc/dwi.nii.gz"
        # t1_fname = "F:/njust/wuye/week09/vgae_pytorch-master6/test02_data/sub-021/proc/t1w.nii.gz"
        # data, affine = load_nifti(hardi_fname)
        # t1_data = load_nifti_data(t1_fname)
        # # Horizontal (axial) slice of T1 data
        # vol_actor1 = actor.slicer(t1_data, affine=affine)
        # vol_actor1.display(z=100)
        # scene.add(vol_actor1)
        #
        # # Vertical (sagittal) slice of T1 data
        # vol_actor2 = actor.slicer(t1_data, affine=affine)
        # vol_actor2.display(x=80)
        # scene.add(vol_actor2)

        window.show(scene)

        D33 = 1.0
        D44 = 0.02
        t = 1
        z = EnhancementKernel(D33, D44, t)


        for idx in range(k):
            scene = window.Scene()
            scene.SetBackground(1, 1, 1)
            primate_near = np.load(cmp_root_sub + 'cluser_{}.npy'.format(nearest_fiber_index[idx]))
            # min_ab_lines = []
            # min_ba_lines = []
            # min_ab = 0
            # min_ba = 0
            # for line in h_fiber:
            #     minv = 9999
            #     min_idx = -1
            #     for idx, i in enumerate(primate_near):
            #         dist = fiber_distance(line, i)
            #         if dist < minv:
            #             min_idx = idx
            #             minv = dist
            #     min_ab += minv
            #     min_ab_lines.append(primate_near[min_idx])
            # min_ab_lines = np.array(min_ab_lines)
            #
            # for line in primate_near:
            #     minv = 9999
            #     min_idx = -1
            #     for idx, i in enumerate(h_fiber):
            #         dist = fiber_distance(line, i)
            #         if dist < minv:
            #             min_idx = idx
            #             minv = dist
            #     min_ba += minv
            #     # print(minv)
            #     min_ba_lines.append(h_fiber[min_idx])
            # min_ba_lines = np.array(min_ba_lines)
            # print(min_ab, min_ba, min_ab - min_ba)
            # print(min_ab / h_fiber.shape[0], min_ba / len(primate_near),
            #       (min_ab / h_fiber.shape[0]) - (min_ba / len(primate_near)))

            print("第{}近的纤维束索引：".format(idx + 1) + str(nearest_fiber_index[idx]))
            print("第{}近的纤维束尺寸：".format(idx + 1) + str(primate_near.shape))
            print("第{}近的纤维束距离：".format(idx + 1) + str(min_distance[idx]))
            fbc = FBCMeasures(human_fiber, z)
            fbc_sl_thres, clrs_thres, rfbc_thres = fbc.get_points_rfbc_thresholded(0.5, emphasis=0.01)
            scene.add(actor.line(fbc_sl_thres))  # np.vstack(clrs_thres),
            # scene.add(actor.line(human_fiber)) # , colors=(1, 0, 0)

            fbc2 = FBCMeasures(primate_near, z)
            fbc_sl_thres2, clrs_thres2, rfbc_thres2 = fbc2.get_points_rfbc_thresholded(0.5, emphasis=0.01)
            scene.add(actor.line(fbc_sl_thres2))  # np.vstack(clrs_thres),


            # scene.add(actor.line(primate_near)) # , colors=(0, 0, 1)

            # # Horizontal (axial) slice of T1 data
            # vol_actor1 = actor.slicer(t1_data, affine=affine)
            # vol_actor1.display(z=100)
            # scene.add(vol_actor1)
            #
            # # Vertical (sagittal) slice of T1 data
            # vol_actor2 = actor.slicer(t1_data, affine=affine)
            # vol_actor2.display(x=80)
            # scene.add(vol_actor2)

            window.show(scene)
else:
    cmp_root_sub = "F:/njust/wuye/week08/vgae_pytorch-master/cmp_data_v2/sub-{}/".format(str(test_args.human_start_num +
                                                                                             test_args.human_sub_num - 1).zfill(3))
    for primate_choose_fiber in range(monkey2human_idx, 500):
        # 选择第2个物种的一条纤维（对比时这里修改）
        target_fiber_species1 = feat_code[1]

        # 距离前k近的纤维
        k = 10
        # 计算距离的模式0是欧氏距离，1是余弦相似度
        distance_pattern = 0
        # 找到最近的纤维和距离（对比时这里修改）
        nearest_fiber, min_distance, nearest_fiber_index = find_nearest_fiber(target_fiber_species1[primate_choose_fiber],
                                                                              feat_code[0], k, distance_pattern)
        print(f"最近的纤维索引: {nearest_fiber_index}")
        print(f"最小欧氏距离: {min_distance}")

        scene = window.Scene()
        scene.SetBackground(1, 1, 1)

        tot_h = 0
        tot_p = 0
        human_fiber = []
        for i in range(1000):
            h_fiber = np.load(cmp_root_sub + 'cluser_{}.npy'.format(i))
            tot_h += h_fiber.shape[0]
            if i < 500:
                p_fiber = np.load('F:/njust/wuye/week08/vgae_pytorch-master/cmp_data_v2/sub-001/cluser_{}.npy'.format(i))
                tot_p += p_fiber.shape[0]
                if i in nearest_fiber_index:
                    scene.add(actor.line(h_fiber, colors=(0, 1, 0)))
                    human_fiber.append(h_fiber)
                else:
                    scene.add(actor.line(h_fiber, colors=(1, 0, 0)))
            if i == primate_choose_fiber:
                scene.add(actor.line(p_fiber, colors=(0, 1, 0)))
                primate_fiber = p_fiber
                print("猴子的纤维尺寸:" + str(primate_fiber.shape))
                print("猴子的纤维索引:" + str(primate_choose_fiber))
            else:
                scene.add(actor.line(p_fiber, colors=(0, 0, 1)))

        print(tot_h, tot_p)
        window.show(scene)

        D33 = 1.0
        D44 = 0.02
        t = 1
        z = EnhancementKernel(D33, D44, t)

        # bundle_name = []
        # file_path = "F:/njust/wuye/week03/some_tracts/TOM_trackings/"
        # # 遍历文件夹及其子文件夹中的所有文件
        # for root, _, files in os.walk(file_path):
        #     for file in files:
        #         bundle_name.append(file[:-4])
        # num_dict = {}
        for idx in range(k):
            human_near = np.load(cmp_root_sub + 'cluser_{}.npy'.format(nearest_fiber_index[idx]))

            # cluster_path = 'F:/njust/wuye/week08/vgae_pytorch-master/cmp_data_v2/sub-202/cluser_{}.txt'.format(nearest_fiber_index[idx])
            # with open(cluster_path, 'r') as file:
            #     content = file.read()
            # # 将文本分割成整数列表
            # numbers = [int(x) for x in content.split()]
            # for n in numbers:
            #     if bundle_name[n-1] not in num_dict:
            #         num_dict[bundle_name[n-1]] = 1
            #     else:
            #         num_dict[bundle_name[n-1]] += 1
            scene = window.Scene()
            scene.SetBackground(1, 1, 1)
            print("第{}近的纤维束索引：".format(idx + 1) + str(nearest_fiber_index[idx]))
            print("第{}近的纤维束尺寸：".format(idx + 1) + str(human_near.shape))
            print("第{}近的纤维束距离：".format(idx + 1) + str(min_distance[idx]))
            # print(num_dict)
            # max_value = max(num_dict.values())
            # total_sum = sum(num_dict.values())
            # result = max_value / total_sum
            # print("precision: " + str(result))
            # if idx == 2 or idx == 4 or idx == 9:
            #     # 提取字典的键和值作为直方图的数据
            #     labels = list(num_dict.keys())
            #     values = list(num_dict.values())
            #     # 绘制直方图
            #     plt.bar(labels, values)
            #     # 添加标题和标签
            #     plt.title('Top'+str(idx+1))
            #     plt.xlabel('bundle_result')
            #     plt.ylabel('streamline number')
            #     # 显示图形
            #     plt.show()

            fbc = FBCMeasures(primate_fiber, z)
            fbc_sl_thres, clrs_thres, rfbc_thres = fbc.get_points_rfbc_thresholded(0.5, emphasis=0.01)
            scene.add(actor.line(fbc_sl_thres))  # np.vstack(clrs_thres),

            fbc2 = FBCMeasures(human_near, z)
            fbc_sl_thres2, clrs_thres2, rfbc_thres2 = fbc2.get_points_rfbc_thresholded(0.5, emphasis=0.01)
            scene.add(actor.line(fbc_sl_thres2))  # np.vstack(clrs_thres),
            window.show(scene)