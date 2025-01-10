from torch.utils.data import Dataset

from utils.input_data import load_fiber_data


class GraphDataset(Dataset):
    def __init__(self, args):
        self.adjs, self.features = load_fiber_data(args.human_root_path, args.primate_root_path, args.human_sub_num, args.human_start_num, args.primate_sub_num, args.primate_start_num)
    def __len__(self):
        return len(self.adjs)

    def __getitem__(self, index):
        adjacency_matrix = self.adjs[index]
        feature = self.features[index]
        return feature, adjacency_matrix