from torch.utils.data import Dataset
import numpy as np
from collections import Counter
import scipy.sparse
import torch
import torch.utils.data as data
import os
import scipy.sparse
import random
# 设置随机数种子
np.random.seed(123)
random.seed(123)

def load_labels(label_file):  # please run parsing_label.py first to get the numerical label file (.txt)
    return np.loadtxt(label_file)


def npz_reader(file_name):
    print('load npz matrix:', file_name)
    data = scipy.sparse.load_npz(file_name)
    return data
    
def read_from_file(data_path, label_path = None, protien_path = None):
    data_path = os.path.join(os.path.realpath('.'), data_path)

    labels = None
    input_size, input_size_protein = 0, 0
    
    data_reader = npz_reader(data_path) 
    protein_reader = None
    if label_path is not None:        
        label_path = os.path.join(os.path.realpath('.'), label_path)    
        labels = load_labels(label_path)
    if protien_path is not None:
        protien_path = os.path.join(os.path.realpath('.'), protien_path)    
        protein_reader = npz_reader(protien_path)
        
    return data_reader, labels, protein_reader

def generate_rna_atac_pairs(rna_labels, atac_labels):
    # 统计共同的类别数量
    rna_label_set = set(rna_labels)
    atac_label_set = set(atac_labels)
    common_labels = rna_label_set.intersection(atac_label_set)
    num_common_labels = len(common_labels)
    #print("num_classes:", num_common_labels)

    # 从rna和atac中采样共同类别的pair
    pairs = []
    for label in common_labels:
        rna_indices = np.where(rna_labels == label)[0]
        atac_indices = np.where(atac_labels == label)[0]
        num_pairs = min(len(rna_indices), len(atac_indices))
        if num_pairs > 0:
            np.random.seed(123)
            rna_sample_indices = np.random.choice(rna_indices, size=num_pairs, replace=False)
            np.random.seed(123)
            atac_sample_indices = np.random.choice(atac_indices, size=num_pairs, replace=False)
            for i in range(num_pairs):
                rna_index = rna_sample_indices[i]
                atac_index = atac_sample_indices[i]
                pairs.append((rna_index, atac_index, label))

    label_mapping = {label: i for i, label in enumerate(sorted(common_labels))}
    mapped_pairs = [(rna_index, atac_index, label_mapping[label]) for rna_index, atac_index, label in pairs]
    
    label_set = set()
    for pair in mapped_pairs:
        _, _, label = pair
        label_set.add(label)

    num_classes = len(label_set)
    
    random.seed(123)
    random.shuffle(mapped_pairs)

    # 按需选择类别并放在最前面
    selected_pairs = []
    selected_labels = set()

    for pair in mapped_pairs:
        _, _, label = pair
        if len(selected_labels) < num_classes and label not in selected_labels:
            selected_pairs.append(pair)
            selected_labels.add(label)
    # 将剩余的tuple添加到selected_pairs中
    remaining_pairs = [pair for pair in mapped_pairs if pair not in selected_pairs]
    selected_pairs.extend(remaining_pairs)

    assert len(selected_pairs) == len(mapped_pairs)
    return selected_pairs,num_common_labels

class CITE_ASAP(Dataset):    # length: 3662, num_classes: 7, ,train: 2930, test: 732, feature_dim = 17441
    def __init__(self, partition='train', n_labeled = 200, labeled_dataset = True):

        self.n_labeled = n_labeled
        self.labeled_dataset = labeled_dataset
        self.rna_paths = '/home/zf/dataset/data_cite/citeseq_control_rna.npz' # RNA gene expression from CITE-seq data
        self.rna_label_path = '/home/zf/dataset/data_cite/citeseq_control_cellTypes.txt' # CITE-seq data cell type labels (coverted to numeric) 
        self.atac_paths = '/home/zf/dataset/data_cite/asapseq_control_atac.npz' # ATAC gene activity matrix from ASAP-seq data
        self.atac_label_path = '/home/zf/dataset/data_cite/asapseq_control_cellTypes.txt' # ASAP-seq data cell type labels (coverted to numeric) 

        self.rna, rna_labels, _ = read_from_file(self.rna_paths, self.rna_label_path) 
        self.atac, atac_labels, _ = read_from_file(self.atac_paths, self.atac_label_path) 
        
        self.mapped_pairs,self.num_classes = generate_rna_atac_pairs(rna_labels,atac_labels)

        test_size = int(0.2*len(self.mapped_pairs))
        if 'test' in partition.lower():
            self.mapped_pairs = self.mapped_pairs[-test_size::]
        elif 'train' in partition.lower():
            self.mapped_pairs = self.mapped_pairs[0: -test_size]
            if self.labeled_dataset:
                self.mapped_pairs = self.mapped_pairs[0:self.n_labeled]
            else:
                self.mapped_pairs = self.mapped_pairs[self.n_labeled:]
            
    def __getitem__(self, item):
        rna_idx,atac_idx,label = self.mapped_pairs[item]
        rna = self.rna[rna_idx]
        atac =self.atac[atac_idx]
    
        return torch.tensor(rna.toarray()).squeeze(0).float(), torch.tensor(atac.toarray()).squeeze(0).float(), torch.tensor(label).to(torch.int64), item
    
    def __len__(self):
        return len(self.mapped_pairs)


class snRNA_snATAC(Dataset):   # length: 7904, num_classes: 18, train: 6324, test: 1580, feature_dim = 18603
    def __init__(self, partition='train', n_labeled = 200, labeled_dataset = True):
        self.n_labeled = n_labeled
        self.labeled_dataset = labeled_dataset
        self.rna_paths = '/home/zf/dataset/data_MOp/YaoEtAl_RNA_snRNA_10X_v3_B_exprs.npz' 
        self.rna_label_path = '/home/zf/dataset/data_MOp/YaoEtAl_RNA_snRNA_10X_v3_B_cellTypes.txt'
        self.atac_paths = '/home/zf/dataset/data_MOp/YaoEtAl_ATAC_exprs.npz' 
        self.atac_label_path = '/home/zf/dataset/data_MOp/YaoEtAl_ATAC_cellTypes.txt' 

        self.rna, rna_labels, _ = read_from_file(self.rna_paths, self.rna_label_path) 
        self.atac, atac_labels, _ = read_from_file(self.atac_paths, self.atac_label_path) 
        
        self.mapped_pairs,self.num_classes = generate_rna_atac_pairs(rna_labels,atac_labels)

        test_size = int(0.2*len(self.mapped_pairs))
        if 'test' in partition.lower():
            self.mapped_pairs = self.mapped_pairs[-test_size::]
        elif 'train' in partition.lower():
            self.mapped_pairs = self.mapped_pairs[0: -test_size]
            if self.labeled_dataset:
                self.mapped_pairs = self.mapped_pairs[0:self.n_labeled]
            else:
                self.mapped_pairs = self.mapped_pairs[self.n_labeled:]
            
    def __getitem__(self, item):
        rna_idx,atac_idx,label = self.mapped_pairs[item]
        rna = self.rna[rna_idx]
        atac =self.atac[atac_idx]
    
        return torch.tensor(rna.toarray()).squeeze(0).float(), torch.tensor(atac.toarray()).squeeze(0).float(), torch.tensor(label).to(torch.int64), item
    
    def __len__(self):
        return len(self.mapped_pairs)
    
class snRNA_snmC(Dataset):   # length: 8270, num_classes: 17 ,train: 6616, test: 1654, feature_dim = 18603
    def __init__(self, partition='train', n_labeled = 200, labeled_dataset = True):
        self.n_labeled = n_labeled
        self.labeled_dataset = labeled_dataset
        self.rna_paths = '/home/zf/dataset/data_MOp/YaoEtAl_RNA_snRNA_10X_v3_B_exprs.npz' 
        self.rna_label_path = '/home/zf/dataset/data_MOp/YaoEtAl_RNA_snRNA_10X_v3_B_cellTypes.txt'
        self.atac_paths = '/home/zf/dataset/data_MOp/YaoEtAl_snmC_exprs.npz'
        self.atac_label_path = '/home/zf/dataset/data_MOp/YaoEtAl_snmC_cellTypes.txt' 

        self.rna, rna_labels, _ = read_from_file(self.rna_paths, self.rna_label_path) 
        self.atac, atac_labels, _ = read_from_file(self.atac_paths, self.atac_label_path) 
        
        self.mapped_pairs,self.num_classes = generate_rna_atac_pairs(rna_labels,atac_labels)

        test_size = int(0.2*len(self.mapped_pairs))
        if 'test' in partition.lower():
            self.mapped_pairs = self.mapped_pairs[-test_size::]
        elif 'train' in partition.lower():
            self.mapped_pairs = self.mapped_pairs[0: -test_size]
            if self.labeled_dataset:
                self.mapped_pairs = self.mapped_pairs[0:self.n_labeled]
            else:
                self.mapped_pairs = self.mapped_pairs[self.n_labeled:]
            
    def __getitem__(self, item):
        rna_idx,atac_idx,label = self.mapped_pairs[item]
        rna = self.rna[rna_idx]
        atac =self.atac[atac_idx]
    
        return torch.tensor(rna.toarray()).squeeze(0).float(), torch.tensor(atac.toarray()).squeeze(0).float(), torch.tensor(label).to(torch.int64), item
    
    def __len__(self):
        return len(self.mapped_pairs)
    

if __name__ == '__main__':
    cite_asap_train = CITE_ASAP('train',400, labeled_dataset = True)
    print('labeled train_dataset length:',len(cite_asap_train))
    label_counts = Counter(pair[2] for pair in cite_asap_train.mapped_pairs)
    # 输出类别出现次数
    print("labeled train_dataset class nums:")
    for label, count in label_counts.items():
        print("Label:", label, "Count:", count)
        
    cite_asap_train_unlabeled = CITE_ASAP('train',400, labeled_dataset = False)     
    print('unlabeled train_dataset length:',len(cite_asap_train_unlabeled))
    label_counts = Counter(pair[2] for pair in cite_asap_train_unlabeled.mapped_pairs)
    # 输出类别出现次数
    print("unlabeled train_dataset class nums:")
    for label, count in label_counts.items():
        print("Label:", label, "Count:", count)
        
    cite_asap_test = CITE_ASAP('test')
    print('test_dataset length:',len(cite_asap_test))
    label_counts = Counter(pair[2] for pair in cite_asap_test.mapped_pairs)
    # 输出类别出现次数
    print("test_dataset class nums:")
    for label, count in label_counts.items():
        print("Label:", label, "Count:", count)
        
        
    cite_asap_train = snRNA_snATAC('train',400, labeled_dataset = True)
    print('labeled train_dataset length:',len(cite_asap_train))
    label_counts = Counter(pair[2] for pair in cite_asap_train.mapped_pairs)
    # 输出类别出现次数
    print("labeled train_dataset class nums:")
    for label, count in label_counts.items():
        print("Label:", label, "Count:", count)
        
    cite_asap_train_unlabeled = snRNA_snATAC('train',400, labeled_dataset = False)     
    print('unlabeled train_dataset length:',len(cite_asap_train_unlabeled))
    label_counts = Counter(pair[2] for pair in cite_asap_train_unlabeled.mapped_pairs)
    # 输出类别出现次数
    print("unlabeled train_dataset class nums:")
    for label, count in label_counts.items():
        print("Label:", label, "Count:", count)
        
    cite_asap_test = snRNA_snATAC('test')
    print('test_dataset length:',len(cite_asap_test))
    label_counts = Counter(pair[2] for pair in cite_asap_test.mapped_pairs)
    # 输出类别出现次数
    print("test_dataset class nums:")
    for label, count in label_counts.items():
        print("Label:", label, "Count:", count)
        

    cite_asap_train = snRNA_snmC('train',400, labeled_dataset = True)
    print('labeled train_dataset length:',len(cite_asap_train))
    label_counts = Counter(pair[2] for pair in cite_asap_train.mapped_pairs)
    # 输出类别出现次数
    print("labeled train_dataset class nums:")
    for label, count in label_counts.items():
        print("Label:", label, "Count:", count)
        
    cite_asap_train_unlabeled = snRNA_snmC('train',400, labeled_dataset = False)     
    print('unlabeled train_dataset length:',len(cite_asap_train_unlabeled))
    label_counts = Counter(pair[2] for pair in cite_asap_train_unlabeled.mapped_pairs)
    # 输出类别出现次数
    print("unlabeled train_dataset class nums:")
    for label, count in label_counts.items():
        print("Label:", label, "Count:", count)
        
    cite_asap_test = snRNA_snmC('test')
    print('test_dataset length:',len(cite_asap_test))
    label_counts = Counter(pair[2] for pair in cite_asap_test.mapped_pairs)
    # 输出类别出现次数
    print("test_dataset class nums:")
    for label, count in label_counts.items():
        print("Label:", label, "Count:", count)