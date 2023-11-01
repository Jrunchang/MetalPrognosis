import torch
import math
from torch import nn, optim
import torch.nn.functional as F
from torch.nn.utils.rnn import pack_padded_sequence
from torchsummary import summary
import pickle
import os,argparse
import pandas as pd
from torch.autograd import Variable
from sklearn import metrics
from sklearn.model_selection import KFold
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from torch.nn.parameter import Parameter
from sklearn.manifold import TSNE
import numpy as np
from sklearn import metrics
import scipy.sparse as sp
import torch.nn as nn
from biotransformers import BioTransformers
from Bio import SeqIO
from Bio.PDB import PDBParser
from Bio.Seq import Seq
from Bio.SeqRecord import SeqRecord
from Bio.SeqUtils import seq3


# Path
Dataset_Path = "./Dataset/"
Model_Path = "./Model/"
NUM_CLASSES = 2 # [disease,benign]
NUMBER_EPOCHS = 100
BATCH_SIZE = 16
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')



class ConvNet(nn.Module):
    def __init__(self):
        super(ConvNet, self).__init__()
        self.conv1 = nn.Conv2d(in_channels=1, out_channels=16, kernel_size=3, stride=1, padding=1)
        self.conv2 = nn.Conv2d(in_channels=16, out_channels=32, kernel_size=3, stride=1, padding=1)
        self.relu = nn.ReLU()
        self.bn1 = nn.BatchNorm2d(16)
        self.bn2 = nn.BatchNorm2d(32)
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2)
        self.fc = nn.Linear(32 * 3 * 192, 2)  
        self.dropout = nn.Dropout(p=0.4)
        #32 * 3 * 192,256,448
    def forward(self, x):
        # Input size of x is [batch_size, 1, 15, 1280].
        x = x.unsqueeze(1) 
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.pool(x)
        x = self.conv2(x)
        x = self.bn2(x)
        x = self.relu(x)
        x = self.pool(x)
        x = x.view(x.size(0), -1)
        x = self.fc(x)
        x = self.dropout(x)
        return x
class ProDataset(Dataset):

    def __init__(self, dataframe):
        self.sites = dataframe['ID'].values
        self.sequences = dataframe['sequence'].values
        self.embedds = dataframe['embedd'].values

    def __getitem__(self, index):
        site_name = self.sites[index]
        sequence = self.sequences[index]
        embedd = self.embedds[index]  

        return site_name, sequence, embedd

    def __len__(self):
        return len(self.sequences)
def evaluate(model, data_loader):
    model.eval()
    crossentropyloss=nn.CrossEntropyLoss()
    epoch_loss = 0.0
    n = 0
    valid_pred = []
    valid_true = []
    pred_dict = {}
    valid_loss = []
    for data in data_loader:
        with torch.no_grad():
            sequence_names, sequence1, embedds  = data

            if torch.cuda.is_available():
                
                feature = Variable(embedds.cuda())
            else:
                feature = Variable(embedds)

            feature= torch.squeeze(feature)

            y_pred = model(feature)

            softmax = torch.nn.Softmax(dim=1)
            y_pred = softmax(y_pred)
            
            y_pred = y_pred.cpu().detach().numpy()

            valid_pred += [pred[1] for pred in y_pred]
            pred_dict[sequence_names[0]] = [pred[1] for pred in y_pred]

            n += 1


    return valid_pred, pred_dict
def test(test_dataframe):
    test_loader = DataLoader(dataset=ProDataset(test_dataframe), batch_size=BATCH_SIZE, shuffle=False, num_workers=0)

    for model_name in sorted(os.listdir(Model_Path)):
        print(model_name)
        model = ConvNet()
        #MyCNN()
        if torch.cuda.is_available():
            model.cuda()
        model.load_state_dict(torch.load(Model_Path + model_name, map_location='cuda:0'))

        test_pred, dict = evaluate(model, test_loader)
        
            # Export prediction
        # with open(model_name.split(".")[0] + "_pred.pkl", "wb") as f:
            # pickle.dump(pred_dict, f)
    return  test_pred, dict
def parse_pdb_to_fasta(pdb_file):
  
    three_to_one_letter = {"ALA": "A","ARG": "R","ASN": "N","ASP": "D","CYS": "C","GLN": "Q","GLU": "E","GLY": "G","HIS": "H", "ILE": "I",
                           "LEU": "L","LYS": "K","MET": "M","PHE": "F","PRO": "P","SER": "S","THR": "T", "TRP": "W","TYR": "Y","VAL": "V"}
    parser = PDBParser(QUIET=True)
    structure = parser.get_structure("protein", pdb_file)

    protein_seq = ""
    for model in structure:
        for chain in model:
            for residue in chain:
                if residue.get_id()[0] == ' ':
                    three_letter_code = residue.get_resname()
                    one_letter_code = three_to_one_letter.get(three_letter_code, "X")
                    protein_seq += one_letter_code

    # create a Seq object.
    protein_seq = Seq(protein_seq)
    protein_seq =str(protein_seq)

    return protein_seq


def process_fasta(fasta1,site):
    seqList = []  
    idList = []
    seq15list = []
    site = site.split(',')
    for seq_record in SeqIO.parse(fasta1, "fasta"):
        seq = str(seq_record.seq).upper()
        id = str(seq_record.id)
        seqList.append(str(seq))
        idList.append(str(id))
    for k,si in enumerate(site):
        si = int(si)
        start_pos = max(1, si - 7)
        end_pos = min(len(seqList[k]), si + 7)
        center_sequence = seqList[k][start_pos-1:end_pos]
        if len(center_sequence) < 15:
            add_aa = 'A' * (15 - len(center_sequence))
            center_sequence = center_sequence + add_aa
        seq15list.append(center_sequence)

    return idList,seq15list


def process_another(seq,site):
 
    seq15list = []
    site = site.split(',')
    for si in (site):
        si = int(si)
        start_pos = max(1, si - 7)
        end_pos = min(len(seq), si + 7)
        center_sequence = seq[start_pos-1:end_pos]
        if len(center_sequence) < 15:
            add_aa = 'A' * (15 - len(center_sequence))
            center_sequence = center_sequence + add_aa
        seq15list.append(center_sequence)
        


    return seq15list

def main():
    dict_res={}
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", type = str, help = "Input file (PDB or FASTA)")
    parser.add_argument("--site", type = str, help = "metal binding site")
    parser.add_argument("--outpath", type = str, help = "Output path to save disease-associated predictions")
    args = parser.parse_args()
    idList,seq15list = [],[]
    if args.input.endswith(".pdb"):
        pdb_file = args.input
        id = pdb_file.split('.')[0]
        seqold = parse_pdb_to_fasta(pdb_file)
        seq15list = process_another(seqold,args.site)
        for i in range(len(seq15list)):
            idList.append(id)

    elif args.input.endswith(".fasta"):
        idList,seq15list = process_fasta(args.input,args.site)

    else:
        print("Please input the PDB or FASTA format")
    #################################################################################################################################################
    AAs = []
    outp = args.outpath
    print(outp)
    positions = args.site.split(',')
    for i in seq15list:
        AA = i[7]
        AAs.append(AA)
    test_dic = {"ID": idList, "sequence": seq15list,"embedd":[]}
    print(test_dic)
    bio_trans = BioTransformers(backend="esm1_t6_43M_UR50S")
    embedds = []
    for seq1 in seq15list:
        embedding = bio_trans.compute_embeddings([seq1], pool_mode=('full'),batch_size=1,silent = False)
        embedding = embedding['full']
        embedding = np.array(embedding,dtype=np.float32)#转为np，shape查看维度
        embedding = np.squeeze(embedding, axis=0) 
        embedds.append(embedding)

    test_dic['embedd'] = embedds
    # print(embedds[0].shape)

    test_dataframe = pd.DataFrame(test_dic)
    print(test_dataframe)
    test_pred2,dict1=test(test_dataframe)
    print(test_pred2)
    result = []
    for i in test_pred2:
        if float(i)>=0.5:
            result.append('Yes')
        else:
            result.append('No')

    dict_res = {"Rank":[i+1 for i in range(len(AAs))],"Uniprot ID":idList,"AA":AAs,"position":positions,"Probability":test_pred2,"Prediction Result":result}
    # print(dict_res)
    df = pd.DataFrame(dict_res)
    df.to_csv(outp,index=False)



if __name__ == "__main__":
    main()
