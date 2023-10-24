import os
import pandas as pd
from torch.autograd import Variable
from sklearn import metrics
from sklearn.model_selection import KFold
from cnn_model import *
from model import *
from torch_geometric.nn import GCNConv 
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from torch.nn.parameter import Parameter
import matplotlib.pyplot as plt
from sklearn.manifold import TSNE
from biotransformers import BioTransformers
import numpy as np
from sklearn import metrics
# Path
Dataset_Path = "./Dataset/"
Model_Path = "./Model/"


def train_one_epoch(model, data_loader):
    epoch_loss_train = 0.0
    train_loss=[]
    crossentropyloss=nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr = LEARNING_RATE, weight_decay = WEIGHT_DECAY)
    n = 0
    all_embedds = []  
    all_labels = [] 
    for data in data_loader:
        optimizer.zero_grad()
        _, _, embedds, labels = data

        if torch.cuda.is_available():
            feature = Variable(embedds.cuda())
            y_true = Variable(labels.cuda())
        else:
            feature = Variable(embedds)
            y_true = Variable(labels)

        feature = torch.squeeze(feature)
        y_true = torch.squeeze(y_true)

        y_pred = model(feature)  

      
        loss = crossentropyloss(y_pred, y_true.long())
    
        # backward gradient
        loss.backward()

        # update all parameters
        optimizer.step()

        epoch_loss_train += loss.item()
        n += 1


    epoch_loss_train_avg = epoch_loss_train / n


    return epoch_loss_train_avg


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
            sequence_names, _, embedds, labels  = data

            if torch.cuda.is_available():
                
                feature = Variable(embedds.cuda())
                y_true = Variable(labels.cuda())
            else:
                feature = Variable(embedds)
                y_true = Variable(labels)

            feature= torch.squeeze(feature)
            y_true = torch.squeeze(y_true)

            y_pred = model(feature)
            loss = crossentropyloss(y_pred, y_true.long())

            softmax = torch.nn.Softmax(dim=1)
            y_pred = softmax(y_pred)
            
            y_pred = y_pred.cpu().detach().numpy()
            y_true = y_true.cpu().detach().numpy()
            valid_pred += [pred[1] for pred in y_pred]
            valid_true += list(y_true)
            pred_dict[sequence_names[0]] = [pred[1] for pred in y_pred]

            epoch_loss += loss.item()
            n += 1
    epoch_loss_avg = epoch_loss / n


    return epoch_loss_avg, valid_true, valid_pred, pred_dict


def analysis(y_true, y_pred, best_threshold = None):
    if best_threshold == None:
        best_f1 = 0
        best_threshold = 0
        for threshold in range(0, 100):
            threshold = threshold / 100
            binary_pred = [1 if pred >= threshold else 0 for pred in y_pred]
            binary_true = y_true
            f1 = metrics.f1_score(binary_true, binary_pred)
            if f1 > best_f1:
                best_f1 = f1
                best_threshold = threshold

    binary_pred = [1 if pred >= best_threshold else 0 for pred in y_pred]
    binary_true = y_true

    # binary evaluate
    binary_acc = metrics.accuracy_score(binary_true, binary_pred)
    precision = metrics.precision_score(binary_true, binary_pred)
    recall = metrics.recall_score(binary_true, binary_pred)
    f1 = metrics.f1_score(binary_true, binary_pred)
    AUC = metrics.roc_auc_score(binary_true, y_pred)
    precisions, recalls, thresholds = metrics.precision_recall_curve(binary_true, y_pred)
    AUPRC = metrics.auc(recalls, precisions)
    mcc = metrics.matthews_corrcoef(binary_true, binary_pred)
    tn, fp, fn, tp = metrics.confusion_matrix(y_true, binary_pred).ravel()
    spe = tn / (tn + fp)

    # plot_roc_curve(binary_true,binary_pred)
    results = {
        'binary_acc': binary_acc,
        'precision': precision,
        'recall': recall,
        'f1': f1,
        'AUC': AUC,
        'AUPRC': AUPRC,
        'mcc': mcc,
        'threshold': best_threshold,
        'spe':spe
    }
    return results


def train(model, train_dataframe, valid_dataframe, fold = 0):
    train_loader = DataLoader(dataset=ProDataset(train_dataframe), batch_size=BATCH_SIZE, shuffle=True, num_workers=0)
    valid_loader = DataLoader(dataset=ProDataset(valid_dataframe), batch_size=BATCH_SIZE, shuffle=False, num_workers=0)

    best_epoch = 0
    best_val_auc = 0
    best_val_aupr = 0
    train_acc = []
    valid_acc = []
    valid_loss= []
    train_loss= []
    for epoch in range(NUMBER_EPOCHS):
        print("\n========== Train epoch " + str(epoch + 1) + " ==========")
        model.train()

        epoch_loss_train_avg = train_one_epoch(model, train_loader)
        print("========== Evaluate Train set ==========")
        _, train_true, train_pred, _ = evaluate(model, train_loader)
        result_train = analysis(train_true, train_pred, 0.5)
        print("Train loss: ", epoch_loss_train_avg)
        print("Train binary acc: ", result_train['binary_acc'])
        print("Train AUC: ", result_train['AUC'])
        print("Train AUPRC: ", result_train['AUPRC'])

        train_loss.append(epoch_loss_train_avg)
        f = open("./train_loss.txt", 'w') 
        f.write(str(train_loss))

        train_acc.append(result_train['binary_acc'])  
        f = open("./train_acc.txt", 'w')
        f.write(str(train_acc))

        print("========== Evaluate Valid set ==========")
        epoch_loss_valid_avg, valid_true, valid_pred, _ = evaluate(model, valid_loader)
        result_valid = analysis(valid_true, valid_pred, 0.5)

       

        print("Valid loss: ", epoch_loss_valid_avg)
        print("Valid binary acc: ", result_valid['binary_acc'])
        print("Valid precision: ", result_valid['precision'])
        print("Valid recall: ", result_valid['recall'])
        print("Valid f1: ", result_valid['f1'])
        print("Valid AUC: ", result_valid['AUC'])
        print("Valid AUPRC: ", result_valid['AUPRC'])
        print("Valid mcc: ", result_valid['mcc'])

        valid_loss.append(epoch_loss_valid_avg)
      
        f1 = open("./valid_loss.txt", 'w') 
        f1.write(str(valid_loss))

        valid_acc.append(result_valid['binary_acc'])
        f1 = open("./valid_acc.txt", 'w')
        f1.write(str(valid_acc))

        if best_val_auc < result_valid['AUC']:
            best_epoch = epoch + 1
            best_val_auc = result_valid['AUC']
            best_val_aupr = result_valid['AUPRC']
            torch.save(model.state_dict(), os.path.join(Model_Path, 'Fold' + str(fold) + '_best_model.pkl'))

    return best_epoch, best_val_auc, best_val_aupr


def cross_validation(all_dataframe, fold_number = 5):
    print("Random seed:", SEED)
    print("Learning rate:", LEARNING_RATE)
    print("Training epochs:", NUMBER_EPOCHS)

    sequence_names = all_dataframe['ID'].values
    sequence_labels = all_dataframe['label'].values
    kfold = KFold(n_splits = fold_number, shuffle = True)
    fold = 0
    best_epochs = []
    valid_aucs = []
    valid_auprs = []

    for train_index, valid_index in kfold.split(sequence_names, sequence_labels):
        print("\n\n========== Fold " + str(fold + 1) + " ==========")
        train_dataframe = all_dataframe.iloc[train_index, :]
        valid_dataframe = all_dataframe.iloc[valid_index, :]
        print("Train on", str(train_dataframe.shape[0]), "samples, validate on", str(valid_dataframe.shape[0]),
              "samples")
        model = ConvNetWithSelfAttention()

        if torch.cuda.is_available():
            model.cuda()

        best_epoch, valid_auc, valid_aupr = train(model, train_dataframe, valid_dataframe, fold + 1)
        best_epochs.append(str(best_epoch))
        valid_aucs.append(valid_auc)
        valid_auprs.append(valid_aupr)
        fold += 1
    # print(model)
    
    # for param in model.parameters():
    #     print(type(param.data), param.size())
    print("\n\nBest epoch: " + " ".join(best_epochs))
    print("Average AUC of {} fold: {:.4f}".format(fold_number, sum(valid_aucs) / fold_number))
    for i in valid_aucs:
        print(i)
        print('\n')
    return round(sum([int(epoch) for epoch in best_epochs]) / fold_number)

def test(test_dataframe):
    test_loader = DataLoader(dataset=ProDataset(test_dataframe), batch_size=BATCH_SIZE, shuffle=False, num_workers=0)

    for model_name in sorted(os.listdir(Model_Path)):
        print(model_name)
        model = ConvNet()
        #MyCNN()
        if torch.cuda.is_available():
            model.cuda()
        model.load_state_dict(torch.load(Model_Path + model_name, map_location='cuda:0'))

        epoch_loss_test_avg, test_true, test_pred, dict = evaluate(model, test_loader)
        result_test = analysis(test_true, test_pred,0.5)
        print("========== Evaluate Test set ==========")
        print("Test binary acc: ", result_test['binary_acc'])
        print("Test precision:", result_test['precision'])
        print("Test recall: ", result_test['recall'])
        print("Test spe: ", result_test['spe'])
        print("Test f1: ", result_test['f1'])
        print("Test AUC: ", result_test['AUC'])
        print("Test AUPRC: ", result_test['AUPRC'])
        print("Test mcc: ", result_test['mcc'])
        print("Threshold: ", result_test['threshold'])
        
            # Export prediction
        # with open(model_name.split(".")[0] + "_pred.pkl", "wb") as f:
            # pickle.dump(pred_dict, f)
    return test_true, test_pred, dict

def plot_roc_curve(y_true, y_pred,y_true1,y_pred1):
    fpr, tpr, thresholds = metrics.roc_curve(y_true, y_pred)
    roc_auc = metrics.auc(fpr, tpr)
    fpr1, tpr1, thresholds = metrics.roc_curve(y_true1, y_pred1)
    roc_auc1 = metrics.auc(fpr1, tpr1)
   
    plt.figure()
    lw = 2
    # label='GCN ROC curve (area = %0.2f)' % roc_auc
    plt.plot(fpr, tpr, color='Red', lw=lw, label='MetalPrognosis ROC curve (area = %0.3f)' %roc_auc)
    plt.plot(fpr1, tpr1, color='darkorange', lw=lw, label='MCCNN ROC curve (area = %0.3f)' %roc_auc1)
    plt.plot([0, 1], [0, 1], color='navy', lw=lw, linestyle='--')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate',fontname='Arial')
    plt.ylabel('True Positive Rate',fontname='Arial')
    plt.tick_params(labelsize=10)
    plt.title('Receiver Operating Characteristic',fontsize = 10)
    plt.legend(loc="lower right")
    plt.show()
    plt.savefig('roc1.png')
    




def main():
    
    with open("./neg_dic_85M.pkl", "rb") as f:
        neg_dic = pickle.load(f)
    with open("./ca_dic_85M.pkl", "rb") as f:
        ca_dic = pickle.load(f)
    
    neg_dataframe = pd.DataFrame(neg_dic)
    
    ca_dataframe = pd.DataFrame(ca_dic)
  
    # .iloc[:100],ca_dataframe.iloc[:100],mg_dataframe.iloc[:100]
    all_dataframe = pd.concat([ca_dataframe.iloc[:150],neg_dataframe], axis=0)
    all_dataframe.reset_index(drop=True, inplace=True)
    print(all_dataframe)

    aver_epoch = cross_validation(all_dataframe, fold_number = 4)
def main1():
    with open("./testpos_dic_85M.pkl", "rb") as f:
        pos_dic = pickle.load(f)  
    with open("./test_neg_dic_85M.pkl", "rb") as f:
        neg_dic = pickle.load(f)
    mccnn_valid_true=np.load('mccnn_actual.npy')
    mccnn_valid_pre=np.load('mccnn_pred.npy')
    neg_dataframe = pd.DataFrame(neg_dic)
    dict1 = {}
    pos_dataframe = pd.DataFrame(pos_dic)
    
    test_dataframe = pd.concat([pos_dataframe,neg_dataframe], axis=0)
    test_dataframe.reset_index(drop=True, inplace=True)
    print(test_dataframe)
    test_true2,test_pred2,dict1=test(test_dataframe)
    plot_roc_curve(test_true2,test_pred2,mccnn_valid_true,mccnn_valid_pre)
    # print(test_pred2)
    # print(test_true2)
    # print(dict1)

if __name__ == "__main__":
    main1()
