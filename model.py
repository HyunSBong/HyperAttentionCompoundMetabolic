import torch
import torch.nn as nn
import torch.nn.functional as F

class RMSELoss(nn.Module):
    def __init__(self, eps=1e-6):
        super().__init__()
        self.mse = nn.MSELoss()
        self.eps = eps
        
    def forward(self,yhat,y):
        loss = torch.sqrt(self.mse(yhat,y) + self.eps)
        return loss

class RMSLELoss(nn.Module):
    def __init__(self):
        super().__init__()
        self.mse = nn.MSELoss()
        
    def forward(self, pred, actual):
        return torch.sqrt(self.mse(torch.log(pred + 1), torch.log(actual + 1)))
    
class AttentionDTI(nn.Module):
    def __init__(self,hp,
                 protein_MAX_LENGH = 1000,
                 drug_MAX_LENGH = 100):
        super(AttentionDTI, self).__init__()
        self.dim = hp.char_dim
        self.conv = hp.conv
        self.f_dim = hp.f_dim
        self.f_out_dim = hp.f_out_dim
        self.drug_MAX_LENGH = drug_MAX_LENGH
        self.drug_kernel = hp.drug_kernel
        self.f_layer = hp.f_layer
        self.batch_size = hp.Batch_size
        
        ### Embedding layer ###
        self.drug_embed = nn.Embedding(65, self.dim, padding_idx=0)
        
        ### Smiles ###
        self.Drug_CNNs = nn.Sequential(
            nn.Conv1d(in_channels=self.dim, out_channels= self.conv,  kernel_size=self.drug_kernel[0]),
            nn.ReLU(),
            nn.Conv1d(in_channels=self.conv, out_channels= self.conv*2,  kernel_size=self.drug_kernel[1]),
            nn.ReLU(),
            nn.Conv1d(in_channels=self.conv*2, out_channels= self.conv*4,  kernel_size=self.drug_kernel[2]),
            nn.ReLU(),
        )   
        self.Drug_max_pool = nn.MaxPool1d(self.drug_MAX_LENGH-self.drug_kernel[0]-self.drug_kernel[1]-self.drug_kernel[2]+3)
        
        ### Other features ###
        self.features_NN = nn.Sequential(
            nn.Linear(self.f_dim, 64),
            nn.ReLU(),
            nn.BatchNorm1d(num_features=64),
            nn.Linear(64, 64),
            nn.ReLU(),
            nn.BatchNorm1d(num_features=64),
            nn.Linear(64, self.f_out_dim),
            nn.ReLU(),
        )
        
        self.features_CNNs = nn.Sequential(
            nn.Conv1d(in_channels=self.f_dim, out_channels=self.conv, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Conv1d(in_channels=self.conv, out_channels=self.conv * 2, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Conv1d(in_channels=self.conv * 2, out_channels=self.conv * 4, kernel_size=3, padding=1),
            nn.ReLU(),
        )
        
        ### Layers ###
        self.attention_layer = nn.Linear(self.conv*4,self.conv*4)
        self.drug_attention_layer = nn.Linear(self.conv * 4, self.conv * 4)
        self.feature_attention_layer1 = nn.Linear(self.conv * 4, self.conv * 4)
        self.feature_attention_layer2 = nn.Linear(self.f_out_dim, self.f_out_dim)
        
        self.dropout1 = nn.Dropout(0.1)
        self.dropout2 = nn.Dropout(0.1)
        self.dropout3 = nn.Dropout(0.1)
        self.relu = nn.ReLU()
        self.tanh = nn.Tanh()
        self.sigmoid = nn.Sigmoid()
        self.leaky_relu = nn.LeakyReLU()
        
        self.fc1 = nn.Linear(self.conv*8, 1024)
        self.fc2 = nn.Linear(1024, 1024)
        self.fc3 = nn.Linear(1024, 512)
        self.out1 = nn.Linear(512, 1)
        # self.out2 = nn.Linear(512, 14)

    def forward(self, drug, feature):
        # print(drug.shape, feature.shape) # torch.Size([32, 100]) torch.Size([32, 7])
        drugembed = self.drug_embed(drug) # torch.Size([32, 100, 64])
        drugembed = drugembed.permute(0, 2, 1) # torch.Size([32, 64, 100])
        
        drugConv = self.Drug_CNNs(drugembed) # torch.Size([32, 160, 85])
        drug_att = self.drug_attention_layer(drugConv.permute(0, 2, 1)) # [32, 85, 160]
        
        if self.f_layer == 'cnn':
            # feature = feature.view(32, 7, -1)
            feature = feature.view(feature.shape[0], 7, -1)
            featureNN = self.features_CNNs(feature) # [32, 160, 1]
            feature_att = self.feature_attention_layer1(featureNN.permute(0, 2, 1)) # [32, 1, 160]
        elif self.f_layer == 'gnn':
            pass
        
        elif self.f_layer == 'mlp':
            featureNN = self.features_NN(feature) # torch.Size([32, 85])
            feature_att = self.feature_attention_layer2(featureNN) # torch.Size([32, 85])
        
        ### repeat along other feature size
        d_att_layers = torch.unsqueeze(drug_att, 2).repeat(1, 1, featureNN.shape[-1], 1)  
        ### repeat along drug size
        f_att_layers = torch.unsqueeze(feature_att, 1).repeat(1, drugConv.shape[-1], 1, 1)

        
        Atten_matrix = self.attention_layer(self.relu(d_att_layers + f_att_layers))
        '''
        mean operation which returns the mean value of each row of Input 
        in the given dimension dim
        '''
        Compound_atte = torch.mean(Atten_matrix, 2) # [32, 85, 160]
        Feature_atte = torch.mean(Atten_matrix, 1) # [32, 1, 160]
        
        Compound_atte = self.sigmoid(Compound_atte.permute(0, 2, 1))
        Feature_atte = self.sigmoid(Feature_atte.permute(0, 2, 1))

        '''
        We then apply a global-max pooling operation over Da and Pa to obtain 
        the feature vectors, vdrug and vprotein, which are concatenated and 
        feed into the output block. 
        '''
        # print(drugConv.shape, Compound_atte.shape) # [32, 160, 85], [32, 160, 85]
        drugConv = drugConv * 0.5 + drugConv * Compound_atte # [32, 160, 85]
        # print(featureNN.shape, Feature_atte.shape) # [32, 160, 1], [32, 160, 1]
        featureNN = featureNN * 0.5 + featureNN * Feature_atte # [32, 160, 1]
        
        drugConv = self.Drug_max_pool(drugConv).squeeze(2)
        featureNN = featureNN.squeeze(2)
        
        ''' 
        Concat & Predict 
        '''
        pair = torch.cat([drugConv, featureNN], dim=1)
        pair = self.dropout1(pair)
        fully1 = self.leaky_relu(self.fc1(pair))
        fully1 = self.dropout2(fully1)
        fully2 = self.leaky_relu(self.fc2(fully1))
        fully2 = self.dropout3(fully2)
        fully3 = self.leaky_relu(self.fc3(fully2))
        pred = self.out1(fully3)
        # if feature.shape[0] == self.batch_size:
        #     pred = self.out1(fully3)
        # else:
        #     pred = self.out2(fully3)
        return pred


