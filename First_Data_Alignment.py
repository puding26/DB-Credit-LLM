import pandas as pd
import numpy as np

import torch
import torch.nn as nn
import torch.nn.functional as F

from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import StandardScaler

#Take loan data as example, download raw data from the provided linkages in the paper, other datasets are similar.

datanew = pd.read_csv('Loan_Data.csv')
encoder = LabelEncoder()
datanew['AES'] = encoder.fit_transform(datanew['AES'])
datanew['RES'] = encoder.fit_transform(datanew['RES'])
print(datanew['AES'])
print(datanew['RES'])
datanew['AES'] = pd.to_numeric(datanew['AES'])
datanew['RES'] = pd.to_numeric(datanew['RES'])
X=datanew.drop(columns=['BAD'])
y=datanew['BAD']

#devide demographic and behavioral features

featuredemotrain = X[['YOB','NKID','DEP','PHON','SINC','AES','DAINC','RES']]
featurebehtrain = X[['DHVAL','DMORT','DOUTM','DOUTL','DOUTHP','DOUTCC']]
scaler = StandardScaler()
X1_scaled = scaler.fit_transform(featuredemotrain)
X2_scaled = scaler.fit_transform(featurebehtrain)
X1_scaled = X1_scaled.astype(np.float32)
X2_scaled = X2_scaled.astype(np.float32)
featuredemotensor = torch.tensor(X1_scaled)
featurebehtensor = torch.tensor(X2_scaled)


y_tensor = torch.tensor(y.values.astype(np.float32))


class CustomLinear(nn.Module):
    def __init__(self):
        super(CustomLinear, self).__init__()
        #self.factorized = factorized

    def forward(self, input, weights, biases):

        return torch.mul(input, weights) + biases


def elu_feature_map(dim):
    """Example of a feature map function."""
    return lambda x: F.elu(x) + 1

class LinearAttention(nn.Module):
    def __init__(self,query_dimensions, feature_map=None):
        super().__init__()
        self.feature_map = (
            feature_map(query_dimensions) if feature_map else
            elu_feature_map(query_dimensions)
        )
    def forward(self, Q, K, V):
        # Linear attention mechanism
        Q = self.feature_map(Q)  # (B, L, D)
        K = self.feature_map(K)  # (B, L, D)

        attention_scores = torch.matmul(Q, K.transpose(-2, -1))
        attention_scores = F.softmax(attention_scores, dim=-1)
        output = torch.matmul(attention_scores, V)
        
        return output
class FFTLinearAttention(nn.Module):
    def __init__(self, query_dimensions, feature_map=None, eps=1e-6):
        super(FFTLinearAttention, self).__init__()
        self.feature_map = (
            feature_map(query_dimensions) if feature_map else
            elu_feature_map(query_dimensions)
        )
        self.eps = eps

    def forward(self, queries, keys, values):
        # Apply the feature map to the queries and keys
        Q = self.feature_map(queries)
        K = self.feature_map(keys)

        # Perform FFT on the queries and keys
        Q_fft = torch.fft.fft(Q, dim=-1)  # FFT on the last dimension
        K_fft = torch.fft.fft(K, dim=-1)
        #print(Q_fft.shape, K_fft.shape)
        # Compute the KV matrix in Fourier domain
        KV = torch.einsum("nd,nd->nd", K_fft, values)  # K needs to match values

        # Compute the normalizer
        Z = 1 / (torch.einsum("nd,nd->n", Q_fft, K_fft) + self.eps)

        # Finally compute and return the new values
        V = torch.einsum("nd,nd,n->nd", Q_fft, KV, Z)

        return V.contiguous()


class LinearFeatureAttention(nn.Module):
    def __init__(self, d_model, feature_dim=None, eps=1e-6):
        super().__init__()
        self.d_model = d_model
        self.feature_dim = feature_dim or d_model // 2
        self.eps = eps

        #embedding
        self.Q_feature = nn.Sequential(
            nn.Linear(d_model, self.feature_dim),
            nn.ELU(inplace=True)
        )
        self.K_feature = nn.Sequential(
            nn.Linear(d_model, self.feature_dim),
            nn.ELU(inplace=True)
        )

        
        self.V_proj = nn.Linear(d_model, d_model)

    def forward(self, Q, K, V):
        """
        input shape: (batch, seq_len, d_model)
        output shape: (batch, seq_len, d_model)
         O(N) complexity
        """
        print(Q.shape)
        batch_size, seq_len = Q.shape

        # feature embedding
        Q_feat = self.Q_feature(Q)  # (B, L)
        K_feat = self.K_feature(K)  # (B, L)
        V_proj = self.V_proj(V)  # (B, D)

        print('Q_feat shape', Q_feat.shape)
        print('K_feat shape', K_feat.shape)
        print('V_proj shape', V_proj.shape)


        KV = torch.einsum('bi,bj->bij', K_feat, V_proj)  # (B, F, D)

        #attention weights
        QKV = torch.einsum('bi,bij->bj', Q_feat, KV)  # (B, L, D)


        K_sum = K_feat.sum(dim=1)

        print(K_sum.shape)
        Z = 1 / (torch.einsum('bi,b->b', Q_feat, K_sum) + self.eps) # (B, L)


        V_out = QKV * Z.unsqueeze(-1)

        return V_out



class BeIntraPatchAttention(nn.Module):
    def __init__(self, n_samples, query_dimensions):
        super(BeIntraPatchAttention, self).__init__()

        
        self.custom_linear = CustomLinear()
        self.n_samples = n_samples
        self.query_dimensions = query_dimensions
  

        self.weights_distinct = nn.ParameterList([
            nn.Parameter(torch.ones(self.n_samples, self.query_dimensions)) for _ in range(2)  # For key and value
        ])
        
        self.biases_distinct = nn.ParameterList([
            nn.Parameter(torch.zeros(self.n_samples, self.query_dimensions)) for _ in range(2)  # For key and value
        ])
        

        self.intra_attention = FFTLinearAttention(query_dimensions)
        # Fully connected layers for demographic and behavioral scores
         # Output size 1

    def forward(self, query, key, value):
        #batch_size = query.shape[0]
        print('key is', key)
        print('query is', query)
        print('value is', value)
        key = self.custom_linear(key, self.weights_distinct[0], self.biases_distinct[0])
        value = self.custom_linear(value, self.weights_distinct[1], self.biases_distinct[1])

       

        # Optionally, use FFT attention
        intra_output = self.intra_attention(query, key, value)

        
        return intra_output

class DeIntraPatchAttention(nn.Module):
    def __init__(self, n_samples, query_dimensions):
        super(DeIntraPatchAttention, self).__init__()

        # self.head_size = 4
        self.custom_linear = CustomLinear()
        self.n_samples = n_samples
        self.query_dimensions = query_dimensions
       

        self.weights_distinct = nn.ParameterList([
            nn.Parameter(torch.ones(self.n_samples, self.query_dimensions)) for _ in range(2)  # For key and value
        ])
        
        self.biases_distinct = nn.ParameterList([
            nn.Parameter(torch.zeros(self.n_samples, self.query_dimensions)) for _ in range(2)  # For key and value
        ])
        

        self.intra_attention = LinearFeatureAttention(query_dimensions)



    def forward(self, query, key, value):
        #batch_size = query.shape[0]
        print('key is', key)
        print('query is', query)
        print('value is', value)
        key = self.custom_linear(key, self.weights_distinct[0], self.biases_distinct[0])
        value = self.custom_linear(value, self.weights_distinct[1], self.biases_distinct[1])

       

        # Optionally, use FFT attention
        intra_output = self.intra_attention(query, key, value)

        # Combine or select between linear and FFT outputs (e.g., averaging)
        #output = (linear_output + fft_output) / 2  # Simple averaging for demonstration
        return intra_output



class BeInterPatchAttention(nn.Module):
    def __init__(self, feature_behavioral, feature_demographic):
        super(BeInterPatchAttention, self).__init__()
        self.linear_attention = LinearAttention(feature_behavioral)
        #self.fft_attention = FFTAttention(d_model)
        self.to_out = nn.Linear(feature_behavioral, feature_demographic)


    def forward(self, Q, K, V):
        # Use linear attention
        linear_output = self.linear_attention(Q, K, V)

        # Project to output dimension
        output = self.to_out(linear_output)
        return output

class DeInterPatchAttention(nn.Module):
    def __init__(self, feature_demographic, feature_behavioral):
        super(DeInterPatchAttention, self).__init__()
        self.linear_attention = LinearAttention(feature_demographic)
        #self.fft_attention = FFTAttention(d_model)
        self.to_out = nn.Linear(feature_demographic, feature_behavioral)


    def forward(self, Q, K, V):
        # Use linear attention
        linear_output = self.linear_attention(Q, K, V)

        # Project to output dimension
        output = self.to_out(linear_output)
        return output

class AttentionModel(nn.Module):
    def __init__(self, feature_demographic, feature_behavioral, out_dim, n_samples):
        super(AttentionModel, self).__init__()
        self.n_samples = n_samples
        self.intra_attention_demographic = DeIntraPatchAttention(n_samples, feature_demographic)
        self.intra_attention_behavioral = BeIntraPatchAttention(n_samples, feature_behavioral)
        self.Beinter_attention = BeInterPatchAttention(feature_behavioral, feature_demographic)
        self.Deinter_attention = DeInterPatchAttention(feature_demographic, feature_behavioral)
        self.fc_demographic = nn.Linear(feature_demographic, out_dim)
        self.fc_behavioral = nn.Linear(feature_behavioral, out_dim)

    def forward(self, demographics, behaviors):
        # Intra-attention on demographics
        demographic_output = self.intra_attention_demographic(demographics, demographics, demographics)

        # Intra-attention on behaviors
        behavioral_output = self.intra_attention_behavioral(behaviors, behaviors, behaviors)
        demographic_output = demographic_output
        behavioral_output =  behavioral_output.real
        # Inter-attention between demographic and behavioral outputs
        final_score1 = self.Beinter_attention(demographic_output, demographic_output, behavioral_output)
        demographic_score = torch.abs(F.gelu(self.fc_demographic(final_score1)))
        #final_score1.view(final_score1.size(0), 1)
        final_score2 = self.Deinter_attention(behavioral_output,behavioral_output, demographic_output)
        behavioral_score = torch.abs(F.gelu(self.fc_behavioral(final_score2)))
        #final_score2.view(final_score2.size(0), 1)
        return demographic_score, behavioral_score # Reshape to (n_samples, 1)



# Example usage
n_samples =  featuredemotensor.shape[0]
feature_demographic = 8
feature_behavioral = 6# Dimension of model
out_dim = 1  # Output dimension


# Dummy data
demographics1 = featuredemotensor
behaviors1 = featurebehtensor


model = AttentionModel(feature_demographic, feature_behavioral, out_dim, n_samples)
output = model(demographics1, behaviors1)
score1 = output[0]
score2 = output[1]

column_name1 = ['demo_score']
column_name2 = ['beh_score']
score1 = score1.detach().numpy()
score2 = score2.detach().numpy()

df1 = pd.DataFrame(score1, columns = column_name1)
df2 = pd.DataFrame(score2, columns = column_name2)

df = pd.concat([df1,df2],axis=1)
df.to_csv('scoreloandatalatest.csv', index=None)
