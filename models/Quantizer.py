import torch
import torch.nn as nn
from vector_quantize_pytorch import VectorQuantize ,ResidualVQ ,GroupedResidualVQ


class Quantizer(nn.Module):

    '''

    VQ has been successfully used by Deepmind and OpenAI for high quality generation of images (VQ-VAE-2) and music (Jukebox).    
    
    In this project, we utilized a pre-existing Vector Quantization implementation from a GitHub repository :
    https://github.com/lucidrains/vector-quantize-pytorch.git
    
    
    We integrated it into our model to improve the accuracy and quality of future image reconstruction.        
    
    
    '''

    def __init__(self,embedding_dim,Quantizer_name='VectorQuantize',commitment_weight=1.,codebook_size=128,decay=0.8,num_quantizers=4):
        super(Quantizer, self).__init__()

        if(Quantizer_name=='ResidualVQ'):
            
            self.vq = ResidualVQ(
                     dim = embedding_dim,
                     codebook_size = codebook_size,         # codebook size
                     decay = decay,                         # the exponential moving average decay, lower means the dictionary will change faster
                     num_quantizers = num_quantizers,       # specify number of quantizers
                     commitment_weight = commitment_weight, # the weight on the commitment loss
                     stochastic_sample_codes = True,
                     sample_codebook_temp = 0.1,            # temperature for stochastically sampling codes, 0 would be equivalent to non-stochastic
                     shared_codebook = True,                # whether to share the codebooks for all quantizers or not
                     accept_image_fmap = True              
                       )
            

        elif(Quantizer_name=='GroupedResidualVQ'):

            self.vq = GroupedResidualVQ(
                     dim = embedding_dim,
                     codebook_size = codebook_size,         # codebook size
                     decay = decay,                         # the exponential moving average decay, lower means the dictionary will change faster
                     num_quantizers = num_quantizers,       # specify number of quantizers
                     commitment_weight = commitment_weight, # the weight on the commitment loss
                     stochastic_sample_codes = True,
                     sample_codebook_temp = 0.1,            # temperature for stochastically sampling codes, 0 would be equivalent to non-stochastic
                     shared_codebook = True,                # whether to share the codebooks for all quantizers or not
                     accept_image_fmap = True               
                       )
            
        else :
            self.vq = VectorQuantize(
                     dim = embedding_dim,
                     codebook_size = codebook_size,         # codebook size
                     decay = decay ,                        # the exponential moving average decay, lower means the dictionary will change faster
                     commitment_weight = commitment_weight, # the weight on the commitment loss
                     accept_image_fmap = True
                       )


    def forward(self,inputs):

        quantized,_, commit_loss = self.vq(inputs)
        return quantized ,torch.mean(commit_loss)




class VectorQuantizerNormal(nn.Module):
    def __init__(self, num_embeddings, embedding_dim, commitment_cost):
        super(VectorQuantizerNormal, self).__init__()
        
        self._embedding_dim = embedding_dim
        self._num_embeddings = num_embeddings
        
        self._embedding = nn.Embedding(self._num_embeddings, self._embedding_dim)
        self._embedding.weight.data.uniform_(-1/self._num_embeddings, 1/self._num_embeddings)
        self._commitment_cost = commitment_cost

    def forward(self, inputs):
        # convert inputs from BCHW -> BHWC
        inputs = inputs.permute(0, 2, 3, 1).contiguous()
        input_shape = inputs.shape
        
        # Flatten input
        flat_input = inputs.view(-1, self._embedding_dim)
        
        # Calculate distances
        distances = (torch.sum(flat_input**2, dim=1, keepdim=True) 
                    + torch.sum(self._embedding.weight**2, dim=1)
                    - 2 * torch.matmul(flat_input, self._embedding.weight.t()))
            
        # Encoding
        encoding_indices = torch.argmin(distances, dim=1).unsqueeze(1)
        encodings = torch.zeros(encoding_indices.shape[0], self._num_embeddings, device=inputs.device)
        encodings.scatter_(1, encoding_indices, 1)
        
        # Quantize and unflatten
        quantized = torch.matmul(encodings, self._embedding.weight).view(input_shape)
        
        # Loss
        e_latent_loss = F.mse_loss(quantized.detach(), inputs)
        q_latent_loss = F.mse_loss(quantized, inputs.detach())
        loss = q_latent_loss + self._commitment_cost * e_latent_loss
        
        quantized = inputs + (quantized - inputs).detach()
        avg_probs = torch.mean(encodings, dim=0)
        perplexity = torch.exp(-torch.sum(avg_probs * torch.log(avg_probs + 1e-10)))
        
        # convert quantized from BHWC -> BCHW
        return loss, quantized.permute(0, 3, 1, 2).contiguous(), perplexity, encodings
    


class VectorQuantizerEMA(nn.Module):
    def __init__(self, num_embeddings, embedding_dim, commitment_cost, decay, epsilon=1e-5):
        super(VectorQuantizerEMA, self).__init__()
        
        self._embedding_dim = embedding_dim
        self._num_embeddings = num_embeddings
        
        self._embedding = nn.Embedding(self._num_embeddings, self._embedding_dim)
        self._embedding.weight.data.normal_()
        self._commitment_cost = commitment_cost
        
        self.register_buffer('_ema_cluster_size', torch.zeros(num_embeddings))
        self._ema_w = nn.Parameter(torch.Tensor(num_embeddings, self._embedding_dim))
        self._ema_w.data.normal_()
        
        self._decay = decay
        self._epsilon = epsilon

    def forward(self, inputs):
        # convert inputs from BCHW -> BHWC
        inputs = inputs.permute(0, 2, 3, 1).contiguous()
        input_shape = inputs.shape
        
        # Flatten input
        flat_input = inputs.view(-1, self._embedding_dim)
        
        # Calculate distances
        distances = (torch.sum(flat_input**2, dim=1, keepdim=True) 
                    + torch.sum(self._embedding.weight**2, dim=1)
                    - 2 * torch.matmul(flat_input, self._embedding.weight.t()))
            
        # Encoding
        encoding_indices = torch.argmin(distances, dim=1).unsqueeze(1)
        encodings = torch.zeros(encoding_indices.shape[0], self._num_embeddings, device=inputs.device)
        encodings.scatter_(1, encoding_indices, 1)
        
        # Quantize and unflatten
        quantized = torch.matmul(encodings, self._embedding.weight).view(input_shape)
        
        # Use EMA to update the embedding vectors
        if self.training:
            self._ema_cluster_size = self._ema_cluster_size * self._decay + \
                                     (1 - self._decay) * torch.sum(encodings, 0)
            
            # Laplace smoothing of the cluster size
            n = torch.sum(self._ema_cluster_size.data)
            self._ema_cluster_size = (
                (self._ema_cluster_size + self._epsilon)
                / (n + self._num_embeddings * self._epsilon) * n)
            
            dw = torch.matmul(encodings.t(), flat_input)
            self._ema_w = nn.Parameter(self._ema_w * self._decay + (1 - self._decay) * dw)
            
            self._embedding.weight = nn.Parameter(self._ema_w / self._ema_cluster_size.unsqueeze(1))
        
        # Loss
        e_latent_loss = F.mse_loss(quantized.detach(), inputs)
        loss = self._commitment_cost * e_latent_loss
        
        # Straight Through Estimator
        quantized = inputs + (quantized - inputs).detach()
        avg_probs = torch.mean(encodings, dim=0)
        perplexity = torch.exp(-torch.sum(avg_probs * torch.log(avg_probs + 1e-10)))
        
        # convert quantized from BHWC -> BCHW
        return loss, quantized.permute(0, 3, 1, 2).contiguous(), perplexity, encodings