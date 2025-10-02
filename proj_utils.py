import torch
import os

def read_proj(model,filename):
    """
    Reads projections from memory
    Builds tensors of shape (h,d,r) for K,Q,V,Wo
    """
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    d = model.config.head_dim
    res = []
    for l in range(model.config.num_hidden_layers):
        projk = torch.load(os.environ["SCRATCH"]+"/"+filename+"/"+str(l)+"/projk.pt")
        projq = torch.load(os.environ["SCRATCH"]+"/"+filename+"/"+str(l)+"/projq.pt")

        #print("diff to id KQT:",torch.norm(torch.eye(d).to(device)-projk[0]@(projq[0].T)))

        projv = torch.load(os.environ["SCRATCH"]+"/"+filename+"/"+str(l)+"/projv.pt")
        projwo = torch.load(os.environ["SCRATCH"]+"/"+filename+"/"+str(l)+"/projw.pt")

        #print("diff to id VWO:",torch.norm(torch.eye(d).to(device)-projv[0]@(projwo[0].T)))

        res.append([projk,projq,projv,projwo])
        #print(projk.shape,projq.shape,projv.shape,projwo.shape)
    
    return res

def read_spectrums(model,filename):
    """
    Reads spectrums from memory
    Returns a list of size L containing a list of size 2 (for K and V)
    Tensors in this list of size 2 have size (h,d). the ith row is the spectrum for the ith head
    """
    res = []
    for l in range(model.config.num_hidden_layers):
        speck = torch.load(os.path.join(os.environ["SCRATCH"],filename,str(l),"speck.pt"))
        specv = torch.load(os.path.join(os.environ["SCRATCH"],filename,str(l),"specv.pt"))

        res.append([speck.detach().to("cpu"),specv.detach().to("cpu")])
    return res

def computing_rank(model,spec,epsilons,square=True):
    """
    Computes ranks based on spectrums and an epsilon
    The epsilon represents a relative frobenius norm error
    More precisely, the rank return is such that
    \sum_{i=r+1}^{d} \sigma_i^2 \le \epsilon^2 (\sum_{i=1}^{d} \sigma_i^2)

    The square option can be set to False if the spectrums are from an SVD of KQ^T and hence shouldn't be squared,
    to have a singular value decay comparable to the SVD of K
    Return a list of pairs of ranks, representing for each layer the rank for K and for V
    """
    #epsilons is a list of couples of relative errors on the frobenius norm
    ranks = []
    for l in range(model.config.num_hidden_layers):
        couple = []
        for c in range(2):
            if square:
                squared = torch.square(spec[l][c])
            else:
                squared = spec[l][c]
            squared = squared.flip(1)
            cumsum = torch.cumsum(squared,dim=1)
            sum = torch.sum(squared,dim=1)
            data = cumsum/sum[:,None]
            data = data.mean(dim=0)

            truncpost = [
                idx for idx, element in enumerate(data) if element <= epsilons[l][c]**2
            ]
            rank = len(data) - len(truncpost)
            couple.append(rank)
        ranks.append(couple)
    return ranks