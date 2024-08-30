# Author: Amit Parag
# Date : 11 May, 2022

# This script contains the main supervised training procedure for learning value functions, state and control trajectories



from time import perf_counter
import torch
from torch.optim import Adam
from torch.nn.functional import mse_loss as mse
from tqdm import trange
from torch.utils.data import TensorDataset, DataLoader









def weighted_mse_loss(input, target, alpha=1):
    try:
        assert target.shape == input.shape
    except AssertionError:
        target = target.reshape(input.shape)
    return (alpha * mse(input,target)).mean()








#### NOT USED
def compute_anchor_loss(input, target,horizon):
    """
    The first term of xs_predicted must match the x0s . Must penalize this heavily
    """
    x0s_predicted   =   torch.vstack([i[0] for i in input.view(-1,horizon+1,target.shape[1])])
    x0s_predicted   =   x0s_predicted.clone().detach().requires_grad_(True)
    loss            =   weighted_mse_loss(x0s_predicted,target)
    return loss




def torch_dataloader(datas,device='cpu'):

    datas = list(map(torch.Tensor, datas))
    for idx,data in enumerate(datas):
        datas[idx] = data.requires_grad_(True)
        datas[idx] = data.to(device)

    size        = datas[0].shape[0]
    datas       = TensorDataset(*datas)
    data_loader = DataLoader(datas,batch_size=3*size//5,shuffle=True)
    datas       = None
    return data_loader



def learn_policy(policy,training_params,datas,model):


    print("\n")

    epochs  =   training_params["epochs"]
    lr      =   training_params["lr"]
    device  =   torch.device(training_params['device'])

    if device == 'cuda':
        torch.cuda.empty_cache()
        model.to(device)
        assert next(model.parameters()).is_cuda


    data_loader              =   torch_dataloader(datas,device=device)
    dataset_size             =   data_loader.dataset.tensors[0].shape[0]



    if policy.startswith("s"):
        trainable_parameters     =   list(model.state_layers.parameters())
        value = False
        state  = True
        control = False


    if policy.startswith("c"):
        trainable_parameters     =   list(model.control_layers.parameters())
        value = False
        state  = False
        control = True

    
    optimizer                =   Adam(trainable_parameters, lr = lr)


    #### Total loss in training

    total_loss           =   []


    progress_bar         =   trange(epochs, desc = f" Training ... ")

    start_time           =   perf_counter()



    for epoch,_ in zip(range(epochs), progress_bar):
        model.train()
        ### Computation of epoch Loss
        epoch_loss = 0.0

        for idx, (data) in enumerate(data_loader):

            x0s, target =   data


            prediction      =   model(x0s,state=state,control=control,value=value)
            losses          =   mse(prediction,target)


            
            optimizer.zero_grad()
       
            losses.backward()
            optimizer.step()

            epoch_loss +=  x0s.shape[0] * losses.item() ## us
        
        
        total_loss.append(epoch_loss / dataset_size)


        if epoch % 2 == 0:

            e_loss =   "{:.3e}".format(epoch_loss)

            desc = f"Learning {dataset_size} {policy} !! Epoch:{epoch} : Loss:: {e_loss}"
            progress_bar.set_description(desc)
            progress_bar.refresh()


    end_time = perf_counter()
    info_dict = {}
    info_dict["Mse"] = total_loss
    info_dict["Train Time"] = end_time - start_time


    total_loss  =   None
    losses      =   None


    if device == 'cuda':
        x0s = None
        target = None
        data_loader = None
        torch.cuda.empty_cache()



    return info_dict, model



def learn_value(training_params, datas, model):

    print("\n")

    early_stop  =   training_params['early_stop']
    epochs      =   training_params["epochs"]
    lr          =   training_params["lr"]
    device      =   torch.device(training_params['device'])

    if device == 'cuda':
        torch.cuda.empty_cache()
        model.to(device)
        assert next(model.residual_layers.parameters()).is_cuda



    data_loader              =   torch_dataloader(datas,device=device)
    dataset_size             =   data_loader.dataset.tensors[0].shape[0]
    trainable_parameters     =   list(model.residual_layers.parameters())
    optimizer                =   Adam(trainable_parameters, lr = lr)


    #### Total loss in training
    total_loss_v            =   []
    total_loss_vx           =   []



    progress_bar            =   trange(epochs, desc = f" Training ... ")

    start_time              =   perf_counter()


    for epoch,_ in zip(range(epochs), progress_bar):
        model.train()
        ### Computation of epoch Loss
        loss_v,loss_vx = 0.0, 0.0

        for idx, (data) in enumerate(data_loader):

            x0s, v, vx  =   data


            v_,vx_      =   model(x0s,state=False,control=False,value=True)
            losses      =   list(map(mse, [v_,vx_,], [v,vx]))


            
            optimizer.zero_grad()
            

            loss = sum(losses)/len(losses)
            loss.backward()
            optimizer.step()

            loss_v  +=  x0s.shape[0] * losses[0].item() ## v 
            loss_vx +=  x0s.shape[0] * losses[1].item() ## vx


            

        total_loss_v.append(loss_v   / dataset_size)
        total_loss_vx.append(loss_vx / dataset_size)


        if epoch % 2 == 0:
            lv  =   "{:.3e}".format(loss_v)
            lvx =   "{:.3e}".format(loss_vx)


            desc = f"Learning {dataset_size} values !! Epoch:{epoch} : Losses:: v:{lv}, vx:{lvx}"
            progress_bar.set_description(desc)
            progress_bar.refresh()

        if loss_v <= early_stop:
            lv  =   "{:.3e}".format(loss_v)
            lvx =   "{:.3e}".format(loss_vx)
            desc = f"Early Stopping !! Epoch:{epoch} : Losses:: v:{lv}, vx:{lvx}"
            progress_bar.set_description(desc)
            progress_bar.refresh()
            break


    end_time = perf_counter()
    info_dict = {}
    info_dict["Mse v"] = total_loss_v
    info_dict["Mse vx"] = total_loss_vx
    info_dict["Train Time"] = end_time - start_time


    total_loss_v            =   None
    total_loss_vx           =   None


    loss_v,loss_vx = None, None
    progress_bar = None


    if device == 'cuda':
        x0s = None
        v   = None
        vx  = None
        data_loader = None
        torch.cuda.empty_cache()


    return info_dict, model

        









