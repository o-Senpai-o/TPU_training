from utils import AverageMeter

import torch

import torch_xla.core.xla_model as xm



def train_fn(model, para_loader, device, optimizer, scheduler,  criterion, batch_verbose = False):
    # first put the model in training mode
    model.train()

    # get the average meter, to keep track of average loss score
    train_loss_meter = AverageMeter()

    for batch_idx, (inputs, labels) in enumerate(para_loader):

        # put inputs and labels to device
        inputs = inputs.to(device)
        labels = labels.to(device)

        # zero grad the optimizer in the satrt of forward pass
        optimizer.zero_grad()

        # pass the inputs to model and get the outputs
        preds = model(inputs)
        loss = criterion(labels, preds)
        loss.backward()

        #! to step the optimizer use the following, 
        #! barrier is required on single-core training but can be dropped with multiple cores
    
        xm.optimizer_step(optimizer, barrier=True)                                                  # diff from normal training

        # store the loss in average meter
        train_loss_meter.update(loss.detach().item(), inputs.size(0))

        # print the batch number and average loss till now
        if batch_idx > 0 and batch_idx % batch_verbose == 0:                                        # diff from normal training
            xm.master_print('-- batch {} | cur_loss = {:.6f}, avg_loss = {:.6f}').format(               
                            batch_idx, loss.item(), train_loss_meter.avg
            )
        

        # clear memory
        del inputs, labels , preds, loss
        
        # early stop
        if batch_idx > batch_per_epoch:
            break


    # after one epoch, we ste the scheduler
    scheduler.step()

    # free the memory
    del para_loader, batch_idx
    gc.collect

    return train_loss_meter.avg




def eval(model, para_loader, device, criterion, batch_verbose=False):
    # put model in eval model
    model.eval()

    # get the averrage meter to keep track of average loss
    val_avg_meter = AverageMeter()

    for batch_idx, (inputs, labels) in enumerate(para_loader):

        inputs = inputs.to(device)
        labels = labels.to(device)

        # compute predictions
        with torch.no_grad():

            preds = model(inputs)
            loss = criterion(labels, preds)

        
        val_avg_meter.update(loss.detach().item(), inputs.size(0))

        # feedback
        if batch_idx > 0 and batch_idx % batch_verbose:
            xm.master_print("--batch {} | curr loss {:.6f} , avg loss {:.6f}".format(
                batch_idx, loss.item(), val_avg_meter.avg
            ))
        

        # empty the memory
        del inputs, labels, preds, loss
        gc.collect

    
    del para_loader, batch_idx
    gc.collect

    return val_avg_meter.avg

    

























    