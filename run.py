
#? The master modeling function also includes several TPU-based modifications.

#* First, we need to create a distributed data sampler that reads our Dataset object and distributes batches over TPU cores. 
      # This is done with torch.utils.data.distributed.DistributedSampler(),
      # which allows data loaders from different cores to only take a portion of the whole dataset.
      # Setting num_replicas to xm.xrt_world_size() checks the number of available TPU cores.
      # After defining the sampler, we can set up a data loader that uses the sampler.


#* Second, the model is sent to TPU with the following code:       

      #?  device = xm.xla_device()
      #?  xm = model.to(device)

#* Third, we need to update learning rate since the modeling is done simultaneously on batches on different cores:

      #?  scaled_eta = eta * xm.xrt_world_size().


#* Finally, we continue keeping track of the memory and clearing it whenever possible,
#* and use xm.master_print() for displaying intermediate results
#* Set up the function to return lists with the training and validation loss values



import torch

import torch_xla.core.xla_model as xm



def run():

    #? Data Prep
    # first we need to get the dataset, both train and test dataset
    # get the train adn test data sampler

    train_sampler = torch.utils.data.Distributed.DistributedSampler(train_dataset,
                                                                    num_replicas = xm.xrt_world_size(),
                                                                    rank = xm.get_ordinal(),
                                                                    shuffle = True,
                                                                    )

    valid_sampler = torch.utils.data.Distributed.DistributedSampler(valid_dataset,
                                                                    num_replicas = xm.xrt_world_size(),
                                                                    rank = xm.get_ordinal(),
                                                                    shuffle=False)
    
    # dataloader
    train_loader = torch.utils.data.DataLoader(train_dataset,
                                               batch_size = 16,
                                               sampler = train_sampler,
                                               shuffle = False,
                                               num_worker = 0,
                                               pin_memory = True)

    Valid_loader = torch.utils.data.DataLoader(valid_dataset,
                                               batch_size = 8,
                                               sampler = valid_sampler,
                                               num_workers = 0,
                                               pin_memory = True)
    

    #? MODEL PREP

    # push the model to xla device
    device = xm.xla_device()
    model = model.to(device)

    # we need learning rate
    # scale LR
    scaled_eta = eta * xm.xrt_world_size()


    # we need optimizer\
    # optimizer and loss
    criterion = nn.BCEWithLogitsLoss()
    optimizer = optim.Adam(model.parameters(), lr = scaled_eta)
    scheduler = lr_scheduler.StepLR(optimizer, step_size = step, gamma = gamma)


     # placeholders
    trn_losses = []
    val_losses = []
    best_val_loss = 1
        
    # modeling loop
    gc.collect()
    for epoch in range(num_epochs):
        
        # display info
        xm.master_print('-'*55)
        xm.master_print('EPOCH {}/{}'.format(epoch + 1, num_epochs))
        xm.master_print('-'*55)            
        xm.master_print('- initialization | TPU cores = {}, lr = {:.6f}'.format(
            xm.xrt_world_size(), scheduler.get_lr()[len(scheduler.get_lr()) - 1] / xm.xrt_world_size()))
        epoch_start = time.time()
        gc.collect()
        
        # update train_loader shuffling
        train_loader.sampler.set_epoch(epoch)
        
        # training pass
        train_start = time.time()
        xm.master_print('- training...')

        #! get a parallelLoader which will load data parallely
        para_loader = pl.ParallelLoader(train_loader, [device])
        trn_loss = train_fn(epoch       = epoch + 1, 
                            para_loader = para_loader.per_device_loader(device), 
                            criterion   = criterion,
                            optimizer   = optimizer, 
                            scheduler   = scheduler,
                            device      = device)
        del para_loader
        gc.collect()
        
        # validation pass
        valid_start = time.time()
        xm.master_print('- validation...')
        para_loader = pl.ParallelLoader(valid_loader, [device])
        val_loss = valid_fn(epoch       = epoch + 1, 
                            para_loader = para_loader.per_device_loader(device), 
                            criterion   = criterion, 
                            device      = device)
        del para_loader
        gc.collect()

        # save weights
        if val_loss < best_val_loss:
            xm.save(model.state_dict(), 'weights_{}.pt'.format(model_name))
            best_val_loss = val_loss
                        
        # display info
        xm.master_print('- elapsed time | train = {:.2f} min, valid = {:.2f} min'.format(
            (valid_start - train_start) / 60, (time.time() - valid_start) / 60))
        xm.master_print('- average loss | train = {:.6f}, valid = {:.6f}'.format(
            trn_loss, val_loss))
        xm.master_print('-'*55)
        xm.master_print('')
        
        # save losses
        trn_losses.append(trn_loss)
        val_losses.append(val_loss)
        del trn_loss, val_loss
        gc.collect()
        
    # print results
    xm.master_print('Best results: loss = {:.6f} (epoch {})'.format(np.min(val_losses), np.argmin(val_losses) + 1))
    
    return trn_losses, val_losses





# wrapper function
def _mp_fn(rank, flags):
    torch.set_default_tensor_type('torch.FloatTensor')
    trn_losses, val_losses = _run(model)
    np.save('trn_losses.npy', np.array(trn_losses))
    np.save('val_losses.npy', np.array(val_losses))
    
# modeling
gc.collect()
FLAGS = {}
xmp.spawn(_mp_fn, args = (FLAGS,), nprocs = num_tpu_workers, start_method = 'fork')