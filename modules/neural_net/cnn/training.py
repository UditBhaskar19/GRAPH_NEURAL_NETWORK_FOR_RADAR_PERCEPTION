# --------------------------------------------------------------------------------------------------------------
# Author : Udit Bhaskar
# Description : model optimization function
# --------------------------------------------------------------------------------------------------------------
import numpy as np
import time, os, torch

# --------------------------------------------------------------------------------------------------------------
def create_weight_dir(model_weights_main_dir):
    curr_weight_dir = str(round(time.time() * 1000))
    if not os.path.exists(model_weights_main_dir): os.mkdir(model_weights_main_dir)
    weight_dir = os.path.join(model_weights_main_dir, curr_weight_dir)
    if not os.path.exists(weight_dir): os.mkdir(weight_dir)
    return weight_dir

def save_model_weights(detector, weight_dir, weights_name):
    weight_path = os.path.join(weight_dir, weights_name)
    torch.save(detector.state_dict(), weight_path)

# --------------------------------------------------------------------------------------------------------------
def get_all_trainable_parameters(detector_train):
    weights = [p for p in detector_train.parameters() if p.requires_grad]
    return weights

def debug_weights(weights):
    weights_flat = [torch.flatten(weight) for weight in weights]
    weights_1d = torch.cat(weights_flat)
    print(f"max and min weight: {weights_1d.max()}, {weights_1d.min()}")
    assert not torch.isnan(weights_1d).any()
    assert not torch.isinf(weights_1d).any()

def debug_gradients(weights):
    grad_flat = [torch.flatten(weight.grad) for weight in weights if weight.grad != None]
    if len(grad_flat) > 0:
        grad_1d = torch.cat(grad_flat)
        print(f"max and min gradient: {grad_1d.max()}, {grad_1d.min()}")
        assert not torch.isnan(grad_1d).any()
        assert not torch.isinf(grad_1d).any()

def skip_batch(loss):
    skip = False
    if torch.isnan(loss):
        skip = True
        print('WARNING!...skipping current batch since loss is nan due to corrupted sample')
    return skip

# --------------------------------------------------------------------------------------------------------------
def train_model_accumulate_grad(
    detector,
    optimizer,
    lr_scheduler,
    dataloader_train,
    dataloader_val,
    tb_writer,
    max_iters,
    grad_accumulation_steps,
    log_period,
    val_period,
    iter_start_offset,
    model_weights_dir,
    weights_name):

    loss_tracker = LossTracker()        # track losses for tensorboard visualization
    weight_dir = create_weight_dir(model_weights_dir)    # create directory to save model weights 

    for iter_train_outer in range(iter_start_offset, max_iters):
        detector.train()
        skip = False
        for _ in range(grad_accumulation_steps):
            # ==============================================================================================
            image_grid, meas_vr_grid, meas_rcs_grid, labels = next(dataloader_train)    
            losses = detector(image_grid, labels['class'], labels['offsets'], meas_vr_grid, meas_rcs_grid)
            total_loss = losses['loss_cls'] + losses['loss_reg']
            total_loss = total_loss / grad_accumulation_steps

            skip = skip_batch(total_loss)
            if skip == True: break
            total_loss.backward()

        if skip == False:
            optimizer.step()
            lr_scheduler.step()  
        optimizer.zero_grad()

        # ==============================================================================================      
        # validate model, print loss on console, save the model weights
        total_loss = total_loss * grad_accumulation_steps
        if total_loss > 0:
            loss_tracker.append_training_loss_for_tb(total_loss, losses)
            loss_tracker.loss_history.append(total_loss.item())
    
        if iter_train_outer % log_period == 0:
            loss_str = f"[Iter {iter_train_outer}][loss: {total_loss.item():.5f}]"
            for key, value in losses.items():
                loss_str += f"[{key}: {value.item():.5f}]"
            print(loss_str)
        
        condition = (iter_train_outer % val_period == 0) or (iter_train_outer == max_iters - 1)
        if condition:   # write to tb,  and run validation 
            print('-'*100)
            print('saving model')
            save_model_weights(detector, weight_dir, weights_name)

            # bdd dataset validation
            print('performing validation Radar Scenes Dataset')
            print('-'*100)

            detector.eval()
            with torch.no_grad():
                for iter_val, (image_grid, meas_vr_grid, meas_rcs_grid, labels) in enumerate(dataloader_val):      
                    losses = detector(image_grid, labels['class'], labels['offsets'], meas_vr_grid, meas_rcs_grid)
                    total_loss = sum(losses.values())
                    if total_loss > 0:
                        loss_tracker.append_validation_loss_for_tb(total_loss, losses)     

                    if iter_val % log_period == 0:   # Print losses periodically on the console
                        loss_str = f"[Iter {iter_val}][loss: {total_loss.item():.5f}]"
                        for key, value in losses.items():
                            loss_str += f"[{key}: {value.item():.5f}]"
                        print(loss_str)

            # ==============================================================================================
            # write the loss to tensor board
            train_loss_tb, cls_train_loss_tb, reg_train_loss_tb= loss_tracker.compute_avg_training_loss()

            val_loss_tb, cls_val_loss_tb, reg_val_loss_tb = loss_tracker.compute_avg_val_loss()

            print('train_losses_tb : ', train_loss_tb)          
            print('val_losses_tb   : ', val_loss_tb)      
            print("end of validation : Resuming Training")
            print("-"*100)

            tb_writer.add_scalars('Total_Loss', 
                                {'train':train_loss_tb, 
                                'val':val_loss_tb}, iter_train_outer)

            tb_writer.add_scalars('Classification_Loss', 
                                {'train':cls_train_loss_tb, 
                                'val':cls_val_loss_tb}, iter_train_outer)

            tb_writer.add_scalars('Offset_Loss', 
                                {'train':reg_train_loss_tb, 
                                'val':reg_val_loss_tb}, iter_train_outer)

            # reset train_losses_tb
            loss_tracker.reset_training_loss_for_tb()
            loss_tracker.reset_validation_loss_for_tb()


# --------------------------------------------------------------------------------------------------------------
class LossTracker:
    def __init__(self):
        # training losses
        self.loss_history = []      # Keep track of training loss for plotting.
        self.train_losses_tb = []   # train_losses for tensor board visualization
        self.cls_train_losses_tb = []  # classification train_losses for tensor board visualization
        self.reg_train_losses_tb = []  # box regression train_losses for tensor board visualization

        # val losses
        self.val_losses_tb = []
        self.cls_val_losses_tb = []  
        self.reg_val_losses_tb = []   

    def append_training_loss_for_tb(self, total_loss, losses):
        self.train_losses_tb.append(total_loss.item()) 
        self.cls_train_losses_tb.append(losses['loss_cls'].item())
        self.reg_train_losses_tb.append(losses['loss_reg'].item())

    def append_validation_loss_for_tb(self, total_loss, losses):
        self.val_losses_tb.append(total_loss.item()) 
        self.cls_val_losses_tb.append(losses['loss_cls'].item())
        self.reg_val_losses_tb.append(losses['loss_reg'].item())

    def reset_training_loss_for_tb(self):
        self.train_losses_tb = []   
        self.cls_train_losses_tb = []  
        self.reg_train_losses_tb = []  

    def reset_validation_loss_for_tb(self):
        self.val_losses_tb = []
        self.cls_val_losses_tb = []  
        self.reg_val_losses_tb = []    

    def compute_avg_training_loss(self):
        total_train_loss_tb = np.mean(np.array(self.train_losses_tb))   
        cls_train_loss_tb = np.mean(np.array(self.cls_train_losses_tb))
        reg_train_loss_tb = np.mean(np.array(self.reg_train_losses_tb))
        return total_train_loss_tb, cls_train_loss_tb, reg_train_loss_tb
    
    def compute_avg_val_loss(self):
        val_loss_tb = np.mean(np.array(self.val_losses_tb))   
        cls_val_loss_tb = np.mean(np.array(self.cls_val_losses_tb))
        reg_val_loss_tb = np.mean(np.array(self.reg_val_losses_tb))
        return val_loss_tb, cls_val_loss_tb, reg_val_loss_tb