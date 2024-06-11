# --------------------------------------------------------------------------------------------------------------
# Author : Udit Bhaskar
# Description : model optimization function
# --------------------------------------------------------------------------------------------------------------
import numpy as np
import time, os, torch

# --------------------------------------------------------------------------------------------------------------
def create_weight_dir(model_weights_main_dir):
    curr_weight_dir = str(round(time.time() * 1000))
    os.makedirs(model_weights_main_dir, exist_ok=True)
    weight_dir = os.path.join(model_weights_main_dir, curr_weight_dir)
    if not os.path.exists(weight_dir): os.mkdir(weight_dir)
    return weight_dir

def save_model_weights(detector, weight_dir, weights_name):
    weight_path = os.path.join(weight_dir, weights_name)
    torch.save(detector.state_dict(), weight_path)

def skip_batch(loss):
    skip = False
    if torch.isnan(loss):
        skip = True
        print('WARNING!...skipping current batch since loss is nan due to corrupted sample')
    return skip

# --------------------------------------------------------------------------------------------------------------
def train_model(
    detector,
    optimizer,
    lr_scheduler,
    dataloader_train,
    dataloader_val,
    tb_writer,
    max_iters,
    log_period,
    val_period,
    model_weights_dir,
    weights_name):

    loss_tracker = LossTracker()        # track losses for tensorboard visualization
    acc_tracker = AccuracyTracker()     # track accuracies for tensorboard visualization
    weight_dir = create_weight_dir(model_weights_dir)    # create directory to save model weights 

    for iter_train_outer in range(max_iters):
        graph_features, labels = next(dataloader_train) 

        if graph_features != None:
            detector.train()
            loss, accuracy = detector(
                node_features = graph_features['node_features_dyn'],
                edge_features = graph_features['edge_features_dyn'],
                other_features = graph_features['other_features_dyn'],
                edge_index = graph_features['edge_index_dyn'],
                adj_matrix = graph_features['adj_matrix_dyn'],
                node_class_labels = labels['node_class'] )

            optimizer.zero_grad()
            skip = skip_batch(loss)
            if skip == False:
                loss.backward()
                optimizer.step()
                if lr_scheduler != None: lr_scheduler.step()  

        # ==============================================================================================      
        # validate model, print loss on console, save the model weights
        if loss > 0:
            loss_tracker.append_training_loss_for_tb(loss)
            loss_tracker.loss_history.append(loss.item())
            acc_tracker.append_training_acc_for_tb(accuracy)
    
        if iter_train_outer % log_period == 0:
            loss_str = f"[Iter {iter_train_outer}][loss: {loss.item():.5f}]"
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
                for iter_val, (graph_features, labels) in enumerate(dataloader_val):

                    if graph_features != None:
                        loss, accuracy = detector(
                            node_features = graph_features['node_features_dyn'],
                            edge_features = graph_features['edge_features_dyn'],
                            other_features = graph_features['other_features_dyn'],
                            edge_index = graph_features['edge_index_dyn'],
                            adj_matrix = graph_features['adj_matrix_dyn'],
                            node_class_labels = labels['node_class'] )
                        
                        if loss > 0:
                            loss_tracker.append_validation_loss_for_tb(loss)    
                            acc_tracker.append_validation_acc_for_tb(accuracy) 

                        if iter_val % log_period == 0:   # Print losses periodically on the console
                            loss_str = f"[Iter {iter_val}][loss: {loss.item():.5f}]"
                            print(loss_str)

            # ==============================================================================================
            # write the loss to tensor board
            obj_cls_train_loss_tb = loss_tracker.compute_avg_training_loss()
            obj_cls_val_loss_tb = loss_tracker.compute_avg_val_loss()

            print('train_losses_tb : ', obj_cls_train_loss_tb)          
            print('val_losses_tb   : ', obj_cls_val_loss_tb) 

            Object_Classification_Loss = {'train':obj_cls_train_loss_tb, 'val':obj_cls_val_loss_tb}
            tb_writer.add_scalars('Loss_Object_Classification', Object_Classification_Loss, iter_train_outer)

            # reset train_losses_tb
            loss_tracker.reset_training_loss_for_tb()
            loss_tracker.reset_validation_loss_for_tb()

            # ==============================================================================================
            # write the accuracy to tensor board
            obj_cls_train_acc_tb = acc_tracker.compute_avg_training_acc()
            obj_cls_val_acc_tb = acc_tracker.compute_avg_val_acc()

            Object_Classification_Acc = {'train':obj_cls_train_acc_tb, 'val':obj_cls_val_acc_tb}
            tb_writer.add_scalars('Acc_Object_Classification', Object_Classification_Acc, iter_train_outer)

            # reset train_losses_tb
            acc_tracker.reset_training_acc_for_tb()
            acc_tracker.reset_validation_acc_for_tb()

            print("end of validation : Resuming Training")
            print("-"*100)

# --------------------------------------------------------------------------------------------------------------
class LossTracker:
    def __init__(self):
        # training losses
        self.loss_history = []      # Keep track of training loss for plotting.
        self.train_losses_tb = []   # train_losses for tensor board visualization
        self.val_losses_tb = []     # val losses

    def append_training_loss_for_tb(self, total_loss):
        self.train_losses_tb.append(total_loss.item()) 

    def append_validation_loss_for_tb(self, total_loss):
        self.val_losses_tb.append(total_loss.item()) 

    def reset_training_loss_for_tb(self):
        self.train_losses_tb = []   

    def reset_validation_loss_for_tb(self):
        self.val_losses_tb = []

    def compute_avg_training_loss(self):
        total_train_loss_tb = np.mean(np.array(self.train_losses_tb))   
        return total_train_loss_tb
    
    def compute_avg_val_loss(self):
        val_loss_tb = np.mean(np.array(self.val_losses_tb))   
        return val_loss_tb
    
# --------------------------------------------------------------------------------------------------------------
class AccuracyTracker:
    def __init__(self):
        self.obj_cls_train_acc_tb = []   # object classification train_accuracy for tensor board visualization
        self.obj_cls_val_acc_tb = []   

    def append_training_acc_for_tb(self, accuracy):
        self.obj_cls_train_acc_tb.append(accuracy.item())

    def append_validation_acc_for_tb(self, accuracy):
        self.obj_cls_val_acc_tb.append(accuracy.item())

    def reset_training_acc_for_tb(self):
        self.obj_cls_train_acc_tb = []  

    def reset_validation_acc_for_tb(self):
        self.obj_cls_val_acc_tb = []   

    def compute_avg_training_acc(self):
        obj_cls_train_acc_tb = np.mean(np.array(self.obj_cls_train_acc_tb))
        return obj_cls_train_acc_tb
    
    def compute_avg_val_acc(self):
        obj_cls_val_acc_tb = np.mean(np.array(self.obj_cls_val_acc_tb))
        return obj_cls_val_acc_tb
    
    