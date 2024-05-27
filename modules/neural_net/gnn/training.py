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
    iter_start_offset,
    model_weights_dir,
    weights_name):

    loss_tracker = LossTracker()        # track losses for tensorboard visualization
    acc_tracker = AccuracyTracker()     # track accuracies for tensorboard visualization
    weight_dir = create_weight_dir(model_weights_dir)    # create directory to save model weights 

    for iter_train_outer in range(iter_start_offset, max_iters):
        graph_features, labels = next(dataloader_train) 

        if graph_features != None:
            detector.train()
            loss, accuracy = detector(
                node_features = graph_features['node_features_dyn'],
                edge_features = graph_features['edge_features_dyn'],
                edge_index = graph_features['edge_index_dyn'],
                adj_matrix = graph_features['adj_matrix_dyn'],
                labels = labels )
            total_loss = loss['loss_node_cls'] + loss['loss_node_reg'] + loss['loss_edge_cls'] + loss['loss_obj_cls']

            skip = skip_batch(total_loss)
            if skip == False:
                total_loss.backward()
                optimizer.step()
                if lr_scheduler != None:
                    lr_scheduler.step()  
            optimizer.zero_grad()

        # ==============================================================================================      
        # validate model, print loss on console, save the model weights
        if total_loss > 0:
            loss_tracker.append_training_loss_for_tb(total_loss, loss)
            loss_tracker.loss_history.append(total_loss.item())
            acc_tracker.append_training_acc_for_tb(accuracy)
    
        if iter_train_outer % log_period == 0:
            loss_str = f"[Iter {iter_train_outer}][loss: {total_loss.item():.5f}]"
            for key, value in loss.items():
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
                for iter_val, (graph_features, labels) in enumerate(dataloader_val):

                    if graph_features != None:   
                        loss, accuracy = detector(
                            node_features = graph_features['node_features_dyn'],
                            edge_features = graph_features['edge_features_dyn'],
                            edge_index = graph_features['edge_index_dyn'],
                            adj_matrix = graph_features['adj_matrix_dyn'],
                            labels = labels )
                        total_loss = sum(loss.values())

                        if total_loss > 0:
                            loss_tracker.append_validation_loss_for_tb(total_loss, loss)    
                            acc_tracker.append_validation_acc_for_tb(accuracy) 

                        if iter_val % log_period == 0:   # Print losses periodically on the console
                            loss_str = f"[Iter {iter_val}][loss: {total_loss.item():.5f}]"
                            for key, value in loss.items():
                                loss_str += f"[{key}: {value.item():.5f}]"
                            print(loss_str)

            # ==============================================================================================
            # write the loss to tensor board
            train_loss_tb, node_cls_train_loss_tb, node_reg_train_loss_tb, edge_cls_train_loss_tb, obj_cls_train_loss_tb \
                = loss_tracker.compute_avg_training_loss()

            val_loss_tb, node_cls_val_loss_tb, node_reg_val_loss_tb, edge_cls_val_loss_tb, obj_cls_val_loss_tb  \
                = loss_tracker.compute_avg_val_loss()

            print('train_losses_tb : ', train_loss_tb)          
            print('val_losses_tb   : ', val_loss_tb)      

            Total_Loss = {'train':train_loss_tb, 'val':val_loss_tb}
            tb_writer.add_scalars('Total_Loss', Total_Loss, iter_train_outer)

            Node_Classification_Loss = {'train':node_cls_train_loss_tb, 'val':node_cls_val_loss_tb}
            tb_writer.add_scalars('Loss_Node_Segmentation', Node_Classification_Loss, iter_train_outer)

            Node_Offset_Loss = {'train':node_reg_train_loss_tb, 'val':node_reg_val_loss_tb}
            tb_writer.add_scalars('Loss_Node_Offset', Node_Offset_Loss, iter_train_outer)
            
            Edge_Classification_Loss = {'train':edge_cls_train_loss_tb, 'val':edge_cls_val_loss_tb}
            tb_writer.add_scalars('Loss_Edge_Classification', Edge_Classification_Loss, iter_train_outer)

            Object_Classification_Loss = {'train':obj_cls_train_loss_tb, 'val':obj_cls_val_loss_tb}
            tb_writer.add_scalars('Loss_Object_Classification', Object_Classification_Loss, iter_train_outer)

            # reset train_losses_tb
            loss_tracker.reset_training_loss_for_tb()
            loss_tracker.reset_validation_loss_for_tb()

            # ==============================================================================================
            # write the accuracy to tensor board

            node_cls_train_acc_tb, edge_cls_train_acc_tb, obj_cls_train_acc_tb \
                = acc_tracker.compute_avg_training_acc()

            node_cls_val_acc_tb, edge_cls_val_acc_tb, obj_cls_val_acc_tb  \
                = acc_tracker.compute_avg_val_acc()

            Node_Segmentation_Acc = {'train':node_cls_train_acc_tb, 'val':node_cls_val_acc_tb}
            tb_writer.add_scalars('Acc_Node_Segmentation', Node_Segmentation_Acc, iter_train_outer)
            
            Edge_Classification_Acc = {'train':edge_cls_train_acc_tb, 'val':edge_cls_val_acc_tb}
            tb_writer.add_scalars('Acc_Edge_Classification', Edge_Classification_Acc, iter_train_outer)

            Object_Classification_Acc = {'train':obj_cls_train_acc_tb, 'val':obj_cls_val_acc_tb}
            tb_writer.add_scalars('Acc_Object_Classification', Object_Classification_Acc, iter_train_outer)

            # reset train_losses_tb
            acc_tracker.reset_training_acc_for_tb()
            acc_tracker.reset_validation_acc_for_tb()

            print("end of validation : Resuming Training")
            print("-"*100)

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
    acc_tracker = AccuracyTracker()     # track accuracies for tensorboard visualization
    weight_dir = create_weight_dir(model_weights_dir)    # create directory to save model weights 

    for iter_train_outer in range(iter_start_offset, max_iters):
        detector.train()
        skip = True
        for _ in range(grad_accumulation_steps):
            # ==============================================================================================
            graph_features, labels = next(dataloader_train)  

            if graph_features != None:
                loss, accuracy = detector(
                    node_features = graph_features['node_features_dyn'],
                    edge_features = graph_features['edge_features_dyn'],
                    edge_index = graph_features['edge_index_dyn'],
                    adj_matrix = graph_features['adj_matrix_dyn'],
                    labels = labels )
                total_loss = loss['loss_node_cls'] + loss['loss_node_reg'] + loss['loss_edge_cls'] + loss['loss_obj_cls']
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
        if total_loss > 0 and (not torch.isnan(total_loss)):
            loss_tracker.append_training_loss_for_tb(total_loss, loss)
            loss_tracker.loss_history.append(total_loss.item())
            acc_tracker.append_training_acc_for_tb(accuracy)

        if iter_train_outer % log_period == 0:
            loss_str = f"[Iter {iter_train_outer}][loss: {total_loss.item():.5f}]"
            for key, value in loss.items():
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
                for iter_val, (graph_features, labels) in enumerate(dataloader_val):

                    if graph_features != None:     
                        loss, accuracy = detector(
                            node_features = graph_features['node_features_dyn'],
                            edge_features = graph_features['edge_features_dyn'],
                            edge_index = graph_features['edge_index_dyn'],
                            adj_matrix = graph_features['adj_matrix_dyn'],
                            labels = labels )
                        total_loss = sum(loss.values())
                        if total_loss > 0:
                            loss_tracker.append_validation_loss_for_tb(total_loss, loss)  
                            acc_tracker.append_validation_acc_for_tb(accuracy)   

                        if iter_val % log_period == 0:   # Print losses periodically on the console
                            loss_str = f"[Iter {iter_val}][loss: {total_loss.item():.5f}]"
                            for key, value in loss.items():
                                loss_str += f"[{key}: {value.item():.5f}]"
                            print(loss_str)

            # ==============================================================================================
            # write the loss to tensor board
            train_loss_tb, node_cls_train_loss_tb, node_reg_train_loss_tb, edge_cls_train_loss_tb, obj_cls_train_loss_tb \
                = loss_tracker.compute_avg_training_loss()

            val_loss_tb, node_cls_val_loss_tb, node_reg_val_loss_tb, edge_cls_val_loss_tb, obj_cls_val_loss_tb  \
                = loss_tracker.compute_avg_val_loss()

            print('train_losses_tb : ', train_loss_tb)          
            print('val_losses_tb   : ', val_loss_tb)      

            Total_Loss = {'train':train_loss_tb, 'val':val_loss_tb}
            tb_writer.add_scalars('Total_Loss', Total_Loss, iter_train_outer)

            Node_Classification_Loss = {'train':node_cls_train_loss_tb, 'val':node_cls_val_loss_tb}
            tb_writer.add_scalars('Loss_Node_Segmentation', Node_Classification_Loss, iter_train_outer)

            Node_Offset_Loss = {'train':node_reg_train_loss_tb, 'val':node_reg_val_loss_tb}
            tb_writer.add_scalars('Loss_Node_Offset', Node_Offset_Loss, iter_train_outer)
            
            Edge_Classification_Loss = {'train':edge_cls_train_loss_tb, 'val':edge_cls_val_loss_tb}
            tb_writer.add_scalars('Loss_Edge_Classification', Edge_Classification_Loss, iter_train_outer)

            Object_Classification_Loss = {'train':obj_cls_train_loss_tb, 'val':obj_cls_val_loss_tb}
            tb_writer.add_scalars('Loss_Object_Classification', Object_Classification_Loss, iter_train_outer)

            # reset train_losses_tb
            loss_tracker.reset_training_loss_for_tb()
            loss_tracker.reset_validation_loss_for_tb()

            # ==============================================================================================
            # write the accuracy to tensor board

            node_cls_train_acc_tb, edge_cls_train_acc_tb, obj_cls_train_acc_tb \
                = acc_tracker.compute_avg_training_acc()

            node_cls_val_acc_tb, edge_cls_val_acc_tb, obj_cls_val_acc_tb  \
                = acc_tracker.compute_avg_val_acc()

            Node_Segmentation_Acc = {'train':node_cls_train_acc_tb, 'val':node_cls_val_acc_tb}
            tb_writer.add_scalars('Acc_Node_Segmentation', Node_Segmentation_Acc, iter_train_outer)
            
            Edge_Classification_Acc = {'train':edge_cls_train_acc_tb, 'val':edge_cls_val_acc_tb}
            tb_writer.add_scalars('Acc_Edge_Classification', Edge_Classification_Acc, iter_train_outer)

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
        self.node_cls_train_losses_tb = []  # node classification train_losses for tensor board visualization
        self.node_reg_train_losses_tb = []  # node box regression train_losses for tensor board visualization
        self.edge_cls_train_losses_tb = []  # edge classification train_losses for tensor board visualization
        self.obj_cls_train_losses_tb = []  # object classification train_losses for tensor board visualization

        # val losses
        self.val_losses_tb = []
        self.node_cls_val_losses_tb = []  
        self.node_reg_val_losses_tb = []  
        self.edge_cls_val_losses_tb = []  
        self.obj_cls_val_losses_tb = []   

    def append_training_loss_for_tb(self, total_loss, losses):
        self.train_losses_tb.append(total_loss.item()) 
        self.node_cls_train_losses_tb.append(losses['loss_node_cls'].item())
        self.node_reg_train_losses_tb.append(losses['loss_node_reg'].item())
        self.edge_cls_train_losses_tb.append(losses['loss_edge_cls'].item())
        self.obj_cls_train_losses_tb.append(losses['loss_obj_cls'].item())

    def append_validation_loss_for_tb(self, total_loss, losses):
        self.val_losses_tb.append(total_loss.item()) 
        self.node_cls_val_losses_tb.append(losses['loss_node_cls'].item())
        self.node_reg_val_losses_tb.append(losses['loss_node_reg'].item())
        self.edge_cls_val_losses_tb.append(losses['loss_edge_cls'].item())
        self.obj_cls_val_losses_tb.append(losses['loss_obj_cls'].item())

    def reset_training_loss_for_tb(self):
        self.train_losses_tb = []   
        self.node_cls_train_losses_tb = []  
        self.node_reg_train_losses_tb = []  
        self.edge_cls_train_losses_tb = []  
        self.obj_cls_train_losses_tb = []

    def reset_validation_loss_for_tb(self):
        self.val_losses_tb = []
        self.node_cls_val_losses_tb = []  
        self.node_reg_val_losses_tb = []   
        self.edge_cls_val_losses_tb = []
        self.obj_cls_val_losses_tb = []  

    def compute_avg_training_loss(self):
        total_train_loss_tb = np.mean(np.array(self.train_losses_tb))   
        node_cls_train_loss_tb = np.mean(np.array(self.node_cls_train_losses_tb))
        node_reg_train_loss_tb = np.mean(np.array(self.node_reg_train_losses_tb))
        edge_cls_train_loss_tb = np.mean(np.array(self.edge_cls_train_losses_tb))
        obj_cls_train_loss_tb = np.mean(np.array(self.obj_cls_train_losses_tb))
        return total_train_loss_tb, node_cls_train_loss_tb, node_reg_train_loss_tb, edge_cls_train_loss_tb, obj_cls_train_loss_tb
    
    def compute_avg_val_loss(self):
        val_loss_tb = np.mean(np.array(self.val_losses_tb))   
        node_cls_val_loss_tb = np.mean(np.array(self.node_cls_val_losses_tb))
        node_reg_val_loss_tb = np.mean(np.array(self.node_reg_val_losses_tb))
        edge_cls_val_loss_tb = np.mean(np.array(self.edge_cls_val_losses_tb))
        obj_cls_val_loss_tb = np.mean(np.array(self.obj_cls_val_losses_tb))
        return val_loss_tb, node_cls_val_loss_tb, node_reg_val_loss_tb, edge_cls_val_loss_tb, obj_cls_val_loss_tb
    
# --------------------------------------------------------------------------------------------------------------
class AccuracyTracker:
    def __init__(self):
        # training acc
        self.node_cls_train_acc_tb = []  # node classification train_accuracy for tensor board visualization
        self.edge_cls_train_acc_tb = []  # edge classification train_accuracy for tensor board visualization
        self.obj_cls_train_acc_tb = []   # object classification train_accuracy for tensor board visualization

        # val acc
        self.node_cls_val_acc_tb = []  
        self.edge_cls_val_acc_tb = []  
        self.obj_cls_val_acc_tb = []   

    def append_training_acc_for_tb(self, accuracy):
        self.node_cls_train_acc_tb.append(accuracy['segment_accuracy'].item())
        self.edge_cls_train_acc_tb.append(accuracy['edge_accuracy'].item())
        self.obj_cls_train_acc_tb.append(accuracy['object_accuracy'].item())

    def append_validation_acc_for_tb(self, accuracy):
        self.node_cls_val_acc_tb.append(accuracy['segment_accuracy'].item())
        self.edge_cls_val_acc_tb.append(accuracy['edge_accuracy'].item())
        self.obj_cls_val_acc_tb.append(accuracy['object_accuracy'].item())

    def reset_training_acc_for_tb(self):
        self.node_cls_train_acc_tb = []  
        self.edge_cls_train_acc_tb = []  
        self.obj_cls_train_acc_tb = []  

    def reset_validation_acc_for_tb(self):
        self.node_cls_val_acc_tb = []  
        self.edge_cls_val_acc_tb = []  
        self.obj_cls_val_acc_tb = []   

    def compute_avg_training_acc(self):
        node_cls_train_acc_tb = np.mean(np.array(self.node_cls_train_acc_tb))
        edge_cls_train_acc_tb = np.mean(np.array(self.edge_cls_train_acc_tb))
        obj_cls_train_acc_tb = np.mean(np.array(self.obj_cls_train_acc_tb))
        return node_cls_train_acc_tb, edge_cls_train_acc_tb, obj_cls_train_acc_tb
    
    def compute_avg_val_acc(self):
        node_cls_val_acc_tb = np.mean(np.array(self.node_cls_val_acc_tb))
        edge_cls_val_acc_tb = np.mean(np.array(self.edge_cls_val_acc_tb))
        obj_cls_val_acc_tb = np.mean(np.array(self.obj_cls_val_acc_tb))
        return node_cls_val_acc_tb, edge_cls_val_acc_tb, obj_cls_val_acc_tb
