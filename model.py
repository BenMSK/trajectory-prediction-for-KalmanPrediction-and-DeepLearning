import torch
import torch.nn as nn
from torch import optim
from vanilla_gru import NNPred
from utils import DataSet, maskedMSE, maskedLastPositionLoss, class_objtype
from torch.utils.data import DataLoader
from torch.autograd import gradcheck

import os
import matplotlib.pyplot as plt
import numpy as np
import time
import wandb
import colorama
from colorama import Fore, Style

class VanillaGRU:
    def __init__(self, args):

        self.args = {}
        self.args['dataset'] = args.dataset_name
        self.args["batch_size"] = args.batch_size
        self.args['cuda'] = args.cuda
        self.args['device'] = args.device
        self.args['epoch'] = args.epoch
        
        self.args['in_length'] = args.obs_length
        self.args['out_length'] = args.pred_length
        self.args['save_name'] = args.save_name
        self.args['load_name'] = args.load_name

        # self.args['nll_only'] = True
        self.args["learning_rate"] = 1e-4
        self.args["w_decay"] = 1e-4
        
        self.args['name'] = 'test.tar'
        self.args["optim"] = 'Adam'
        self.args['train_loss'] = 'MSE'

        self.wandb = False
        if args.wandb:
            self.wandb = True
            print("Wandb is initialized...")
            wandb.init(project="vanilla_gru",\
                # name="obs: {}, pred: {}".format(args.obs_length, args.pred_length),
                config=self.args)
            
        self.net = NNPred(args)
        if self.args['cuda']:
            self.net = self.net.cuda(self.args['device'])

        # for training// dataset
        self.train_dataset = DataSet(args, 'train')
        self.val_dataset = DataSet(args, 'val')
        self.test_dataset = DataSet(args, 'test')
        # for i in range(12300, 12500):
        # hist, hist_mask, fut, fut_mask, ref_pose, AgentInfo = self.test_dataset[0]
        # print(AgentInfo)
        # print(hist)
        # print(fut)
        # # A = fut+ref_pose#// These lines for 
        # # A[:,0] = ((A[:,0]+1)/2)*(self.test_dataset.max_position_x - self.test_dataset.min_position_x)+ self.test_dataset.min_position_x
        # # A[:,1] = ((A[:,1]+1)/2)*(self.test_dataset.max_position_y - self.test_dataset.min_position_y)+ self.test_dataset.min_position_y
        # print(A)
        # # print(fut[:,1]+ref_pose[1])
        # quit()
        
        self.trainDataloader = DataLoader(self.train_dataset, batch_size=self.args["batch_size"], shuffle=True, \
                                 num_workers=6, collate_fn=self.train_dataset.GetBatch, drop_last=True)
        # print("trainDataloader completed!")
        self.valDataloader = DataLoader(self.val_dataset, batch_size=self.args["batch_size"], shuffle=True, \
                                num_workers=6, collate_fn=self.val_dataset.GetBatch, drop_last=True)    
        # print("valDataloader completed!")
        self.testDataloader = DataLoader(self.test_dataset, batch_size=self.args["batch_size"], shuffle=True, \
                                num_workers=6, collate_fn=self.test_dataset.GetBatch, drop_last=True)    
        # print("testDataloader completed!")                                
        # time.sleep(300)

    def train(self):   

        total_epoch = self.args['epoch']
        avg_trn_loss = 0
        avg_val_loss = 0
        if self.args["optim"] == 'Adam':
            optim = torch.optim.Adam(self.net.parameters(),lr=self.args['learning_rate'],weight_decay=self.args["w_decay"])
        else:
            print("Undefined optimizer.")
            return
        criterion = torch.nn.MSELoss()
        print("Start training....")
        for epoch in range(total_epoch):
            tr_count = 0
            val_count = 0
            self.net.train_flag = True
            # for data in self.testDataloader:#for check
            for data in self.trainDataloader:# TRAINING PHASE
                # print("count! ", tr_count)
                hist_batch, fut_batch, fut_mask_batch, _, _ = data
                if self.args['cuda']:
                    hist_batch = hist_batch.cuda(self.args['device'])
                    fut_batch = fut_batch.cuda(self.args['device'])
                    fut_mask_batch = fut_mask_batch.cuda(self.args['device'])
                
                fut_pred = self.net(hist_batch, fut_batch)
                # fut_mask_batch = torch.cat((fut_mask_batch, fut_mask_batch),2)#if network output has sigX
                
                fut_pred = fut_mask_batch*fut_pred
                
                loss = criterion(fut_pred, fut_batch)
                # loss = maskedMSE(fut_pred, fut_batch, fut_mask_batch, device=self.args['device'])
                # loss = maskedLastPositionLoss(fut_pred, fut_batch, fut_mask_batch, device=self.args['device'])
                optim.zero_grad()
                loss.backward()
                
                # nn.utils.clip_grad_norm_(self.net.parameters(), 1)
                optim.step()
                
                avg_trn_loss += loss.item()
                tr_count+=1
                if self.wandb and self.args['dataset']=='Lyft':
                    # print("logging!")
                    wandb.log({'Avg Train Loss per step': loss.item()})
                
            avg_trn_loss /= tr_count
            print("Epoch: {} AvgTrainLoss: {}".format(epoch, avg_trn_loss))

            ############################################################################################
            self.net.train_flag = False
            for data in self.valDataloader:# VALIDATION PHASE
                hist_batch, fut_batch, fut_mask_batch, _, _ = data
                if self.args['cuda']:
                    hist_batch = hist_batch.cuda(self.args['device'])
                    fut_batch = fut_batch.cuda(self.args['device'])
                    fut_mask_batch = fut_mask_batch.cuda(self.args['device'])
    
                fut_pred = self.net(hist_batch, fut_batch)

                # fut_mask_batch = torch.cat((fut_mask_batch, fut_mask_batch),2)#if network output has sigX
                
                fut_pred = fut_mask_batch*fut_pred
                
                # loss = maskedMSE(fut_pred, fut_batch, fut_mask_batch, device=self.args['device'])
                # loss = maskedLastPositionLoss(fut_pred, fut_batch, fut_mask_batch, device=self.args['device'])
                loss = criterion(fut_pred, fut_batch)

                avg_val_loss += loss.item()
                val_count+=1
                if self.wandb and self.args['dataset']=='Lyft':
                    wandb.log({'Avg Val Loss per step': loss.item()})

            avg_val_loss /= val_count
            print("Epoch: {} AvgValLoss: {}".format(epoch, avg_val_loss))
            print("===================================================")
            if self.wandb:
                wandb.log({'Avg Train Loss per Epoch': avg_trn_loss, 'Avg Val Loss per Epoch': avg_val_loss})
            self.saveModel(epoch)

    def saveModel(self, epoch):
        save_dir = os.path.join("./weight/", str(epoch)+"."+self.args['save_name'])
        # name = os.path.join(self.args['modelLoc'], "epochs.{}.".format(engine.state.epoch)+self.args['name'])
        torch.save(self.net.state_dict(), save_dir)
        print("Model saved {}.".format(save_dir))
    

    def load(self):
        load_dir = os.path.join("./weight/", self.args['load_name'])
        if os.path.exists(load_dir):
            self.net.load_state_dict(torch.load(load_dir))
            print(Fore.YELLOW)
            print("\n[INFO]: model {} loaded, successfully!\n".format(load_dir))
            print(Style.RESET_ALL)

        else:
            print(Fore.RED)
            print("\n[INFO]: CAN NOT FIND MODEL AT {}".format(load_dir))
            print(Style.RESET_ALL)

    def test(self):
        self.net.train_flag = False
        # (m_x, m_y, sigX, sigY) = self.test_dataset.scale_param# for gaussian scailing
        (min_position_x, max_position_x, min_position_y, max_position_y) = self.test_dataset.scale_param
        for data in self.testDataloader:
            hist_batch, fut_batch, fut_mask_batch, ref_pose_batch, AgentsInfo = data
            if self.args['cuda']:
                hist_batch = hist_batch.cuda(self.args['device'])
            
            fut_pred_batch = self.net(hist_batch, fut_batch)
            
            # print(fut_pred_batch.shape)
            hist_batch = hist_batch.cpu().detach().numpy()
            fut_pred_batch = fut_pred_batch[:,:,:2].cpu().detach().numpy()
            fut_batch = fut_batch.numpy()
            fut_mask_batch = fut_mask_batch.numpy()
            ref_pose_batch = ref_pose_batch.numpy()

            for ind, agent_traj in enumerate(fut_batch):
                print("dataset: {}, AgentID: {}, AgentType: {}"\
                    .format(AgentsInfo[ind][0], AgentsInfo[ind][2], class_objtype(AgentsInfo[ind][3], self.args['dataset'])))
                
                # masking point helps for cutting non-enough gt case when visualizing
                masking_point = np.where(fut_mask_batch[ind]==0)[0][0] if np.where(fut_mask_batch[ind]==0)[0].shape[0] != 0 else len(fut_mask_batch[ind])
                
                agent_traj += ref_pose_batch[ind]
                fut_pred_batch[ind] += ref_pose_batch[ind]
                hist_batch[ind] += ref_pose_batch[ind]
                hist_x = ((hist_batch[ind][:,0]+1)/2)*(max_position_x - min_position_x) + min_position_x
                hist_y = ((hist_batch[ind][:,1]+1)/2)*(max_position_y - min_position_y) + min_position_y

                gt_x = ((fut_batch[ind][:masking_point,0]+1)/2)*(max_position_x - min_position_x) + min_position_x
                gt_y = ((fut_batch[ind][:masking_point,1]+1)/2)*(max_position_y - min_position_y) + min_position_y
                # print(gt_x[0], gt_y[0])
                # print(len(gt_x))
                pred_x = ((fut_pred_batch[ind][:masking_point,0]+1)/2)*(max_position_x - min_position_x) + min_position_x
                pred_y = ((fut_pred_batch[ind][:masking_point,1]+1)/2)*(max_position_y - min_position_y) + min_position_y

                plt.plot(hist_x, hist_y, "ko-", alpha=0.5)
                plt.plot(hist_x[-1], hist_y[-1], "yo", alpha=0.8)
                plt.plot(gt_x, gt_y, "g^-")
                plt.plot(pred_x, pred_y, "r^-")
                plt.axis('equal')
                
                plt.show()




