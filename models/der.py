import logging
import multiprocessing

import numpy as np
from tqdm import tqdm
import torch
from torch import nn
from torch import optim
from torch.nn import functional as F
from torch.utils.data import DataLoader
from models.base import BaseLearner
from utils.inc_net import DERNet, IncrementalNet
from utils.toolkit import count_parameters, target2onehot, tensor2numpy
import wandb

EPSILON = 1e-8

# Init task
init_epoch = 200
init_lr = 0.001
init_milestones = [70, 150]
init_lr_decay = 0.1
init_weight_decay = 0
init_early_stop_patience = 40

# Incremental task
epochs = 150
lrate = 0.001
milestones = [60, 100]
lrate_decay = 0.1
weight_decay = 0
early_stop_patience = 30

num_workers = multiprocessing.cpu_count()
batch_size = 64

sparsity_lambda = 4


class DER(BaseLearner):

    def __init__(self, args):
        super().__init__(args)
        self._network = DERNet(args['convnet_type'], args['pretrained'], dropout=args.get('dropout'))
        self.weight_align = args.get('weight_align', True)
        self.min_delta = args.get('min_delta', 0)
        self.sparsity_lambda = args.get('sparsity_lambda', 0)

    def after_task(self):
        self._known_classes = self._total_classes
        logging.info('Exemplar size: {}'.format(self.exemplar_size))

    def incremental_train(self, data_manager):
        self._cur_task += 1
        self._total_classes = self._known_classes + data_manager.get_task_size(self._cur_task)
        self._network.update_fc(self._total_classes)
        logging.info('Learning on {}-{}'.format(self._known_classes, self._total_classes))

        if self._cur_task > 0:
            for i in range(self._cur_task):
                for p in self._network.convnets[i].parameters():
                    p.requires_grad = False

        logging.info('All params: {}'.format(count_parameters(self._network)))
        logging.info('Trainable params: {}'.format(count_parameters(self._network, True)))

        # Train split
        train_dataset = data_manager.get_dataset(
            np.arange(self._known_classes, self._total_classes),
            source='train', mode='train', appendent=self._get_memory()
        )
        self.train_loader = DataLoader(
            train_dataset, batch_size=batch_size, shuffle=True, num_workers=num_workers)
        # Validation split
        val_dataset = data_manager.get_dataset(
            np.arange(0, self._total_classes),
            source='val', mode='test'
        )
        self.val_loader = DataLoader(
            val_dataset, batch_size=batch_size, shuffle=False, num_workers=num_workers)
        # Test split
        test_dataset = data_manager.get_dataset(
            np.arange(0, self._total_classes),
            source='test', mode='test'
        )
        self.test_loader = DataLoader(
            test_dataset, batch_size=batch_size, shuffle=False, num_workers=num_workers)

        self._network = nn.DataParallel(self._network, self._multiple_gpus)
        self._train(self.train_loader, self.val_loader)
        self.build_rehearsal_memory(data_manager, self.samples_per_class)
        self._network = self._network.module

        # Prune network
        if self.sparsity_lambda:
            n_param_before_pruning = count_parameters(self._network)
            logging.info(f'Pruning: N. parameters before pruning: {n_param_before_pruning / 10**6:.2f}M')
            self._network.prune_last_cnn()
            n_param_after_pruning = count_parameters(self._network)
            logging.info(f'Pruning: N. parameters after pruning: {n_param_after_pruning / 10**6:.2f}M')
            n_pruned_parameters = n_param_before_pruning - n_param_after_pruning
            logging.info(f'Pruning: N. pruned parameters: {n_pruned_parameters / 10**6:.2f}M')

    def train(self):
        self._network.train()
        self._network.module.convnets[-1].train()
        if self._cur_task >= 1:
            for i in range(self._cur_task):
                self._network.module.convnets[i].eval()

    def _train(self, train_loader, validation_loader):
        self._network.to(self._device)
        if self._cur_task == 0:
            lr_task, weight_decay_task = init_lr, init_weight_decay
            milestones_task, gamma_task = init_milestones, init_lr_decay
        else:
            lr_task, weight_decay_task = lrate_decay, weight_decay
            milestones_task, gamma_task = milestones, lrate_decay

        parameters = list(filter(lambda p: p.requires_grad, self._network.parameters()))

        # Embeddings for pruning
        if self.sparsity_lambda:
            parameters += self._network.module.e

        optimizer = optim.Adam(parameters, lr=lr_task, weight_decay=weight_decay_task)

        scheduler = optim.lr_scheduler.MultiStepLR(
            optimizer=optimizer, milestones=milestones_task, gamma=gamma_task)
        self._update_representation(train_loader, validation_loader, optimizer, scheduler)

        if self._cur_task > 0 and self.weight_align:
            self._network.module.weight_align(self._total_classes - self._known_classes)

    def _update_representation(
            self, train_loader, val_loader, optimizer, scheduler, patience=early_stop_patience):
        val_acc_list = []
        prog_bar = tqdm(range(epochs))

        # Early stopping
        best_network_so_far = self._network.module.copy()
        if self.sparsity_lambda != 0:
            best_stopping_value = float('inf')
        else:
            best_stopping_value = 0

        curr_patience = patience

        for _, epoch in enumerate(prog_bar):

            # Early stopping
            if curr_patience == 0:
                logging.info(f"Early stopping on task {self._cur_task}, Epoch {epoch}/{epochs}")
                break

            self.train()
            losses = 0.
            losses_clf = 0.
            losses_aux = 0.
            losses_sparsity = 0.
            correct, total = 0, 0
            for b, (_, inputs, targets) in enumerate(train_loader):
                inputs, targets = inputs.to(self._device), targets.to(self._device)
                outputs = self._network(inputs, b=b, B=len(train_loader))
                logits, aux_logits, loss_sparsity = (
                    outputs["logits"], outputs["aux_logits"], outputs["sparsity_loss"])
                loss_clf = F.cross_entropy(logits, targets)
                aux_targets = targets.clone()
                aux_targets = torch.where(aux_targets - self._known_classes + 1 > 0,
                                          aux_targets - self._known_classes + 1, 0)
                loss_aux = F.cross_entropy(aux_logits, aux_targets)
                loss = loss_clf + loss_aux + (sparsity_lambda * loss_sparsity)

                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
                losses += loss.item()
                losses_aux += loss_aux.item()
                losses_clf += loss_clf.item()
                losses_sparsity += loss_sparsity.item()

                _, preds = torch.max(logits, dim=1)
                correct += preds.eq(targets.expand_as(preds)).cpu().sum()
                total += len(targets)

            scheduler.step()
            train_acc = np.around(tensor2numpy(correct) * 100 / total, decimals=2)

            val_acc, loss = self._compute_accuracy(
                self._network, val_loader, sparsity_lambda=sparsity_lambda)

            # Early stopping
            if self.sparsity_lambda != 0:
                stopping_value = sum(loss.values())
            else:
                stopping_value = val_acc

            if self.sparsity_lambda != 0:
                need_to_stop = lambda x: x + self.min_delta <= best_stopping_value
            else:
                need_to_stop = lambda x: x - self.min_delta >= best_stopping_value

            if need_to_stop(stopping_value):
                curr_patience = patience
                best_network_so_far = self._network.module.copy()
                best_stopping_value = stopping_value
            else:
                curr_patience -= 1

            info = 'Task {}, Epoch {}/{} => ' \
                   'Loss {:.3f}, Loss_clf {:.3f}, ' \
                   'Loss_sparsity {:.3f}, ' \
                   'Loss_aux {:.3f}, ' \
                   'Train_accy {:.2f}, ' \
                   'Val_accy {:.2f}' \
                .format(self._cur_task, epoch + 1, epochs, losses / len(train_loader),
                        losses_clf / len(train_loader), losses_aux / len(train_loader),
                        losses_sparsity / len(train_loader), train_acc, val_acc)
            val_acc_list.append(val_acc)

            prog_bar.set_description(info)

            # Wandb
            wandb.log({
                f'task{self._cur_task}/train_acc': train_acc,
                f'task{self._cur_task}/val_acc': val_acc,
                f'task{self._cur_task}/val_clf_loss': loss['clf'],
                f'task{self._cur_task}/val_sparsity_loss': loss['sparsity'],
                f'task{self._cur_task}/val_loss': sum(loss.values()),
                'epoch': epoch})

        # Use the best network
        self._network.module = best_network_so_far

        logging.info(info)
        logging.info(f'Task {self._cur_task}, Accuracy validation history => {val_acc_list}')
