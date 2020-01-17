import numpy as np
import torch
import pandas as pd
import torch.nn as nn
from torch.autograd import Variable
import torch.nn.functional as F
import data_preproc_val as dp
from P_BN_1 import Network
#from P_BN_1_relu_0.1 import Network
from torch import Tensor
from tensorboardX import SummaryWriter
#from torch.utils.tensorboard import SummaryWriter

run_mode = 'cuda'
if run_mode == 'cuda':
    ttype = torch.cuda.FloatTensor
else:
    ttype = torch.FloatTensor
writer = SummaryWriter()


class Plasticity:

    def __init__(self):
        self.activation = 'tanh'
        self.update_rule = 'hebb'
        self.learnable_plasticity = True
        # self.nr_classes = y_shape
        self.pattern_shot = 1  # Number of times each pattern is to be presented
        self.presentation_delay = 0  # Duration of zero-input interval between presentations
        # self.image_size = x_shape
        self.learning_rate = 3e-5
        self.nr_iterations = 30000
        self.test_every = 444
        self.nr_it_eval = 44
        self.nb_episodes = 6
        plastic_celeb_data = dp.PlasticDataCreator(run_mode)
        self.plastic_celeb_data = plastic_celeb_data
        self.nr_classes = len(pd.Series(self.plastic_celeb_data.celeb_loader.train_targets).unique())
        self.all_validation_losses = []
        self.avg_valid_losses = []

        return

    def get_num_correct(self, preds, labels):
        return preds.argmax(dim=1).eq(labels).sum().item()

    def test_time(self, num_iter, y, target):
        print(num_iter, "====")
        td = target.cpu().numpy()
        yd = y.data.cpu().numpy()[0]
        print("y: ", yd[:10])
        print("target: ", td[:10])
        error = np.abs(td - yd)
        # print('Mean {np.mean(error)} / median {np.median(error)} / max abs diff {np.max(error)}')
        print("Correlation (full / sign): ", np.corrcoef(td, yd)[0][1], np.corrcoef(np.sign(td), np.sign(yd))[0][1])
        return

    def eval_net(self, net, iteration):
        net.eval()
        validation_losses = []
        n_correct_eval = 0

        for i in range(self.nr_it_eval):
            inputs, labels, test_labels, additional_targets = self.plastic_celeb_data.create_input_plastic_network(
                type_ds='validation')
            for num_step in range(self.nb_episodes):
                y, hebb_trace = net(
                    inputs[num_step],
                    labels[num_step], additional_targets[num_step])
                true_label = labels[num_step]
                if num_step == 5:
                    true_label = test_labels
                to_compare = true_label.long()
                value = to_compare.argmax(dim=1)

                loss = F.cross_entropy(y, value)
                loss_value = float(loss.item())
                validation_losses.append(loss_value)
                n_correct_eval += self.get_num_correct(y, value)

                self.all_validation_losses.append(loss_value)

        accuracy_eval = n_correct_eval / self.nr_it_eval
        avg_validation_loss = sum(validation_losses) / len(validation_losses)
        self.avg_valid_losses.append(avg_validation_loss)
        writer.add_scalar('Accuracy eval', accuracy_eval, iteration)
        writer.add_scalar('Loss avg eval:', avg_validation_loss, iteration)
        #writer.add_histogram('Alpha validation:', net.alpha, iteration)

        return

    def train(self, params):
        net = Network(self.nr_classes, 6, run_mode)
        optimizer = torch.optim.Adam(net.parameters(), lr=1.0 * params['lr'])

        all_losses_objective = []
        lossbetweensaves = 0.0
        n_correct_train = 0

        for iteration in range(self.nr_iterations):
            net.init_hebb_trace(self.nr_classes, 6)
            is_validation_step = ((iteration + 1) % self.test_every == 0)
            is_test_step = False
            inputs, labels, test_labels, additional_targets = self.plastic_celeb_data.create_input_plastic_network(
                type_ds='train')

            #print('--------Episode {0}-------------'.format(iteration))

            for num_step in range(self.nb_episodes):
                optimizer.zero_grad()
                y, hebb_trace = net(
                    Variable(inputs[num_step], requires_grad=False),
                    Variable(labels[num_step], requires_grad=False), additional_targets[num_step])

                true_label = labels[num_step]
                if num_step == 5:
                    true_label = test_labels
                to_compare = true_label.long()
                value = to_compare.argmax(dim=1)
                loss = F.cross_entropy(y, Variable(value, requires_grad=False))

                n_correct_train += self.get_num_correct(y, value)

                #print('Output:{0}'.format(y[0].nonzero()))
                #print('Actual output:{0}'.format(to_compare.nonzero()))
                if is_test_step == False:
                    loss.backward(retain_graph=True)
                    optimizer.step()

            #print('-------------------------------------')
            loss_value = float(loss.item())
            #scheduler.step(loss_value)
            if iteration % 50 == 0 and iteration != 0:
                accuracy_train = n_correct_train / (iteration * self.nb_episodes)*100
                writer.add_scalar('Accuracy train', accuracy_train, iteration)
                n_correct_train = 0
                accuracy_train = 0


            #if is_validation_step:
            #    self.eval_net(net, iteration)

            writer.add_scalar('Loss train :', loss_value, iteration)
            #writer.add_histogram('Alpha train:', net.alpha, iteration)
            # writer.add_histogram('Weights:', net.w, iteration)

            lossbetweensaves += loss_value
            #all_losses_objective.append(loss_value)
        writer.close()
        torch.save(net.state_dict(), './saved')


if __name__ == "__main__":
    plasticity = Plasticity()
    plasticity.train({
        'lr': 0.0006,
        'gamma': .01,
        'nbf': 64,
        'rule': 'hebb',
        'nr_classes': plasticity.nr_classes,
        'alpha': True,
        'steplr': 1e6,
        'activation': 'tanh'
    })
