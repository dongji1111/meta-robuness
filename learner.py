import torch
from torch import nn
from torch.nn import functional as F


class LambdaLayer(nn.Module):
    def __init__(self, lambd):
        super(LambdaLayer, self).__init__()
        self.lambd = lambd

    def forward(self, x):
        return self.lambd(x)


class Learner(nn.Module):

    def __init__(self, config):
        """
        :param config: network config file, type:list of (string, list)
        """
        super(Learner, self).__init__()
        self.config = config
        # this dict contains all tensors needed to be optimized
        self.vars = nn.ParameterList()
        # running_mean and running_var
        #self.vars_bn = nn.ParameterList()
        self.shortcut = nn.Sequential()

        for i, (name, param) in enumerate(self.config):
            if name is 'basicblock':
                conv1 = nn.Parameter(torch.ones(*param[:4]))
                torch.nn.init.kaiming_normal_(conv1)
                self.vars.append(conv1)
                #self.vars.append(nn.Parameter(torch.zeros(param[0])))
                bn1 = nn.Parameter(torch.ones(param[0]))
                self.vars.append(bn1)
                #self.vars.append(nn.Parameter(torch.zeros(param[0])))
                '''running_mean1 = nn.Parameter(torch.zeros(param[0]), requires_grad=False)
                running_var1 = nn.Parameter(torch.ones(param[0]), requires_grad=False)
                self.vars_bn.extend([running_mean1, running_var1])'''

                conv2 = nn.Parameter(torch.ones(param[0],param[0],param[2],param[3]))
                torch.nn.init.kaiming_normal_(conv2)
                self.vars.append(conv2)
                #self.vars.append(nn.Parameter(torch.zeros(param[0])))
                bn2 = nn.Parameter(torch.ones(param[0]))
                self.vars.append(bn2)
                #self.vars.append(nn.Parameter(torch.zeros(param[0])))
                '''running_mean2 = nn.Parameter(torch.zeros(param[0]), requires_grad=False)
                running_var2 = nn.Parameter(torch.ones(param[0]), requires_grad=False)
                self.vars_bn.extend([running_mean2, running_var2])'''

            elif name is 'conv2d':
                # [ch_out, ch_in, kernelsz, kernelsz]
                w = nn.Parameter(torch.ones(*param[:4]))
                # gain=1 according to cbfin's implementation
                torch.nn.init.kaiming_normal_(w)
                self.vars.append(w)
                # [ch_out]
                #self.vars.append(nn.Parameter(torch.zeros(param[0])))

            elif name is 'linear':
                # [ch_out, ch_in]
                w = nn.Parameter(torch.ones(*param))
                # gain=1 according to cbfinn's implementation
                torch.nn.init.kaiming_normal_(w)
                self.vars.append(w)
                # [ch_out]
                self.vars.append(nn.Parameter(torch.zeros(param[0])))

            elif name is 'bn':
                # [ch_out]
                w = nn.Parameter(torch.ones(param[0]))
                self.vars.append(w)
                # [ch_out]
                #self.vars.append(nn.Parameter(torch.zeros(param[0])))

                # must set requires_grad=False
                '''running_mean = nn.Parameter(torch.zeros(param[0]), requires_grad=False)
                running_var = nn.Parameter(torch.ones(param[0]), requires_grad=False)
                self.vars_bn.extend([running_mean, running_var])'''

            elif name in ['tanh', 'relu', 'upsample', 'avg_pool2d', 'max_pool2d',
                          'flatten', 'reshape', 'leakyrelu', 'sigmoid']:
                continue
            else:
                raise NotImplementedError






    def forward(self, x, vars=None, bn_training=True):
        """
        This function can be called by finetunning, however, in finetunning, we dont wish to update
        running_mean/running_var. Thought weights/bias of bn is updated, it has been separated by fast_weights.
        Indeed, to not update running_mean/running_var, we need set update_bn_statistics=False
        but weight/bias will be updated and not dirty initial theta parameters via fast_weiths.
        :param x: [b, c, h, w]
        :param vars:
        :param bn_training: set False to not update
        :return: x, loss, likelihood, kld
        """

        if vars is None:
            vars = self.vars

        idx = 0
        #bn_idx = 0

        for name, param in self.config:
            if name is 'basicblock':
                'conv1,conv1_b,bn1,bn1_b,conv2,conv2_b,bn2,bn2_b = vars[idx],vars[idx+1], vars[idx + 2],vars[idx+3], \
                                                                  vars[idx + 4],vars[idx+5], vars[idx + 6],vars[idx+7]'
                conv1, bn1, conv2,  bn2= vars[idx], vars[idx + 1], vars[idx + 2], vars[idx + 3]
                '''running_mean1, running_var1, running_mean2, running_var2 = self.vars_bn[bn_idx],self.vars_bn[bn_idx + 1],\
                                                                           self.vars_bn[bn_idx+2], self.vars_bn[bn_idx+3]'''
                running_mean = nn.Parameter(torch.zeros(param[0]), requires_grad=False)
                running_var = nn.Parameter(torch.ones(param[0]), requires_grad=False)
                #idx += 8
                idx +=4
                #bn_idx +=4
                '''out = F.relu(F.batch_norm(F.conv2d(x, conv1, conv1_b, stride=param[4], padding=param[5]),
                                          running_mean1,running_var1,weight=bn1,bias=bn1_b,training=bn_training))
                out = F.batch_norm(F.conv2d(out,conv2,conv2_b,stride=1,padding=param[5]),
                                   running_mean2,running_var2, weight=bn2, bias=bn2_b, training=bn_training)'''
                out = F.relu(F.batch_norm(F.conv2d(x, conv1, stride=param[4], padding=param[5]),
                                          running_mean, running_var, weight=bn1, training=bn_training))
                out = F.batch_norm(F.conv2d(out, conv2, stride=1, padding=param[5]),
                                   running_mean, running_var, weight=bn2, training=bn_training)

                if param[4] != 1 or param[1] != param[0]:
                    self.shortcut = LambdaLayer(lambda x:
                                                F.pad(x[:, :, ::2, ::2], (0, 0, 0, 0, param[0] // 4, param[0] // 4),
                                                      "constant", 0))
                out_add = self.shortcut(x)
                #print(out.shape)
                #print(out_add.shape)
                out += self.shortcut(x)
                self.shortcut = nn.Sequential()
                x = F.relu(out)
            elif name is 'conv2d':
                #w, b = vars[idx], vars[idx + 1]
                # remember to keep synchrozied of forward_encoder and forward_decoder!
                #x = F.conv2d(x, w, b, stride=param[4], padding=param[5])
                #idx += 2
                w = vars[idx]
                x = F.conv2d(x, w, stride=param[4], padding=param[5])
                idx += 1
                # print(name, param, '\tout:', x.shape)
            elif name is 'linear':
                w, b = vars[idx], vars[idx + 1]
                x = F.linear(x, w, b)
                idx += 2
                # print('forward:', idx, x.norm().item())
            elif name is 'bn':
                '''w, b = vars[idx], vars[idx + 1]
                running_mean, running_var = self.vars_bn[bn_idx], self.vars_bn[bn_idx+1]
                x = F.batch_norm(x, running_mean, running_var, weight=w, bias=b, training=bn_training)
                idx += 2'''
                w = vars[idx]
                '''running_mean, running_var = self.vars_bn[bn_idx], self.vars_bn[bn_idx + 1]'''
                running_mean = nn.Parameter(torch.zeros(param[0]), requires_grad=False)
                running_var = nn.Parameter(torch.ones(param[0]), requires_grad=False)
                x = F.batch_norm(x, running_mean, running_var, weight=w, training=bn_training)
                idx += 1
                #bn_idx += 2

            elif name is 'flatten':
                # print(x.shape)
                x = x.view(x.size(0), -1)
            elif name is 'reshape':
                # [b, 8] => [b, 2, 2, 2]
                x = x.view(x.size(0), *param)
            elif name is 'relu':
                x = F.relu(x, inplace=param[0])
            elif name is 'leakyrelu':
                x = F.leaky_relu(x, negative_slope=param[0], inplace=param[1])
            elif name is 'tanh':
                x = F.tanh(x)
            elif name is 'sigmoid':
                x = torch.sigmoid(x)
            elif name is 'upsample':
                x = F.upsample_nearest(x, scale_factor=param[0])
            elif name is 'max_pool2d':
                x = F.max_pool2d(x, param[0], param[1], param[2])
            elif name is 'avg_pool2d':
                x = F.avg_pool2d(x, param[0])

            else:
                raise NotImplementedError

        # make sure variable is used properly
        assert idx == len(vars)
        #assert bn_idx == len(self.vars_bn)
        return x

    def zero_grad(self, vars=None):
        """

        :param vars:
        :return:
        """
        with torch.no_grad():
            if vars is None:
                for p in self.vars:
                    if p.grad is not None:
                        p.grad.zero_()
            else:
                for p in vars:
                    if p.grad is not None:
                        p.grad.zero_()

    def parameters(self):
        """
        override this function since initial parameters will return with a generator.
        :return:
        """
        return self.vars