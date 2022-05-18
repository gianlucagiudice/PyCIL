import copy
import torch
from torch import nn
from convs.cifar_resnet import resnet32
from convs.resnet import resnet18, resnet34, resnet50
from convs.ucir_cifar_resnet import resnet32 as cosine_resnet32
from convs.ucir_resnet import resnet18 as cosine_resnet18
from convs.ucir_resnet import resnet34 as cosine_resnet34
from convs.ucir_resnet import resnet50 as cosine_resnet50
from convs.linears import SimpleLinear, SplitCosineLinear, CosineLinear
import logging

def get_convnet(convnet_type, pretrained=False):
    name = convnet_type.lower()
    if name == 'resnet32':
        return resnet32()
    elif name == 'resnet18':
        return resnet18(pretrained=pretrained)
    elif name == 'resnet34':
        return resnet34(pretrained=pretrained)
    elif name == 'resnet50':
        return resnet50(pretrained=pretrained)
    elif name == 'cosine_resnet18':
        return cosine_resnet18(pretrained=pretrained)
    elif name == 'cosine_resnet32':
        return cosine_resnet32()
    elif name == 'cosine_resnet34':
        return cosine_resnet34(pretrained=pretrained)
    elif name == 'cosine_resnet50':
        return cosine_resnet50(pretrained=pretrained)
    else:
        raise NotImplementedError('Unknown type {}'.format(convnet_type))


class BaseNet(nn.Module):

    def __init__(self, convnet_type, pretrained):
        super(BaseNet, self).__init__()

        self.convnet = get_convnet(convnet_type, pretrained)
        self.fc = None

    @property
    def feature_dim(self):
        return self.convnet.out_dim

    def extract_vector(self, x):
        return self.convnet(x)['features']

    def forward(self, x):
        x = self.convnet(x)
        out = self.fc(x['features'])
        '''
        {
            'fmaps': [x_1, x_2, ..., x_n],
            'features': features
            'logits': logits
        }
        '''
        out.update(x)

        return out

    def update_fc(self, nb_classes):
        pass

    def generate_fc(self, in_dim, out_dim):
        pass

    def copy(self):
        return copy.deepcopy(self)

    def freeze(self):
        for param in self.parameters():
            param.requires_grad = False
        self.eval()

        return self


class IncrementalNet(BaseNet):

    def __init__(self, convnet_type, pretrained, gradcam=False):
        super().__init__(convnet_type, pretrained)
        self.gradcam = gradcam
        if hasattr(self, 'gradcam') and self.gradcam:
            self._gradcam_hooks = [None, None]
            self.set_gradcam_hook()

    def update_fc(self, nb_classes):
        fc = self.generate_fc(self.feature_dim, nb_classes)
        if self.fc is not None:
            nb_output = self.fc.out_features
            weight = copy.deepcopy(self.fc.weight.data)
            bias = copy.deepcopy(self.fc.bias.data)
            fc.weight.data[:nb_output] = weight
            fc.bias.data[:nb_output] = bias

        del self.fc
        self.fc = fc

    def weight_align(self, increment):
        weights = self.fc.weight.data
        newnorm = (torch.norm(weights[-increment:, :], p=2, dim=1))
        oldnorm = (torch.norm(weights[:-increment, :], p=2, dim=1))
        meannew = torch.mean(newnorm)
        meanold = torch.mean(oldnorm)
        gamma = meanold / meannew
        print('alignweights,gamma=', gamma)
        self.fc.weight.data[-increment:, :] *= gamma

    def generate_fc(self, in_dim, out_dim):
        fc = SimpleLinear(in_dim, out_dim)

        return fc

    def forward(self, x):
        x = self.convnet(x)
        out = self.fc(x['features'])
        out.update(x)
        if hasattr(self, 'gradcam') and self.gradcam:
            out['gradcam_gradients'] = self._gradcam_gradients
            out['gradcam_activations'] = self._gradcam_activations

        return out

    def unset_gradcam_hook(self):
        self._gradcam_hooks[0].remove()
        self._gradcam_hooks[1].remove()
        self._gradcam_hooks[0] = None
        self._gradcam_hooks[1] = None
        self._gradcam_gradients, self._gradcam_activations = [None], [None]

    def set_gradcam_hook(self):
        self._gradcam_gradients, self._gradcam_activations = [None], [None]

        def backward_hook(module, grad_input, grad_output):
            self._gradcam_gradients[0] = grad_output[0]
            return None

        def forward_hook(module, input, output):
            self._gradcam_activations[0] = output
            return None

        self._gradcam_hooks[0] = self.convnet.last_conv.register_backward_hook(backward_hook)
        self._gradcam_hooks[1] = self.convnet.last_conv.register_forward_hook(forward_hook)


class CosineIncrementalNet(BaseNet):

    def __init__(self, convnet_type, pretrained, nb_proxy=1):
        super().__init__(convnet_type, pretrained)
        self.nb_proxy = nb_proxy

    def update_fc(self, nb_classes, task_num):
        fc = self.generate_fc(self.feature_dim, nb_classes)
        if self.fc is not None:
            if task_num == 1:
                fc.fc1.weight.data = self.fc.weight.data
                fc.sigma.data = self.fc.sigma.data
            else:
                prev_out_features1 = self.fc.fc1.out_features
                fc.fc1.weight.data[:prev_out_features1] = self.fc.fc1.weight.data
                fc.fc1.weight.data[prev_out_features1:] = self.fc.fc2.weight.data
                fc.sigma.data = self.fc.sigma.data

        del self.fc
        self.fc = fc

    def generate_fc(self, in_dim, out_dim):
        if self.fc is None:
            fc = CosineLinear(in_dim, out_dim, self.nb_proxy, to_reduce=True)
        else:
            prev_out_features = self.fc.out_features // self.nb_proxy
            # prev_out_features = self.fc.out_features
            fc = SplitCosineLinear(in_dim, prev_out_features, out_dim - prev_out_features, self.nb_proxy)

        return fc


class BiasLayer(nn.Module):
    def __init__(self):
        super(BiasLayer, self).__init__()
        self.alpha = nn.Parameter(torch.ones(1, requires_grad=True))
        self.beta = nn.Parameter(torch.zeros(1, requires_grad=True))

    def forward(self, x, low_range, high_range):
        ret_x = x.clone()
        ret_x[:, low_range:high_range] = self.alpha * x[:, low_range:high_range] + self.beta
        return ret_x

    def get_params(self):
        return (self.alpha.item(), self.beta.item())


class IncrementalNetWithBias(BaseNet):
    def __init__(self, convnet_type, pretrained, bias_correction=False):
        super().__init__(convnet_type, pretrained)

        # Bias layer
        self.bias_correction = bias_correction
        self.bias_layers = nn.ModuleList([])
        self.task_sizes = []

    def forward(self, x):
        x = self.convnet(x)
        out = self.fc(x['features'])
        if self.bias_correction:
            logits = out['logits']
            for i, layer in enumerate(self.bias_layers):
                logits = layer(logits, sum(self.task_sizes[:i]), sum(self.task_sizes[:i + 1]))
            out['logits'] = logits

        out.update(x)

        return out

    def update_fc(self, nb_classes):
        fc = self.generate_fc(self.feature_dim, nb_classes)
        if self.fc is not None:
            nb_output = self.fc.out_features
            weight = copy.deepcopy(self.fc.weight.data)
            bias = copy.deepcopy(self.fc.bias.data)
            fc.weight.data[:nb_output] = weight
            fc.bias.data[:nb_output] = bias

        del self.fc
        self.fc = fc

        new_task_size = nb_classes - sum(self.task_sizes)
        self.task_sizes.append(new_task_size)
        self.bias_layers.append(BiasLayer())

    def generate_fc(self, in_dim, out_dim):
        fc = SimpleLinear(in_dim, out_dim)

        return fc

    def get_bias_params(self):
        params = []
        for layer in self.bias_layers:
            params.append(layer.get_params())

        return params

    def unfreeze(self):
        for param in self.parameters():
            param.requires_grad = True


s_max = 100000


class DERNet(nn.Module):
    def __init__(self, convnet_type, pretrained, dropout=None):
        super(DERNet, self).__init__()
        self.old_state_dict = None
        self.s = None
        self.convnet_type = convnet_type
        self.convnets = nn.ModuleList()
        self.e = []
        self.pretrained = pretrained
        self.out_dim = None
        self.fc = None
        self.aux_fc = None
        self.task_sizes = []
        self.dropout = nn.Dropout(p=dropout) if dropout else None
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.pruned = False

    @property
    def feature_dim(self):
        if self.out_dim is None:
            return 0
        return self.out_dim * len(self.convnets)

    def extract_vector(self, x):
        features = [convnet(x)['features'] for convnet in self.convnets]
        features = torch.cat(features, 1)
        return features

    def forward(self, x, b=None, B=None):
        if self.training:
            s = (1 / s_max) + (s_max - (1 / s_max)) * (b / (B-1))
        else:
            s = s_max
        self.s = torch.tensor(s, requires_grad=False)

        features = [self.masked_features(x, self.convnets[-1])]
        if len(self.convnets) > 1:
            features = [convnet(x)['features'] for convnet in self.convnets[:-1]] + features

        features = torch.cat(features, 1)

        # Dropout
        if self.dropout:
            features = self.dropout(features)

        # Actual output
        out = self.fc(features)  # {logics: self.fc(features)}

        aux_logits = self.aux_fc(features[:, -self.out_dim:])["logits"]

        n = 0.
        d = 0.
        for l in range(1, len(self.e)):
            m_l = torch.sigmoid(self.s * self.e[l])
            m_l_prev = torch.sigmoid(self.s * self.e[l-1])
            norm_l = torch.norm(m_l, p=1)
            if l-1 == 0:
                norm_l_prev = 3
            else:
                norm_l_prev = torch.norm(m_l_prev, p=1)
            kernel_size = self.convolutions[l].kernel_size[0]
            n += kernel_size * norm_l * norm_l_prev
            d += kernel_size * self.convolutions[l].weight.shape[1] * self.convolutions[l-1].weight.shape[1]
        sparsity_loss = n / d
        '''
        print(f'n: {n.detach().clone().item()}')
        print(f'd: {n.detach().clone().item()}')
        print(f'sparsity: {sparsity_loss.detach().clone().item()}')
        '''
        out.update({"aux_logits": aux_logits, "features": features, "sparsity_loss": sparsity_loss})
        return out

    @torch.no_grad()
    def prune_last_cnn(self, thd=0.00001):
        self.pruned = True
        self.old_state_dict = self.convnets[-1].state_dict()

        for i in range(1, len(self.convolutions) - 1, 2):
            conv, bns, e, e_next = self.convolutions[i], self.batch_norms[i], self.e[i], self.e[i+1]
            mask = torch.unsqueeze(torch.sigmoid(e.detach().clone() * s_max), -1)
            mask_next = torch.unsqueeze(torch.sigmoid(e_next.detach().clone() * s_max), -1)
            next_conv = self.convolutions[i + 1]
            # Binarize mask
            new_weight_ids = mask.flatten() > thd
            print(f'Number of channels after pruning (layer {i}): {new_weight_ids.sum().item()}')
            # Prune parameters
            conv_weight = conv.weight.detach().clone() * mask
            next_conv_weight = next_conv.weight.detach().clone() * mask_next
            conv.weight.data = conv_weight[new_weight_ids, :]
            next_conv.weight.data = next_conv_weight[:, new_weight_ids, :]
            #next_next_conv.weight.data = next_next_conv.weight.detach().clone()[:, new_weight_ids, :]

            bns.bias.data = bns.bias.detach().clone()[new_weight_ids]
            bns.weight.data = bns.weight.detach().clone()[new_weight_ids]
            bns.running_mean = bns.running_mean.detach().clone()[new_weight_ids]
            bns.running_var = bns.running_var.detach().clone()[new_weight_ids]

    def masked_features(self, x, convnet):
        e = self.e
        if not self.training:
            e = []
            for e_l in self.e:
                e.append(e_l.detach().clone())

        s = self.s.detach().clone()

        l = 0

        x = convnet.conv1(x)

        # TODO: Change this
        '''
        if self.training:
            x *= torch.sigmoid(masks[l] * s)
        '''
        l += 1
        x = convnet.bn1(x)
        x = convnet.maxpool(x)


        # Layer 1
        all_blocks = [convnet.layer1._modules, convnet.layer2._modules, convnet.layer3._modules, convnet.layer4._modules]
        for blocks in all_blocks:
            for block in blocks.values():
                identity = x
                # 1
                out = block.conv1(x)
                if not self.pruned:
                    out *= torch.sigmoid(e[l] * s)
                l += 1
                out = block.bn1(out)
                out = block.relu(out)
                # 2
                out = block.conv2(out)
                if not self.pruned:
                    out *= torch.sigmoid(e[l] * s)
                l += 1
                out = block.bn2(out)

                if block.downsample is not None:
                    identity = block.downsample(x)

                out += identity
                out = block.relu(out)
                x = out

        pooled = convnet.avgpool(x)
        features = torch.flatten(pooled, 1)

        return features

    def update_fc(self, nb_classes):
        if len(self.convnets) == 0:
            self.convnets.append(get_convnet(self.convnet_type, pretrained=self.pretrained))
        else:
            self.convnets.append(get_convnet(self.convnet_type, pretrained=self.pretrained))
            self.convnets[-1].load_state_dict(self.old_state_dict)

        if self.out_dim is None:
            self.out_dim = self.convnets[-1].out_dim
        fc = self.generate_fc(self.feature_dim, nb_classes)
        if self.fc is not None:
            nb_output = self.fc.out_features
            weight = copy.deepcopy(self.fc.weight.data)
            bias = copy.deepcopy(self.fc.bias.data)
            fc.weight.data[:nb_output, :self.feature_dim - self.out_dim] = weight
            fc.bias.data[:nb_output] = bias

        # Create mask
        l = 0
        convnet = self.convnets[-1]
        self.convolutions = [convnet.conv1]
        self.batch_norms = [convnet.bn1]
        new_masks = []
        mask = torch.rand((convnet.conv1.out_channels, 1, 1), requires_grad=True, device=self.device)
        new_masks.append(mask)
        all_blocks = [convnet.layer1._modules, convnet.layer2._modules, convnet.layer3._modules,
                      convnet.layer4._modules]
        for blocks in all_blocks:
            for block in blocks.values():
                l += 1
                self.convolutions.append(block.conv1)
                self.batch_norms.append(block.bn1)

                mask = torch.rand((block.conv1.out_channels, 1, 1), requires_grad=True, device=self.device)
                new_masks.append(mask)

                l += 1
                self.convolutions.append(block.conv2)
                self.batch_norms.append(block.bn2)

                mask = torch.rand((block.conv2.out_channels, 1, 1), requires_grad=True, device=self.device)
                new_masks.append(mask)
        self.e = new_masks
        self.pruned = False

        # Register hook
        for l, m in enumerate(self.e):
            m.register_hook(self.compensate_gradiant(l))

        del self.fc
        self.fc = fc

        new_task_size = nb_classes - sum(self.task_sizes)
        self.task_sizes.append(new_task_size)

        self.aux_fc = self.generate_fc(self.out_dim, new_task_size + 1)

    def compensate_gradiant(self, l):
        return lambda inputs: self.compensate_gradient_layer(inputs, l)

    def compensate_gradient_layer(self, inputs, l_index):
        with torch.no_grad():
            s = torch.nan_to_num(self.s.detach().clone())
            e = torch.nan_to_num(self.e[l_index].detach().clone())
            grad = torch.nan_to_num(inputs.detach().clone())
            n = torch.sigmoid(e) * (1 - torch.sigmoid(e))
            d = (s * (torch.sigmoid(s * e))) * (1 - torch.sigmoid(s * e))
            res = (n/d) * grad
        return torch.nan_to_num(res)

    def get_cnn_layers(self, cnn):
        return [l for l in cnn.modules() if isinstance(l, torch.nn.Conv2d)]

    def generate_fc(self, in_dim, out_dim):
        fc = SimpleLinear(in_dim, out_dim)

        return fc

    def copy(self):
        return copy.deepcopy(self)

    def freeze(self):
        for param in self.parameters():
            param.requires_grad = False
        self.eval()

        return self

    def freeze_conv(self):
        for param in self.convnets.parameters():
            param.requires_grad = False
        self.convnets.eval()

    def weight_align(self, increment):
        weights = self.fc.weight.data
        newnorm = (torch.norm(weights[-increment:, :], p=2, dim=1))
        oldnorm = (torch.norm(weights[:-increment, :], p=2, dim=1))
        meannew = torch.mean(newnorm)
        meanold = torch.mean(oldnorm)
        gamma = meanold / meannew
        print('alignweights,gamma=', gamma)
        self.fc.weight.data[-increment:, :] *= gamma


class SimpleCosineIncrementalNet(BaseNet):

    def __init__(self, convnet_type, pretrained):
        super().__init__(convnet_type, pretrained)

    def update_fc(self, nb_classes, nextperiod_initialization):
        fc = self.generate_fc(self.feature_dim, nb_classes).cuda()
        if self.fc is not None:
            nb_output = self.fc.out_features
            weight = copy.deepcopy(self.fc.weight.data)
            fc.sigma.data = self.fc.sigma.data
            if nextperiod_initialization is not None:
                weight = torch.cat([weight, nextperiod_initialization])
            fc.weight = nn.Parameter(weight)
        del self.fc
        self.fc = fc

    def generate_fc(self, in_dim, out_dim):
        fc = CosineLinear(in_dim, out_dim)
        return fc
