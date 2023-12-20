import torch
import torch.nn.functional as F
import collections
from generatedata import *
from tool import *

CUDA = torch.cuda.is_available()

def to_np(x):
    return x.detach().cpu().numpy()


def to_tensor(x, **kwargs):
    device = 'cuda' if CUDA else 'cpu'
    return torch.as_tensor(x, device=device, **kwargs)

def data_in_one(inputdata):
    min = np.nanmin(inputdata)
    max = np.nanmax(inputdata)
    outputdata = (inputdata-min)/(max-min)
    return outputdata

def mex_hat(d):
    grid = (np.mgrid[:d, :d] - d // 2) * 1.0
    eucl_grid = (grid ** 2).sum(0) ** 0.5  # euclidean distances
    eucl_grid /= d  # normalize by LIZ length
    return eucl_grid * np.exp(-eucl_grid)  # mex_hat function values

def mex_hat_pitch(d):
    grid = (np.mgrid[:d, :d] - d // 2) * 1.0
    eucl_grid = (grid[0] ** 2) ** 0.5  # euclidean distances
    eucl_grid /= d  # normalize by LIZ length
    return eucl_grid * np.exp(-eucl_grid)  # mex_hat function values


class LateralInhibition_calculate(torch.nn.Module):
    def __init__(self,mex_, l=7, a=0.1, b=0.9):
        mex = {'1':mex_hat(l),
               '2':mex_hat_pitch(l)}
        super().__init__()
        self.len = l
        assert self.len % 2 == 1
        self.a = a
        self.b = b
        self.register_buffer(
            'inhibition_kernel',
            to_tensor(mex[mex_], dtype=torch.float32).view(1, 1, 1, -1))

    def forward(self, x):  # as argument we get max-c map with dimensions 'batch x 1 x n1 x n2'
        assert x.size(1) == 1
        #assert x.size(2) == x.size(3)
        len_ = self.len
        pad = len_ // 2
        batches = x.size(0)
        n1 = x.size(2)
        n2 = x.size(3)

        # unfold x to LIZs for each pixel:
        x_unf = F.unfold(x, (len_, len_), padding=(pad, pad))
        # next line is needed for extend tensor size (from 'batch x kernel x n*n' to 'batch x 1 x kernel x n*n'):
        x_unf = x_unf.view(batches, 1, len_ * len_, n1 * n2)
        x_unf = x_unf.transpose(2, 3)
        # select all middle points in LIZs:
        mid_vals = x.view(x.size(0), 1, n1 * n2, 1)

        average_term = torch.exp(-x_unf.mean(3, keepdim=True)).view(batches, 1, n1, n2)

        differential_term = (self.inhibition_kernel * F.relu(x_unf - mid_vals)
                             ).sum(3, keepdim=True).view(batches, 1, n1, n2)

        suppression_mask = self.a * average_term + self.b * differential_term
        assert x.shape == suppression_mask.shape
        suppression_mask_norm = (suppression_mask ** 2).sum() ** 0.5
        suppression_mask /= suppression_mask_norm
        # because all values are non-negative we can do this:
        filter_ = x > suppression_mask
        suppression_mask = torch.ones_like(x).where(filter_, torch.zeros_like(x))
        return suppression_mask, average_term, differential_term


class ALIDNN(torch.nn.Module):
    def __init__(self, modelA,modelB, l=7, a=0.1, b=0.9,str_inh=0,mean = False,all_channel = True,epocht=10,mex='1'):
        super().__init__()
        self.epocht = epocht
        self.modelA = modelA
        self.modelB = modelB
        self.str_inh = str_inh
        self.mean = mean
        self.all_channel = all_channel
        self.lateralinhibition_calculate = LateralInhibition_calculate(mex, l, a, b)
        if CUDA:
            self.modelA.cuda()
            self.modelB.cuda()
            self.lateralinhibition_calculate.cuda()
        self.relu_layers = [name for name in self.modelA.Encodername]

    def forward(self, img, epoch=0, t=None, show_masks=True):

        while(epoch<self.epocht):
            x, layer_values = self.modelA(img)
            response_attention_map, response_based_sum_c_maps = self.create_attention_map(img, layer_values)
            return x,response_attention_map, response_based_sum_c_maps

        # 1 & 2
        logits, acts, _ = self.modelA.logits_and_activations(to_tensor(img), self.relu_layers,self.str_inh)
        #acts_pre = acts.detach().cpu().numpy()
        for v in acts.values():
            v.retain_grad()

        t = logits.mean()
        y = logits.where(logits>t,torch.zeros_like(logits))
        y.backward(torch.ones_like(y))

        # save them for return operation
        #top100 = logits

        # 3
        suppression_masks = {}
        for layer in self.relu_layers:
            gradient = acts[layer].grad
            # creating max-c map
            if self.all_channel:
                if not self.mean:
                    max_c = gradient.max(1, keepdims=True)[0]
                else:
                    max_c = gradient.mean(1).unsqueeze(1)
                # max-c map is normalized by L2 norm
                max_c_norm = (max_c ** 2).sum() ** 0.5
                max_c /= max_c_norm
                # generating suppression mask through lateral inhibition
                sup_mask, *_ = self.lateralinhibition_calculate(max_c)
                suppression_masks[layer] = sup_mask
            else:
                sup_mask = []
                for l in range(gradient.size()[1]):
                    max_c = gradient[:,l,:,:].unsqueeze(1)
                    max_c_norm = (max_c ** 2).sum() ** 0.5
                    max_c /= max_c_norm
                    # generating suppression mask through lateral inhibition
                    sup_mask_, *_ = self.lateralinhibition_calculate(max_c)
                    if sup_mask==[]:
                        sup_mask = sup_mask_
                    else:
                        sup_mask = torch.cat([sup_mask_,sup_mask],dim=1)
                suppression_masks[layer] = sup_mask

        self.modelA.zero_grad()

        # 4
        logits2, acts2, relu_maps = self.modelB.logits_and_activations(
            to_tensor(img), self.relu_layers,self.str_inh,
            suppression_masks=suppression_masks, save_maps=show_masks)

        #acts_now = acts2.detach().cpu().numpy()
        response_attention_map, response_based_sum_c_maps = self.create_attention_map(img, acts2)

        mask = collections.defaultdict(list)
        for l in self.relu_layers:
            for k in range(0,3):
                _mask = to_tensor(relu_maps[l][k][0]).unsqueeze(0).unsqueeze(0)
                _mask = F.interpolate(_mask, size=(img.shape[1],img.shape[2]), mode='nearest')
                mask[l].append(to_np(_mask).squeeze())
        # 5 & 6 & 7


        #if show_masks:
        #    plt.figure(figsize=(24, 96))
        #    self.show_maps(relu_maps,maskmap)

        r'''
        if draw:
            #plt.figure(figsize=(24, 96))
            #self.print_layers(acts_pre, part=1)
            #self.print_layers(acts_now, part=2)

            # show attention maps
            plt.figure(figsize=(8,10))
            plt.subplot(6, 1, 6)
            librosa.display.specshow(response_attention_map, y_axis='linear')
            plt.colorbar()
            plt.title('sum')
            for i in np.arange(0,5):
                plt.subplot(6, 1, i+1)
                librosa.display.specshow(response_based_sum_c_maps[i], y_axis='linear')
                plt.colorbar()
                plt.title(i+1)
        '''

        return logits2, response_attention_map,response_based_sum_c_maps,mask

    def transform_weight(self):
        for encoderA, encoderB in zip(self.modelA.Encoder, self.modelB.Encoder):
            encoderB[0].weight = encoderA[0].weight
        self.modelB.decoder[0].weight = self.modelA.decoder[0].weight

    def create_attention_map(self, img, acts):
        """
        Function creates response attention map, firstly by summing sum-c maps from all ReLu layers,
          and then normalizing the results by L2 norm.

        """

        response_based_sum_c_maps = []
        for l in self.relu_layers:
            sum_c_map = acts[l][0].sum(0, keepdim=True).unsqueeze(0)
            resized_sum_c_map = F.interpolate(sum_c_map, size=(img.shape[1],img.shape[2]), mode='nearest')   #size -- scale_factor=img_len // map_len
            response_based_sum_c_maps.append(to_np(resized_sum_c_map.squeeze(0).squeeze(0)))

        # creating attention map
        response_attention_map = np.array(response_based_sum_c_maps).sum(0)

        # normalizing attention map
        response_attention_map /= (response_attention_map ** 2).sum() ** 0.5

        return response_attention_map,response_based_sum_c_maps

    def show_maps(self, maps,savepath):
        """
        Function prints three maps for each ReLu layer: suppression mask, ReLu activation map before lateral inhibition
          and ReLu layer activation after lateral inhibition.
        """
        pnum = 1

        for l in self.relu_layers:
            plt.subplot(len(self.relu_layers), 3, pnum)
            pnum += 1
            librosa.display.specshow(maps[l][0][0],  y_axis='linear')
            plt.colorbar(shrink=1.0)
            plt.title(f"suppression mask - {l}",horizontalalignment='right',fontsize='small')

            plt.subplot(len(self.relu_layers), 3, pnum)
            pnum += 1
            plt.imshow(maps[l][1][0], cmap='viridis')
            plt.colorbar(shrink=1.0)
            plt.title(f"normal ReLu - {l}",horizontalalignment='right',fontsize='small')

            plt.subplot(len(self.relu_layers), 3, pnum)
            pnum += 1
            plt.imshow(maps[l][2][0], cmap='viridis')
            plt.colorbar(shrink=1.0)
            plt.title(f"inhibited ReLu - {l}",horizontalalignment='right',fontsize='small')
        plt.savefig(savepath)

    def print_layers(self, acts, part):
        def increase(pnum):
            return pnum + 3 + 1 if pnum % 3 == 0 else pnum + 1

        pnum = (part - 1) * 3 + 1

        for l in self.relu_layers:
            plt.subplot(len(self.relu_layers) * 2, 3, pnum)
            pnum = increase(pnum)
            plt.imshow(acts[l].sum(0), cmap='plasma')
            plt.colorbar(shrink=1.0)
            plt.title(f"sum val {l}")

            plt.subplot(len(self.relu_layers) * 2, 3, pnum)
            pnum = increase(pnum)
            plt.imshow(acts[l].max(0), cmap='plasma')
            plt.colorbar(shrink=1.0)
            plt.title(f"max val {l}")

            plt.subplot(len(self.relu_layers) * 2, 3, pnum)
            pnum = increase(pnum)
            sum_c = acts[l].sum(0)
            norm = (sum_c ** 2).sum() ** 0.5
            plt.imshow(sum_c / norm, cmap='plasma')
            plt.colorbar(shrink=1.0)
            plt.title(f"norm val {l}")

class ALICNN(torch.nn.Module):
    def __init__(self, model, l=7, a=0.1, b=0.9,str_inh=0,band = False, mean = False,all_channel = True,epocht=10):
        super().__init__()
        self.band = band
        self.epocht = epocht
        self.model = model
        self.str_inh = str_inh
        self.mean = mean
        self.all_channel = all_channel
        self.lateralinhibition_calculate = LateralInhibition_calculate('1', l, a, b)
        if CUDA:
            self.model.cuda()
            self.lateralinhibition_calculate.cuda()
            print('net1 in CUDA')

        self.relu_layers = [name for name in self.model.Encodername]
        self.target_layers = [name for name in self.model.target_layers]

    def forward(self, img, show_masks=True):
        # 1 & 2
        logits, acts, _ = self.model.logits_and_activations(to_tensor(img), self.relu_layers,self.str_inh)
        for v in acts.values():
            v.retain_grad()

        t = logits.mean()
        if self.band:
            y = self.band_choose(logits,t)
        else:
            y = logits.where(logits>t,torch.zeros_like(logits))
        y.backward(torch.ones_like(y))

        # save them for return operation
        #top100 = logits

        # 3
        suppression_masks = {}
        for layer,targetlayer in zip(self.relu_layers,self.target_layers):
            gradient = acts[layer].grad
            # creating max-c map
            if self.all_channel:
                if not self.mean:
                    max_c = gradient.max(1, keepdims=True)[0]
                else:
                    max_c = gradient.mean(1).unsqueeze(1)
                # max-c map is normalized by L2 norm
                max_c_norm = (max_c ** 2).sum() ** 0.5
                max_c /= max_c_norm
                # generating suppression mask through lateral inhibition
                sup_mask, *_ = self.lateralinhibition_calculate(max_c)
                suppression_masks[targetlayer] = sup_mask
            else:
                sup_mask = []
                for l in range(gradient.size()[1]):
                    max_c = gradient[:,l,:,:].unsqueeze(1)
                    max_c_norm = (max_c ** 2).sum() ** 0.5
                    max_c /= max_c_norm
                    # generating suppression mask through lateral inhibition
                    sup_mask_, *_ = self.lateralinhibition_calculate(max_c)
                    if sup_mask==[]:
                        sup_mask = sup_mask_
                    else:
                        sup_mask = torch.cat([sup_mask_,sup_mask],dim=1)
                suppression_masks[targetlayer] = sup_mask

        self.model.zero_grad()

        # 4
        # logits2, acts2, relu_maps = self.modelB.logits_and_activations(
        #     to_tensor(img), self.target_layers,self.str_inh,
        #     suppression_masks=suppression_masks, save_maps=show_masks)

        #acts_now = acts2.detach().cpu().numpy()
        # response_attention_map, response_based_sum_c_maps = self.create_attention_map(img, acts2)
        #
        # mask = collections.defaultdict(list)
        # for l in self.relu_layers:
        #     for k in range(0,3):
        #         _mask = to_tensor(relu_maps[l][k][0]).unsqueeze(0).unsqueeze(0)
        #         _mask = F.interpolate(_mask, size=(img.shape[1],img.shape[2]), mode='nearest')
        #         mask[l].append(to_np(_mask).squeeze())
        # 5 & 6 & 7


        #if show_masks:
        #    plt.figure(figsize=(24, 96))
        #    self.show_maps(relu_maps,maskmap)


        # if draw:
        #     #plt.figure(figsize=(24, 96))
        #     #self.print_layers(acts_pre, part=1)
        #     #self.print_layers(acts_now, part=2)
        #
        #     # show attention maps
        #     plt.figure(figsize=(8,10))
        #     plt.subplot(6, 1, 6)
        #     librosa.display.specshow(response_attention_map, y_axis='linear')
        #     plt.colorbar()
        #     plt.title('sum')
        #     for i in np.arange(0,5):
        #         plt.subplot(6, 1, i+1)
        #         librosa.display.specshow(response_based_sum_c_maps[i], y_axis='linear')
        #         plt.colorbar()
        #         plt.title(i+1)


        return logits,suppression_masks

    def band_choose(self,x,t):
        assert len(x.size()) == 3
        for sample in np.arange(0,x.size(0)):
            for f in np.arange(0,x.size(1)):
                sum = x[sample,f,:].sum()
                if sum < t:
                    x[sample,f,:] = 0
        return x



    def load_state(self,state):
        pretrain_dict = torch.load(state)
        net_dict = self.model.state_dict()
        net_dict.update(pretrain_dict)
        self.model.load_state_dict(net_dict)
        print('baseline model state load')

    def create_attention_map(self, img, acts):
        """
        Function creates response attention map, firstly by summing sum-c maps from all ReLu layers,
          and then normalizing the results by L2 norm.

        """

        response_based_sum_c_maps = []
        for l in self.relu_layers:
            sum_c_map = acts[l][0].sum(0, keepdim=True).unsqueeze(0)
            resized_sum_c_map = F.interpolate(sum_c_map, size=(img.shape[1],img.shape[2]), mode='nearest')   #size -- scale_factor=img_len // map_len
            response_based_sum_c_maps.append(to_np(resized_sum_c_map.squeeze(0).squeeze(0)))

        # creating attention map
        response_attention_map = np.array(response_based_sum_c_maps).sum(0)

        # normalizing attention map
        response_attention_map /= (response_attention_map ** 2).sum() ** 0.5

        return response_attention_map,response_based_sum_c_maps

    def show_maps(self, maps,savepath):
        """
        Function prints three maps for each ReLu layer: suppression mask, ReLu activation map before lateral inhibition
          and ReLu layer activation after lateral inhibition.
        """
        pnum = 1

        for l in self.relu_layers:
            plt.subplot(len(self.relu_layers), 3, pnum)
            pnum += 1
            librosa.display.specshow(maps[l][0][0],  y_axis='linear')
            plt.colorbar(shrink=1.0)
            plt.title(f"suppression mask - {l}",horizontalalignment='right',fontsize='small')

            plt.subplot(len(self.relu_layers), 3, pnum)
            pnum += 1
            plt.imshow(maps[l][1][0], cmap='viridis')
            plt.colorbar(shrink=1.0)
            plt.title(f"normal ReLu - {l}",horizontalalignment='right',fontsize='small')

            plt.subplot(len(self.relu_layers), 3, pnum)
            pnum += 1
            plt.imshow(maps[l][2][0], cmap='viridis')
            plt.colorbar(shrink=1.0)
            plt.title(f"inhibited ReLu - {l}",horizontalalignment='right',fontsize='small')
        plt.savefig(savepath)