from LICNN import *
from LI_core import *
from tool import *


class CNN_MASK(nn.Module):
    def __init__(self):
        super().__init__()
        self.encoderchannel = [16, 32, 64, 128, 256]
        self.position = position()
        self.encoder1 = nn.Sequential(nn.Conv2d(1, self.encoderchannel[0], 3, (2, 1), 1),
                                      nn.BatchNorm2d(self.encoderchannel[0]), nn.LeakyReLU())
        self.encoder2 = nn.Sequential(nn.Conv2d(self.encoderchannel[0], self.encoderchannel[1], 3, (2, 1), 1),
                                      nn.BatchNorm2d(self.encoderchannel[1]), nn.LeakyReLU())
        self.encoder3 = nn.Sequential(nn.Conv2d(self.encoderchannel[1], self.encoderchannel[2], 3, (2, 1), 1),
                                      nn.BatchNorm2d(self.encoderchannel[2]), nn.LeakyReLU())
        self.encoder4 = nn.Sequential(nn.Conv2d(self.encoderchannel[2], self.encoderchannel[3], 3, (2, 1), 1),
                                      nn.BatchNorm2d(self.encoderchannel[3]), nn.LeakyReLU())
        self.encoder5 = nn.Sequential(nn.Conv2d(self.encoderchannel[3], self.encoderchannel[4], 3, (2, 1), (0, 1)),
                                      nn.BatchNorm2d(self.encoderchannel[4]), nn.LeakyReLU())

        self.decoder = nn.Sequential(nn.Linear(16*256,513),nn.Sigmoid())

        self.Encoder = [self.encoder1, self.encoder2, self.encoder3, self.encoder4, self.encoder5]
        self.Encodername = ['encoder1', 'encoder2', 'encoder3', 'encoder4', 'encoder5']


    def forward(self,x):
        layer_values = {}

        x = x.unsqueeze(1)

        for name, layer in zip(self.Encodername, self.Encoder):
            x = layer(x)
            layer_values[name] = x

        batch = x.size(0)
        channel = x.size(1)
        f = x.size(2)
        T = x.size(3)
        x = x.transpose(3, 1)
        x = x.contiguous().view(batch, T, channel * f)
        x = self.decoder(x)
        x = x.transpose(2,1)

        return x,layer_values

    def lateralinhibition(self,x,layer,name,suppression_masks,needed_layers,save_maps,layer_values,maps_to_print,str_inh=0):
        if suppression_masks != {}:
            assert all(layer in suppression_masks for layer in needed_layers)
        x = layer(x)

        if name in suppression_masks:
            if save_maps:
                maps_to_print[name].append(to_np(suppression_masks[name].max(1)[0].squeeze()))
                maps_to_print[name].append(to_np(x.squeeze(0).sum(1)))

                # applying suppression mask to relu layer
            if not name == 'encoder5':
                sup_mask_sized_as_x = suppression_masks[name].expand(-1, x.size()[1], -1, -1)
                x = x.where(sup_mask_sized_as_x != 0, str_inh*x)

            if save_maps:
                maps_to_print[name].append(to_np(x.squeeze(0).sum(1)))

        if name in needed_layers:
            layer_values[name] = x
        return x,layer_values,maps_to_print

    def logits_and_activations(self, x, layer_names,str_inh, suppression_masks={} , save_maps=True):
        needed_layers = set(layer_names)
        layer_values = {}
        encodervalue = {}
        maps_to_print = collections.defaultdict(list)
        x = x.unsqueeze(1)

        for encoder, name in zip(self.Encoder, self.Encodername):
            # x = encoder(x)
            x, layer_values, maps_to_print = self.lateralinhibition(x, encoder, name, suppression_masks, needed_layers,
                                                                    save_maps, layer_values, maps_to_print,str_inh)

        batch = x.size(0)
        channel = x.size(1)
        f = x.size(2)
        T = x.size(3)
        x = x.transpose(3, 1)
        x = x.contiguous().view(batch, T, channel * f)
        x = self.decoder(x)
        x = x.transpose(2, 1)
        return x, layer_values, maps_to_print

    def predict(self, x):
        """
        Return predicted class IDs.
        """
        logits = self(x)
        _, prediction = logits.max(1)
        return prediction[0].item()


class baseline_model(nn.Module):
    def __init__(self):
        super().__init__()
        self.encoderchannel = [16,32,64,128,256]
        self.encoder1 = nn.Sequential(nn.Conv2d(1,self.encoderchannel[0],3,(2, 1),1), nn.BatchNorm2d(self.encoderchannel[0]), nn.LeakyReLU())
        self.encoder2 = nn.Sequential(nn.Conv2d(self.encoderchannel[0], self.encoderchannel[1], 3, (2, 1), 1), nn.BatchNorm2d(self.encoderchannel[1]), nn.LeakyReLU())
        self.encoder3 = nn.Sequential(nn.Conv2d(self.encoderchannel[1], self.encoderchannel[2], 3, (2, 1), 1), nn.BatchNorm2d(self.encoderchannel[2]), nn.LeakyReLU())
        self.encoder4 = nn.Sequential(nn.Conv2d(self.encoderchannel[2], self.encoderchannel[3], 3, (2, 1), 1), nn.BatchNorm2d(self.encoderchannel[3]), nn.LeakyReLU())
        self.encoder5 = nn.Sequential(nn.Conv2d(self.encoderchannel[3], self.encoderchannel[4], 3, (2, 1), (0,1)), nn.BatchNorm2d(self.encoderchannel[4]), nn.LeakyReLU())

        self.lstm1 = nn.LSTM(16*256, 16*256,1)
        self.lstm2 = nn.LSTM(16*256, 16*256,1)

        self.decoder1 = nn.Sequential(nn.ConvTranspose2d(self.encoderchannel[4]*2, self.encoderchannel[3], 3, (2, 1), (0,1)), nn.BatchNorm2d(self.encoderchannel[3]), nn.LeakyReLU())
        self.decoder2 = nn.Sequential(nn.ConvTranspose2d(self.encoderchannel[3]*2, self.encoderchannel[2], 3, (2, 1), 1), nn.BatchNorm2d(self.encoderchannel[2]), nn.LeakyReLU())
        self.decoder3 = nn.Sequential(nn.ConvTranspose2d(self.encoderchannel[2]*2, self.encoderchannel[1], 3, (2, 1), 1), nn.BatchNorm2d(self.encoderchannel[1]), nn.LeakyReLU())
        self.decoder4 = nn.Sequential(nn.ConvTranspose2d(self.encoderchannel[1]*2, self.encoderchannel[0], 3, (2, 1), 1), nn.BatchNorm2d(self.encoderchannel[0]), nn.LeakyReLU())
        self.decoder5 = nn.Sequential(nn.ConvTranspose2d(self.encoderchannel[0]*2, 1, 3, (2, 1), 1), nn.BatchNorm2d(1), nn.Sigmoid())

        self.Encoder = [self.encoder1,self.encoder2,self.encoder3,self.encoder4,self.encoder5]
        self.Encodername = ['encoder1','encoder2','encoder3','encoder4','encoder5']
        self.Lstm = [self.lstm1,self.lstm2]
        self.Decoder = [self.decoder1,self.decoder2,self.decoder3,self.decoder4,self.decoder5]
        self.Decodername = ['decoder1', 'decoder2', 'decoder3', 'decoder4', 'decoder5']
        self.feature_names = ['encoder1','encoder2','encoder3','encoder4','encoder5','decoder1', 'decoder2', 'decoder3', 'decoder4', 'decoder5']
        # self.target_layers = ['decoder5', 'decoder4', 'decoder3', 'decoder2', 'decoder1']
        self.target_layers = ['encoder1', 'encoder2', 'encoder3', 'encoder4', 'encoder5']

    def forward(self, x):
        encodervalue = {}
        i = -1
        tf = x
        x = x.unsqueeze(1)

        for encoder in self.Encoder:
            i += 1
            x = encoder(x)
            encodervalue[i] = x

        batch = x.size(0)
        channel = x.size(1)
        f = x.size(2)
        T = x.size(3)
        x = x.transpose(3,1)
        x = x.transpose(1,0)
        x = x.contiguous().view(T,batch,channel*f)

        for lstm in self.Lstm:
            x = lstm(x)
            x = x[0]

        x = x.transpose(1,0)
        x = x.view(batch,T,f,channel)
        x = x.transpose(3,1)

        for decoder in self.Decoder:
            x = torch.cat((x,encodervalue[i]),dim=1)
            x = decoder(x)
            i -= 1

        x = x.squeeze()
        x = x*tf

        return x

    def lateralinhibition_mask(self,x,layer,name,suppression_masks,needed_layers,save_maps,layer_values,maps_to_print):
        if suppression_masks != {}:
            assert all(layer in suppression_masks for layer in needed_layers)
        x = layer(x)

        if name in suppression_masks:
            if save_maps:
                    maps_to_print[name].append(to_np(suppression_masks[name].squeeze()))
                    maps_to_print[name].append(to_np(x.squeeze(0).sum(0)))

                # applying suppression mask to relu layer
            if not name == 'decoder5':
                    sup_mask_sized_as_x = suppression_masks[name].expand(-1, x.size()[1], -1, -1)
                    x = x.where(sup_mask_sized_as_x != 0, torch.zeros_like(x))

            if save_maps:
                    maps_to_print[name].append(to_np(x.squeeze(0).sum(0)))

        if name in needed_layers:
            layer_values[name] = x
        return x,layer_values,maps_to_print

    def logits_and_activations(self, x, layer_names, as_dict=True, suppression_masks={}, save_maps=False):
        tf = x
        needed_layers = set(layer_names)
        layer_values = {}
        encodervalue = {}
        maps_to_print = collections.defaultdict(list)
        i = -1
        x = x.unsqueeze(1)

        for encoder,name in zip(self.Encoder,self.Encodername):
            i += 1
            #x = encoder(x)
            x,layer_values_e,maps_to_print = self.lateralinhibition_mask(x, encoder, name, suppression_masks, needed_layers, save_maps, layer_values,maps_to_print)
            encodervalue[i] = x

        batch = x.size(0)
        channel = x.size(1)
        f = x.size(2)
        T = x.size(3)
        x = x.transpose(3, 1)
        x = x.transpose(1, 0)
        x = x.contiguous().view(T, batch, channel * f)

        for lstm in self.Lstm:
            x = lstm(x)
            x = x[0]

        x = x.transpose(1, 0)
        x = x.view(batch, T, f, channel)
        x = x.transpose(3, 1)

        for decoder, name in zip(self.Decoder, self.Decodername):
            x = torch.cat((x, encodervalue[i]), dim=1)
            #x = decoder(x)
            x,layer_values_d,maps_to_print = self.lateralinhibition_mask(x, decoder, name, suppression_masks, needed_layers, save_maps, layer_values,maps_to_print)
            i -= 1
        '''
        if not as_dict:
            layer_values = [layer_values[n] for n in layer_names]
        '''
        x = x.squeeze()
        x = x * tf
        layer_values = dict(layer_values_e,**layer_values_d)
        return x,layer_values,maps_to_print

class enhance_model(nn.Module):
    def __init__(self):
        super().__init__()
        self.decoder = True
        self.encoder = False
        self.encoderchannel = [16,32,64,128,256]
        self.encoder1 = nn.Sequential(nn.Conv2d(1,self.encoderchannel[0],3,(2, 1),1), nn.BatchNorm2d(self.encoderchannel[0]), nn.LeakyReLU())
        self.encoder2 = nn.Sequential(nn.Conv2d(self.encoderchannel[0], self.encoderchannel[1], 3, (2, 1), 1), nn.BatchNorm2d(self.encoderchannel[1]), nn.LeakyReLU())
        self.encoder3 = nn.Sequential(nn.Conv2d(self.encoderchannel[1], self.encoderchannel[2], 3, (2, 1), 1), nn.BatchNorm2d(self.encoderchannel[2]), nn.LeakyReLU())
        self.encoder4 = nn.Sequential(nn.Conv2d(self.encoderchannel[2], self.encoderchannel[3], 3, (2, 1), 1), nn.BatchNorm2d(self.encoderchannel[3]), nn.LeakyReLU())
        self.encoder5 = nn.Sequential(nn.Conv2d(self.encoderchannel[3], self.encoderchannel[4], 3, (2, 1), (0,1)), nn.BatchNorm2d(self.encoderchannel[4]), nn.LeakyReLU())

        self.lstm1 = nn.LSTM(16*256, 16*256,1)
        self.lstm2 = nn.LSTM(16*256, 16*256,1)

        self.decoder1 = nn.Sequential(nn.ConvTranspose2d(self.encoderchannel[4]*2+1, self.encoderchannel[3], 3, (2, 1), (0,1)), nn.BatchNorm2d(self.encoderchannel[3]), nn.LeakyReLU())
        self.decoder2 = nn.Sequential(nn.ConvTranspose2d(self.encoderchannel[3]*2+1, self.encoderchannel[2], 3, (2, 1), 1), nn.BatchNorm2d(self.encoderchannel[2]), nn.LeakyReLU())
        self.decoder3 = nn.Sequential(nn.ConvTranspose2d(self.encoderchannel[2]*2+1, self.encoderchannel[1], 3, (2, 1), 1), nn.BatchNorm2d(self.encoderchannel[1]), nn.LeakyReLU())
        self.decoder4 = nn.Sequential(nn.ConvTranspose2d(self.encoderchannel[1]*2+1, self.encoderchannel[0], 3, (2, 1), 1), nn.BatchNorm2d(self.encoderchannel[0]), nn.LeakyReLU())
        self.decoder5 = nn.Sequential(nn.ConvTranspose2d(self.encoderchannel[0]*2+1, 1, 3, (2, 1), 1), nn.BatchNorm2d(1), nn.Sigmoid())

        self.Encoder = [self.encoder1,self.encoder2,self.encoder3,self.encoder4,self.encoder5]
        self.Encodername = ['encoder1','encoder2','encoder3','encoder4','encoder5']
        self.Lstm = [self.lstm1,self.lstm2]
        self.Decoder = [self.decoder1,self.decoder2,self.decoder3,self.decoder4,self.decoder5]
        # self.Decodername = ['decoder1', 'decoder2', 'decoder3', 'decoder4', 'decoder5']
        self.Decodername = ['encoder5','encoder4','encoder3','encoder2','encoder1']
        self.feature_names = ['encoder1','encoder2','encoder3','encoder4','encoder5','decoder1', 'decoder2', 'decoder3', 'decoder4', 'decoder5']

        if self.decoder:
            self.target_layers = ['decoder5', 'decoder4', 'decoder3', 'decoder2', 'decoder1']
        if self.encoder:
            self.target_layers = ['encoder1', 'encoder2', 'encoder3', 'encoder4', 'encoder5']

    def forward(self, x, mask):
        encodervalue = {}
        i = -1
        tf = x
        x = x.unsqueeze(1)

        for encoder, name in zip(self.Encoder,self.Encodername):
            i += 1
            x = encoder(x)
            encodervalue[i] = x
            if self.encoder:
                x = torch.cat((x, mask[name]), dim=1)

        batch = x.size(0)
        channel = x.size(1)
        f = x.size(2)
        T = x.size(3)
        x = x.transpose(3,1)
        x = x.transpose(1,0)
        x = x.contiguous().view(T,batch,channel*f)

        for lstm in self.Lstm:
            x = lstm(x)
            x = x[0]

        x = x.transpose(1,0)
        x = x.view(batch,T,f,channel)
        x = x.transpose(3,1)


        for decoder,name in zip(self.Decoder,self.Decodername):
            x = torch.cat((x,encodervalue[i]),dim=1)
            if self.decoder:
                x = torch.cat((x, mask[name]), dim=1)
            x = decoder(x)
            i -= 1


        x = x.squeeze()
        x = x*tf

        return x

class Unet_transformer(nn.Module):
    def __init__(self):
        super().__init__()
        self.encoderchannel = [16,32,64,128,256]
        self.encoder1 = nn.Sequential(nn.Conv2d(1,self.encoderchannel[0],3,(2, 1),1), nn.BatchNorm2d(self.encoderchannel[0]), nn.LeakyReLU())
        self.encoder2 = nn.Sequential(nn.Conv2d(self.encoderchannel[0], self.encoderchannel[1], 3, (2, 1), 1), nn.BatchNorm2d(self.encoderchannel[1]), nn.LeakyReLU())
        self.encoder3 = nn.Sequential(nn.Conv2d(self.encoderchannel[1], self.encoderchannel[2], 3, (2, 1), 1), nn.BatchNorm2d(self.encoderchannel[2]), nn.LeakyReLU())
        self.encoder4 = nn.Sequential(nn.Conv2d(self.encoderchannel[2], self.encoderchannel[3], 3, (2, 1), 1), nn.BatchNorm2d(self.encoderchannel[3]), nn.LeakyReLU())
        self.encoder5 = nn.Sequential(nn.Conv2d(self.encoderchannel[3], self.encoderchannel[4], 3, (2, 1), (0,1)), nn.BatchNorm2d(self.encoderchannel[4]), nn.LeakyReLU())

        self.q_linear = nn.Linear(16*256,256)
        self.k_linear = nn.Linear(16*256,256)
        self.v_linear = nn.Linear(16*256,256)
        self.soft = nn.Softmax(dim=-1)
        self.re_linear = nn.Linear(256,16*256)

        self.decoder1 = nn.Sequential(nn.ConvTranspose2d(self.encoderchannel[4]*2, self.encoderchannel[3], 3, (2, 1), (0,1)), nn.BatchNorm2d(self.encoderchannel[3]), nn.LeakyReLU())
        self.decoder2 = nn.Sequential(nn.ConvTranspose2d(self.encoderchannel[3]*2, self.encoderchannel[2], 3, (2, 1), 1), nn.BatchNorm2d(self.encoderchannel[2]), nn.LeakyReLU())
        self.decoder3 = nn.Sequential(nn.ConvTranspose2d(self.encoderchannel[2]*2, self.encoderchannel[1], 3, (2, 1), 1), nn.BatchNorm2d(self.encoderchannel[1]), nn.LeakyReLU())
        self.decoder4 = nn.Sequential(nn.ConvTranspose2d(self.encoderchannel[1]*2, self.encoderchannel[0], 3, (2, 1), 1), nn.BatchNorm2d(self.encoderchannel[0]), nn.LeakyReLU())
        self.decoder5 = nn.Sequential(nn.ConvTranspose2d(self.encoderchannel[0]*2, 1, 3, (2, 1), 1), nn.BatchNorm2d(1), nn.Sigmoid())

        self.Encoder = [self.encoder1,self.encoder2,self.encoder3,self.encoder4,self.encoder5]
        self.Encodername = ['encoder1','encoder2','encoder3','encoder4','encoder5']
        self.Decoder = [self.decoder1,self.decoder2,self.decoder3,self.decoder4,self.decoder5]
        self.Decodername = ['decoder1', 'decoder2', 'decoder3', 'decoder4', 'decoder5']
        self.feature_names = ['encoder1','encoder2','encoder3','encoder4','encoder5','decoder1', 'decoder2', 'decoder3', 'decoder4', 'decoder5']

    def forward(self, x):
        encodervalue = {}
        i = -1
        tf = x
        x = x.unsqueeze(1)

        for encoder in self.Encoder:
            i += 1
            x = encoder(x)
            encodervalue[i] = x

        batch = x.size(0)
        channel = x.size(1)
        f = x.size(2)
        T = x.size(3)
        x = x.transpose(3,1)
        x = x.contiguous().view(batch,T,channel*f)

        x = self.transformer(x)

        x = x.view(batch,T,f,channel)
        x = x.transpose(3,1)

        for decoder in self.Decoder:
            x = torch.cat((x,encodervalue[i]),dim=1)
            x = decoder(x)
            i -= 1

        x = x.squeeze()
        x = x*tf

        return x

    def transformer(self,x):
        q = self.q_linear(x)
        k = self.k_linear(x)
        v = self.v_linear(x)
        qk = self.soft(torch.matmul(q.transpose(-1,-2),k)/23)
        return self.re_linear(torch.matmul(v,qk))

    def lateralinhibition(self,x,layer,name,suppression_masks,needed_layers,save_maps,layer_values,maps_to_print):
        if suppression_masks != {}:
            assert all(layer in suppression_masks for layer in needed_layers)
        x = layer(x)

        if name in suppression_masks:
            if save_maps:
                    maps_to_print[name].append(to_np(suppression_masks[name].squeeze()))
                    maps_to_print[name].append(to_np(x.squeeze(0).sum(0)))

                # applying suppression mask to relu layer
            if not name == 'decoder5':
                    sup_mask_sized_as_x = suppression_masks[name].expand(-1, x.size()[1], -1, -1)
                    x = x.where(sup_mask_sized_as_x != 0, torch.zeros_like(x))

            if save_maps:
                    maps_to_print[name].append(to_np(x.squeeze(0).sum(0)))

        if name in needed_layers:
            layer_values[name] = x
        return x,layer_values,maps_to_print

    def logits_and_activations(self, x, layer_names, as_dict=False, suppression_masks={}, save_maps=False):
        needed_layers = set(layer_names)
        layer_values = {}
        encodervalue = {}
        maps_to_print = collections.defaultdict(list)
        i = -1
        x = x.unsqueeze(1)

        for encoder,name in zip(self.Encoder,self.Encodername):
            i += 1
            #x = encoder(x)
            x,layer_values,maps_to_print = self.lateralinhibition(x, encoder, name, suppression_masks, needed_layers, save_maps, layer_values,maps_to_print)
            encodervalue[i] = x

        batch = x.size(0)
        channel = x.size(1)
        f = x.size(2)
        T = x.size(3)
        x = x.transpose(3, 1)
        x = x.transpose(1, 0)
        x = x.contiguous().view(T, batch, channel * f)

        for lstm in self.Lstm:
            x = lstm(x)
            x = x[0]

        x = x.transpose(1, 0)
        x = x.view(batch, T, f, channel)
        x = x.transpose(3, 1)

        for decoder, name in zip(self.Decoder, self.Decodername):
            x = torch.cat((x, encodervalue[i]), dim=1)
            #x = decoder(x)
            x,layer_values,maps_to_print = self.lateralinhibition(x, decoder, name, suppression_masks, needed_layers, save_maps, layer_values,maps_to_print)
            i -= 1

        if not as_dict:
            layer_values = [layer_values[n] for n in layer_names]

        x = x.squeeze()

        return x,layer_values,maps_to_print

class LateralInhibition(torch.nn.Module):
    def __init__(self, mex_, l=7, a=0.1, b=0.9):
        mex = {'1': mex_hat(l),
               '2': mex_hat_pitch(l)}
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
        # assert x.size(2) == x.size(3)
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