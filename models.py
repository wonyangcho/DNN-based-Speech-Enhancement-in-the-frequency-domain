import torch
import torch.nn as nn
import torch.nn.functional as F
from tools_for_model import ConvSTFT, ConviSTFT, \
    ComplexConv2d, ComplexConvTranspose2d, NavieComplexLSTM, complex_cat, ComplexBatchNorm, \
    RealConv2d, RealConvTranspose2d, \
    BaseModel, SequenceModel
import config as cfg
from tools_for_loss import sdr, si_sdr, si_snr, get_array_lms_loss, get_array_pmsqe_loss


#######################################################################
#                         complex network                             #
#######################################################################
class DCCRN(nn.Module):

    def __init__(
            self,
            rnn_layers=cfg.rnn_layers,
            rnn_units=cfg.rnn_units,
            win_len=cfg.win_len,
            win_inc=cfg.win_inc,
            fft_len=cfg.fft_len,
            win_type=cfg.window,
            masking_mode=cfg.masking_mode,
            use_cbn=False,
            kernel_size=5
    ):
        '''
            rnn_layers: the number of lstm layers in the crn,
            rnn_units: for clstm, rnn_units = real+imag
        '''

        super(DCCRN, self).__init__()

        # for fft
        self.win_len = win_len
        self.win_inc = win_inc
        self.fft_len = fft_len
        self.win_type = win_type

        input_dim = win_len
        output_dim = win_len

        self.rnn_units = rnn_units
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.hidden_layers = rnn_layers
        self.kernel_size = kernel_size
        kernel_num = cfg.dccrn_kernel_num
        self.kernel_num = [2] + kernel_num
        self.masking_mode = masking_mode

        # bidirectional=True
        bidirectional = False
        fac = 2 if bidirectional else 1

        fix = True
        self.fix = fix
        self.stft = ConvSTFT(self.win_len, self.win_inc, fft_len, self.win_type, 'complex', fix=fix)
        self.istft = ConviSTFT(self.win_len, self.win_inc, fft_len, self.win_type, 'complex', fix=fix)

        self.encoder = nn.ModuleList()
        self.decoder = nn.ModuleList()
        for idx in range(len(self.kernel_num) - 1):
            self.encoder.append(
                nn.Sequential(
                    # nn.ConstantPad2d([0, 0, 0, 0], 0),
                    ComplexConv2d(
                        self.kernel_num[idx],
                        self.kernel_num[idx + 1],
                        kernel_size=(self.kernel_size, 2),
                        stride=(2, 1),
                        padding=(2, 1)
                    ),
                    nn.BatchNorm2d(self.kernel_num[idx + 1]) if not use_cbn else ComplexBatchNorm(
                        self.kernel_num[idx + 1]),
                    nn.PReLU()
                )
            )
        hidden_dim = self.fft_len // (2 ** (len(self.kernel_num)))

        if cfg.lstm == 'complex':
            rnns = []
            for idx in range(rnn_layers):
                rnns.append(
                    NavieComplexLSTM(
                        input_size=hidden_dim * self.kernel_num[-1] if idx == 0 else self.rnn_units,
                        hidden_size=self.rnn_units,
                        bidirectional=bidirectional,
                        batch_first=False,
                        projection_dim=hidden_dim * self.kernel_num[-1] if idx == rnn_layers - 1 else None,
                    )
                )
                self.enhance = nn.Sequential(*rnns)
        else:
            self.enhance = nn.LSTM(
                input_size=hidden_dim * self.kernel_num[-1],
                hidden_size=self.rnn_units,
                num_layers=2,
                dropout=0.0,
                bidirectional=bidirectional,
                batch_first=False
            )
            self.tranform = nn.Linear(self.rnn_units * fac, hidden_dim * self.kernel_num[-1])

        if cfg.skip_type:
            for idx in range(len(self.kernel_num) - 1, 0, -1):
                if idx != 1:
                    self.decoder.append(
                        nn.Sequential(
                            ComplexConvTranspose2d(
                                self.kernel_num[idx] * 2,
                                self.kernel_num[idx - 1],
                                kernel_size=(self.kernel_size, 2),
                                stride=(2, 1),
                                padding=(2, 0),
                                output_padding=(1, 0)
                            ),
                            nn.BatchNorm2d(self.kernel_num[idx - 1]) if not use_cbn else ComplexBatchNorm(
                                self.kernel_num[idx - 1]),
                            nn.PReLU()
                        )
                    )
                else:
                    self.decoder.append(
                        nn.Sequential(
                            ComplexConvTranspose2d(
                                self.kernel_num[idx] * 2,
                                self.kernel_num[idx - 1],
                                kernel_size=(self.kernel_size, 2),
                                stride=(2, 1),
                                padding=(2, 0),
                                output_padding=(1, 0)
                            ),
                        )
                    )
        else:  # you can erase the skip connection
            for idx in range(len(self.kernel_num) - 1, 0, -1):
                if idx != 1:
                    self.decoder.append(
                        nn.Sequential(
                            ComplexConvTranspose2d(
                                self.kernel_num[idx],
                                self.kernel_num[idx - 1],
                                kernel_size=(self.kernel_size, 2),
                                stride=(2, 1),
                                padding=(2, 0),
                                output_padding=(1, 0)
                            ),
                            nn.BatchNorm2d(self.kernel_num[idx - 1]) if not use_cbn else ComplexBatchNorm(
                                self.kernel_num[idx - 1]),
                            # nn.ELU()
                            nn.PReLU()
                        )
                    )
                else:
                    self.decoder.append(
                        nn.Sequential(
                            ComplexConvTranspose2d(
                                self.kernel_num[idx],
                                self.kernel_num[idx - 1],
                                kernel_size=(self.kernel_size, 2),
                                stride=(2, 1),
                                padding=(2, 0),
                                output_padding=(1, 0)
                            ),
                        )
                    )
        self.flatten_parameters()

    def flatten_parameters(self):
        if isinstance(self.enhance, nn.LSTM):
            self.enhance.flatten_parameters()

    def forward(self, inputs, targets=0):
        specs = self.stft(inputs)
        real = specs[:, :self.fft_len // 2 + 1]
        imag = specs[:, self.fft_len // 2 + 1:]
        spec_mags = torch.sqrt(real ** 2 + imag ** 2 + 1e-8)

        spec_phase = torch.atan2(imag, real)
        cspecs = torch.stack([real, imag], 1)
        cspecs = cspecs[:, :, 1:]
        '''
        means = torch.mean(cspecs, [1,2,3], keepdim=True)
        std = torch.std(cspecs, [1,2,3], keepdim=True )
        normed_cspecs = (cspecs-means)/(std+1e-8)
        out = normed_cspecs
        '''

        out = cspecs
        encoder_out = []

        for idx, layer in enumerate(self.encoder):
            out = layer(out)
            #    print('encoder', out.size())
            encoder_out.append(out)

        batch_size, channels, dims, lengths = out.size()
        out = out.permute(3, 0, 1, 2)
        if cfg.lstm == 'complex':
            r_rnn_in = out[:, :, :channels // 2]
            i_rnn_in = out[:, :, channels // 2:]
            r_rnn_in = torch.reshape(r_rnn_in, [lengths, batch_size, channels // 2 * dims])
            i_rnn_in = torch.reshape(i_rnn_in, [lengths, batch_size, channels // 2 * dims])

            r_rnn_in, i_rnn_in = self.enhance([r_rnn_in, i_rnn_in])

            r_rnn_in = torch.reshape(r_rnn_in, [lengths, batch_size, channels // 2, dims])
            i_rnn_in = torch.reshape(i_rnn_in, [lengths, batch_size, channels // 2, dims])
            out = torch.cat([r_rnn_in, i_rnn_in], 2)
        else:
            # to [L, B, C, D]
            out = torch.reshape(out, [lengths, batch_size, channels * dims])
            out, _ = self.enhance(out)
            out = self.tranform(out)
            out = torch.reshape(out, [lengths, batch_size, channels, dims])

        out = out.permute(1, 2, 3, 0)

        if cfg.skip_type:  # use skip connection
            for idx in range(len(self.decoder)):
                out = complex_cat([out, encoder_out[-1 - idx]], 1)
                out = self.decoder[idx](out)
                out = out[..., 1:]  #
        else:
            for idx in range(len(self.decoder)):
                out = self.decoder[idx](out)
                out = out[..., 1:]

        if self.masking_mode == 'Direct(None make)':
            # for loss calculation
            target_specs = self.stft(targets)
            target_real = target_specs[:, :self.fft_len // 2 + 1]
            target_imag = target_specs[:, self.fft_len // 2 + 1:]

            # spectral mapping
            out_real = out[:, 0]
            out_imag = out[:, 1]
            out_real = F.pad(out_real, [0, 0, 1, 0])
            out_imag = F.pad(out_imag, [0, 0, 1, 0])

            out_spec = torch.cat([out_real, out_imag], 1)

            out_wav = self.istft(out_spec)
            out_wav = torch.squeeze(out_wav, 1)
            out_wav = torch.clamp_(out_wav, -1, 1)

            return out_real, target_real, out_imag, target_imag, out_wav
        else:
            #    print('decoder', out.size())
            mask_real = out[:, 0]
            mask_imag = out[:, 1]
            mask_real = F.pad(mask_real, [0, 0, 1, 0])
            mask_imag = F.pad(mask_imag, [0, 0, 1, 0])

            if self.masking_mode == 'E':
                mask_mags = (mask_real ** 2 + mask_imag ** 2) ** 0.5
                real_phase = mask_real / (mask_mags + 1e-8)
                imag_phase = mask_imag / (mask_mags + 1e-8)
                mask_phase = torch.atan2(
                    imag_phase,
                    real_phase
                )

                # mask_mags = torch.clamp_(mask_mags,0,100)
                mask_mags = torch.tanh(mask_mags)
                est_mags = mask_mags * spec_mags
                est_phase = spec_phase + mask_phase
                out_real = est_mags * torch.cos(est_phase)
                out_imag = est_mags * torch.sin(est_phase)
            elif self.masking_mode == 'C':
                out_real, out_imag = real * mask_real - imag * mask_imag, real * mask_imag + imag * mask_real
            elif self.masking_mode == 'R':
                out_real, out_imag = real * mask_real, imag * mask_imag

            out_spec = torch.cat([out_real, out_imag], 1)

            out_wav = self.istft(out_spec)
            out_wav = torch.squeeze(out_wav, 1)
            out_wav = torch.clamp_(out_wav, -1, 1)

            return out_real, out_imag, out_wav

    def get_params(self, weight_decay=0.0):
        # add L2 penalty
        weights, biases = [], []
        for name, param in self.named_parameters():
            if 'bias' in name:
                biases += [param]
            else:
                weights += [param]
        params = [{
            'params': weights,
            'weight_decay': weight_decay,
        }, {
            'params': biases,
            'weight_decay': 0.0,
        }]
        return params

    def loss(self, estimated, target, real_spec=0, img_spec=0, perceptual=False):
        if perceptual:
            if cfg.perceptual == 'LMS':
                clean_specs = self.stft(target)
                clean_real = clean_specs[:, :self.fft_len // 2 + 1]
                clean_imag = clean_specs[:, self.fft_len // 2 + 1:]
                clean_mags = torch.sqrt(clean_real ** 2 + clean_imag ** 2 + 1e-7)

                est_clean_mags = torch.sqrt(real_spec ** 2 + img_spec ** 2 + 1e-7)
                return get_array_lms_loss(clean_mags, est_clean_mags)
            elif cfg.perceptual == 'PMSQE':
                return get_array_pmsqe_loss(target, estimated)
        else:
            if cfg.loss == 'MSE':
                return F.mse_loss(estimated, target, reduction='mean')
            elif cfg.loss == 'SDR':
                return -sdr(target, estimated)
            elif cfg.loss == 'SI-SNR':
                return -(si_snr(estimated, target))
            elif cfg.loss == 'SI-SDR':
                return -(si_sdr(target, estimated))


#######################################################################
#                            real network                             #
#######################################################################
class CRN(nn.Module):
    def __init__(
            self,
            rnn_layers=cfg.rnn_layers,
            rnn_input_size=cfg.rnn_input_size,
            rnn_units=cfg.rnn_units,
            win_len=cfg.win_len,
            win_inc=cfg.win_inc,
            fft_len=cfg.fft_len,
            win_type=cfg.window,
            masking_mode=cfg.masking_mode,
            kernel_size=5
    ):
        '''
            rnn_layers: the number of lstm layers in the crn
        '''

        super(CRN, self).__init__()

        # for fft
        self.win_len = win_len
        self.win_inc = win_inc
        self.fft_len = fft_len
        self.win_type = win_type

        input_dim = win_len
        output_dim = win_len

        self.rnn_input_size = rnn_input_size
        self.rnn_units = rnn_units//2
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.hidden_layers = rnn_layers
        self.kernel_size = kernel_size
        kernel_num = cfg.dccrn_kernel_num
        self.kernel_num = [2] + kernel_num
        self.masking_mode = masking_mode

        # bidirectional=True
        bidirectional = False

        self.stft = ConvSTFT(self.win_len, self.win_inc, fft_len, self.win_type, 'real')
        self.istft = ConviSTFT(self.win_len, self.win_inc, fft_len, self.win_type, 'complex')

        self.encoder = nn.ModuleList()
        self.decoder = nn.ModuleList()
        for idx in range(len(self.kernel_num) - 1):
            self.encoder.append(
                nn.Sequential(
                    RealConv2d(
                        self.kernel_num[idx] // 2,
                        self.kernel_num[idx + 1] // 2,
                        kernel_size=(self.kernel_size, 2),
                        stride=(2, 1),
                        padding=(2, 1)
                    ),
                    nn.BatchNorm2d(self.kernel_num[idx + 1] // 2),
                    nn.PReLU()
                )
            )
        hidden_dim = self.fft_len // (2 ** (len(self.kernel_num)))

        self.enhance = nn.LSTM(
            input_size=self.rnn_input_size,
            hidden_size=self.rnn_units,
            dropout=0.0,
            bidirectional=bidirectional,
            batch_first=False
        )
        self.tranform = nn.Linear(self.rnn_units, self.rnn_input_size)

        if cfg.skip_type:
            for idx in range(len(self.kernel_num) - 1, 0, -1):
                if idx != 1:
                    self.decoder.append(
                        nn.Sequential(
                            RealConvTranspose2d(
                                self.kernel_num[idx],
                                self.kernel_num[idx - 1] // 2,
                                kernel_size=(self.kernel_size, 2),
                                stride=(2, 1),
                                padding=(2, 0),
                                output_padding=(1, 0)
                            ),
                            nn.BatchNorm2d(self.kernel_num[idx - 1] // 2),
                            nn.PReLU()
                        )
                    )
                else:
                    self.decoder.append(
                        nn.Sequential(
                            RealConvTranspose2d(
                                self.kernel_num[idx],
                                self.kernel_num[idx - 1] // 2,
                                kernel_size=(self.kernel_size, 2),
                                stride=(2, 1),
                                padding=(2, 0),
                                output_padding=(1, 0)
                            ),
                        )
                    )
        else:
            for idx in range(len(self.kernel_num) - 1, 0, -1):
                if idx != 1:
                    self.decoder.append(
                        nn.Sequential(
                            nn.ConvTranspose2d(
                                self.kernel_num[idx],
                                self.kernel_num[idx - 1],
                                kernel_size=(self.kernel_size, 2),
                                stride=(2, 1),
                                padding=(2, 0),
                                output_padding=(1, 0)
                            ),
                            nn.BatchNorm2d(self.kernel_num[idx - 1]),
                            # nn.ELU()
                            nn.PReLU()
                        )
                    )
                else:
                    self.decoder.append(
                        nn.Sequential(
                            nn.ConvTranspose2d(
                                self.kernel_num[idx],
                                self.kernel_num[idx - 1],
                                kernel_size=(self.kernel_size, 2),
                                stride=(2, 1),
                                padding=(2, 0),
                                output_padding=(1, 0)
                            ),
                        )
                    )
        self.flatten_parameters()

    def flatten_parameters(self):
        if isinstance(self.enhance, nn.LSTM):
            self.enhance.flatten_parameters()

    def forward(self, inputs, targets=0):
        mags, phase = self.stft(inputs)

        out = mags
        out = out.unsqueeze(1)
        out = out[:, :, 1:]
        encoder_out = []

        for idx, layer in enumerate(self.encoder):
            out = layer(out)
            #    print('encoder', out.size())
            encoder_out.append(out)

        batch_size, channels, dims, lengths = out.size()
        out = out.permute(3, 0, 1, 2)

        rnn_in = torch.reshape(out, [lengths, batch_size, channels * dims])
        out, _ = self.enhance(rnn_in)
        out = self.tranform(out)
        out = torch.reshape(out, [lengths, batch_size, channels, dims])

        out = out.permute(1, 2, 3, 0)

        if cfg.skip_type:  # use skip connection
            for idx in range(len(self.decoder)):
                out = torch.cat([out, encoder_out[-1 - idx]], 1)
                out = self.decoder[idx](out)
                out = out[..., 1:]  #
        else:
            for idx in range(len(self.decoder)):
                out = self.decoder[idx](out)
                out = out[..., 1:]

        # mask_mags = F.pad(out, [0, 0, 1, 0])
        out = out.squeeze(1)
        out = F.pad(out, [0, 0, 1, 0])

        # for loss calculation
        target_mags, _ = self.stft(targets)

        if self.masking_mode == 'Direct(None make)':  # spectral mapping
            out_real = out * torch.cos(phase)
            out_imag = out * torch.sin(phase)

            out_spec = torch.cat([out_real, out_imag], 1)

            out_wav = self.istft(out_spec)
            out_wav = torch.squeeze(out_wav, 1)
            out_wav = torch.clamp_(out_wav, -1, 1)

            return out, target_mags, out_wav
        else:  # T-F masking
            # mask_mags = torch.clamp_(mask_mags,0,100)
            # out = F.pad(out, [0, 0, 1, 0])
            mask_mags = torch.tanh(out)
            est_mags = mask_mags * mags
            out_real = est_mags * torch.cos(phase)
            out_imag = est_mags * torch.sin(phase)

            out_spec = torch.cat([out_real, out_imag], 1)

            out_wav = self.istft(out_spec)
            out_wav = torch.squeeze(out_wav, 1)
            out_wav = torch.clamp_(out_wav, -1, 1)

            return est_mags, target_mags, out_wav

    def get_params(self, weight_decay=0.0):
        # add L2 penalty
        weights, biases = [], []
        for name, param in self.named_parameters():
            if 'bias' in name:
                biases += [param]
            else:
                weights += [param]
        params = [{
            'params': weights,
            'weight_decay': weight_decay,
        }, {
            'params': biases,
            'weight_decay': 0.0,
        }]
        return params

    def loss(self, estimated, target, out_mags=0, target_mags=0, perceptual=False):
        if perceptual:
            if cfg.perceptual == 'LMS':
                return get_array_lms_loss(target_mags, out_mags)
            elif cfg.perceptual == 'PMSQE':
                return get_array_pmsqe_loss(target, estimated)
        else:
            if cfg.loss == 'MSE':
                return F.mse_loss(estimated, target, reduction='mean')
            elif cfg.loss == 'SDR':
                return -sdr(target, estimated)
            elif cfg.loss == 'SI-SNR':
                return -(si_snr(estimated, target))
            elif cfg.loss == 'SI-SDR':
                return -(si_sdr(target, estimated))

            
class FullSubNet(BaseModel):
    def __init__(self,
                 sb_num_neighbors=cfg.sb_num_neighbors,
                 fb_num_neighbors=cfg.fb_num_neighbors,
                 num_freqs=cfg.num_freqs,
                 look_ahead=cfg.look_ahead,
                 sequence_model=cfg.sequence_model,
                 fb_output_activate_function=cfg.fb_output_activate_function,
                 sb_output_activate_function=cfg.sb_output_activate_function,
                 fb_model_hidden_size=cfg.fb_model_hidden_size,
                 sb_model_hidden_size=cfg.sb_model_hidden_size,
                 weight_init=cfg.weight_init,
                 norm_type=cfg.norm_type,
                 ):
        """
        FullSubNet model (cIRM mask)

        Args:
            num_freqs: Frequency dim of the input
            look_ahead: Number of use of the future frames
            fb_num_neighbors: How much neighbor frequencies at each side from fullband model's output
            sb_num_neighbors: How much neighbor frequencies at each side from noisy spectrogram
            sequence_model: Chose one sequence model as the basic model e.g., GRU, LSTM
            fb_output_activate_function: fullband model's activation function
            sb_output_activate_function: subband model's activation function
            norm_type: type of normalization, see more details in "BaseModel" class
        """
        super().__init__()
        assert sequence_model in ("GRU", "LSTM"), f"{self.__class__.__name__} only support GRU and LSTM."

        self.fb_model = SequenceModel(
            input_size=num_freqs,
            output_size=num_freqs,
            hidden_size=fb_model_hidden_size,
            num_layers=2,
            bidirectional=False,
            sequence_model=sequence_model,
            output_activate_function=fb_output_activate_function
        )

        self.sb_model = SequenceModel(
            input_size=(sb_num_neighbors * 2 + 1) + (fb_num_neighbors * 2 + 1),
            output_size=2,
            hidden_size=sb_model_hidden_size,
            num_layers=2,
            bidirectional=False,
            sequence_model=sequence_model,
            output_activate_function=sb_output_activate_function
        )

        self.sb_num_neighbors = sb_num_neighbors
        self.fb_num_neighbors = fb_num_neighbors
        self.look_ahead = look_ahead
        self.norm = self.norm_wrapper(norm_type)

        if weight_init:
            self.apply(self.weight_init)

    def forward(self, noisy_mag):
        """
        Args:
            noisy_mag: noisy magnitude spectrogram

        Returns:
            The real part and imag part of the enhanced spectrogram

        Shapes:
            noisy_mag: [B, 1, F, T]
            return: [B, 2, F, T]
        """
        if not noisy_mag.dim() == 4:
            noisy_mag = noisy_mag.unsqueeze(1)
        noisy_mag = F.pad(noisy_mag, [0, self.look_ahead])  # Pad the look ahead
        batch_size, num_channels, num_freqs, num_frames = noisy_mag.size()
        assert num_channels == 1, f"{self.__class__.__name__} takes the mag feature as inputs."

        # Fullband model
        fb_input = self.norm(noisy_mag).reshape(batch_size, num_channels * num_freqs, num_frames)
        fb_output = self.fb_model(fb_input).reshape(batch_size, 1, num_freqs, num_frames)

        # Unfold fullband model's output, [B, N=F, C, F_f, T]. N is the number of sub-band units
        fb_output_unfolded = self.unfold(fb_output, num_neighbor=self.fb_num_neighbors)
        fb_output_unfolded = fb_output_unfolded.reshape(batch_size, num_freqs, self.fb_num_neighbors * 2 + 1, num_frames)

        # Unfold noisy spectrogram, [B, N=F, C, F_s, T]
        noisy_mag_unfolded = self.unfold(noisy_mag, num_neighbor=self.sb_num_neighbors)
        noisy_mag_unfolded = noisy_mag_unfolded.reshape(batch_size, num_freqs, self.sb_num_neighbors * 2 + 1, num_frames)

        # Concatenation, [B, F, (F_s + F_f), T]
        sb_input = torch.cat([noisy_mag_unfolded, fb_output_unfolded], dim=2)
        sb_input = self.norm(sb_input)

        sb_input = sb_input.reshape(
            batch_size * num_freqs,
            (self.sb_num_neighbors * 2 + 1) + (self.fb_num_neighbors * 2 + 1),
            num_frames
        )

        # [B * F, (F_s + F_f), T] => [B * F, 2, T] => [B, F, 2, T]
        sb_mask = self.sb_model(sb_input)
        sb_mask = sb_mask.reshape(batch_size, num_freqs, 2, num_frames).permute(0, 2, 1, 3).contiguous()

        output = sb_mask[:, :, :, self.look_ahead:]
        output = output.permute(0, 2, 3, 1)
        return output

    def loss(self, estimated, target):
            if cfg.loss == 'MSE':
                return F.mse_loss(estimated, target, reduction='mean')
            elif cfg.loss == 'SDR':
                return -sdr(target, estimated)
            elif cfg.loss == 'SI-SNR':
                return -(si_snr(estimated, target))
            elif cfg.loss == 'SI-SDR':
                return -(si_sdr(target, estimated))
            
