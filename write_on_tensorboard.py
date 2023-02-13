"""
For observing the results using tensorboard

1. wav
2. spectrogram
3. loss
"""
from tensorboardX import SummaryWriter
import matplotlib
import config as cfg


class Writer(SummaryWriter):
    def __init__(self, logdir):
        super(Writer, self).__init__(logdir)
        # mask real/ imag
        cmap_custom = {
            'red': ((0.0, 0.0, 0.0),
                    (1 / 63, 0.0, 0.0),
                    (2 / 63, 0.0, 0.0),
                    (3 / 63, 0.0, 0.0),
                    (4 / 63, 0.0, 0.0),
                    (5 / 63, 0.0, 0.0),
                    (6 / 63, 0.0, 0.0),
                    (7 / 63, 0.0, 0.0),
                    (8 / 63, 0.0, 0.0),
                    (9 / 63, 0.0, 0.0),
                    (10 / 63, 0.0, 0.0),
                    (11 / 63, 0.0, 0.0),
                    (12 / 63, 0.0, 0.0),
                    (13 / 63, 0.0, 0.0),
                    (14 / 63, 0.0, 0.0),
                    (15 / 63, 0.0, 0.0),
                    (16 / 63, 0.0, 0.0),
                    (17 / 63, 0.0, 0.0),
                    (18 / 63, 0.0, 0.0),
                    (19 / 63, 0.0, 0.0),
                    (20 / 63, 0.0, 0.0),
                    (21 / 63, 0.0, 0.0),
                    (22 / 63, 0.0, 0.0),
                    (23 / 63, 0.0, 0.0),
                    (24 / 63, 0.5625, 0.5625),
                    (25 / 63, 0.6250, 0.6250),
                    (26 / 63, 0.6875, 0.6875),
                    (27 / 63, 0.7500, 0.7500),
                    (28 / 63, 0.8125, 0.8125),
                    (29 / 63, 0.8750, 0.8750),
                    (30 / 63, 0.9375, 0.9375),
                    (31 / 63, 1.0, 1.0),
                    (32 / 63, 1.0, 1.0),
                    (33 / 63, 1.0, 1.0),
                    (34 / 63, 1.0, 1.0),
                    (35 / 63, 1.0, 1.0),
                    (36 / 63, 1.0, 1.0),
                    (37 / 63, 1.0, 1.0),
                    (38 / 63, 1.0, 1.0),
                    (39 / 63, 1.0, 1.0),
                    (40 / 63, 1.0, 1.0),
                    (41 / 63, 1.0, 1.0),
                    (42 / 63, 1.0, 1.0),
                    (43 / 63, 1.0, 1.0),
                    (44 / 63, 1.0, 1.0),
                    (45 / 63, 1.0, 1.0),
                    (46 / 63, 1.0, 1.0),
                    (47 / 63, 1.0, 1.0),
                    (48 / 63, 1.0, 1.0),
                    (49 / 63, 1.0, 1.0),
                    (50 / 63, 1.0, 1.0),
                    (51 / 63, 1.0, 1.0),
                    (52 / 63, 1.0, 1.0),
                    (53 / 63, 1.0, 1.0),
                    (54 / 63, 1.0, 1.0),
                    (55 / 63, 1.0, 1.0),
                    (56 / 63, 0.9375, 0.9375),
                    (57 / 63, 0.8750, 0.8750),
                    (58 / 63, 0.8125, 0.8125),
                    (59 / 63, 0.7500, 0.7500),
                    (60 / 63, 0.6875, 0.6875),
                    (61 / 63, 0.6250, 0.6250),
                    (62 / 63, 0.5625, 0.5625),
                    (63 / 63, 0.5000, 0.5000)),
            'green': ((0.0, 0.0, 0.0),
                      (1 / 63, 0.0, 0.0),
                      (2 / 63, 0.0, 0.0),
                      (3 / 63, 0.0, 0.0),
                      (4 / 63, 0.0, 0.0),
                      (5 / 63, 0.0, 0.0),
                      (6 / 63, 0.0, 0.0),
                      (7 / 63, 0.0, 0.0),
                      (8 / 63, 0.0625, 0.0625),
                      (9 / 63, 0.1250, 0.1250),
                      (10 / 63, 0.1875, 0.1875),
                      (11 / 63, 0.2500, 0.2500),
                      (12 / 63, 0.3125, 0.3125),
                      (13 / 63, 0.3750, 0.3750),
                      (14 / 63, 0.4375, 0.4375),
                      (15 / 63, 0.5000, 0.5000),
                      (16 / 63, 0.5625, 0.5625),
                      (17 / 63, 0.6250, 0.6250),
                      (18 / 63, 0.6875, 0.6875),
                      (19 / 63, 0.7500, 0.7500),
                      (20 / 63, 0.8125, 0.8125),
                      (21 / 63, 0.8750, 0.8750),
                      (22 / 63, 0.9375, 0.9375),
                      (23 / 63, 1.0, 1.0),
                      (24 / 63, 1.0, 1.0),
                      (25 / 63, 1.0, 1.0),
                      (26 / 63, 1.0, 1.0),
                      (27 / 63, 1.0, 1.0),
                      (28 / 63, 1.0, 1.0),
                      (29 / 63, 1.0, 1.0),
                      (30 / 63, 1.0, 1.0),
                      (31 / 63, 1.0, 1.0),
                      (32 / 63, 1.0, 1.0),
                      (33 / 63, 1.0, 1.0),
                      (34 / 63, 1.0, 1.0),
                      (35 / 63, 1.0, 1.0),
                      (36 / 63, 1.0, 1.0),
                      (37 / 63, 1.0, 1.0),
                      (38 / 63, 1.0, 1.0),
                      (39 / 63, 1.0, 1.0),
                      (40 / 63, 0.9375, 0.9375),
                      (41 / 63, 0.8750, 0.8750),
                      (42 / 63, 0.8125, 0.8125),
                      (43 / 63, 0.7500, 0.7500),
                      (44 / 63, 0.6875, 0.6875),
                      (45 / 63, 0.6250, 0.6250),
                      (46 / 63, 0.5625, 0.5625),
                      (47 / 63, 0.5000, 0.5000),
                      (48 / 63, 0.4375, 0.4375),
                      (49 / 63, 0.3750, 0.3750),
                      (50 / 63, 0.3125, 0.3125),
                      (51 / 63, 0.2500, 0.2500),
                      (52 / 63, 0.1875, 0.1875),
                      (53 / 63, 0.1250, 0.1250),
                      (54 / 63, 0.0625, 0.0625),
                      (55 / 63, 0.0, 0.0),
                      (56 / 63, 0.0, 0.0),
                      (57 / 63, 0.0, 0.0),
                      (58 / 63, 0.0, 0.0),
                      (59 / 63, 0.0, 0.0),
                      (60 / 63, 0.0, 0.0),
                      (61 / 63, 0.0, 0.0),
                      (62 / 63, 0.0, 0.0),
                      (63 / 63, 0.0, 0.0)),
            'blue': ((0.0, 0.5625, 0.5625),
                     (1 / 63, 0.6250, 0.6250),
                     (2 / 63, 0.6875, 0.6875),
                     (3 / 63, 0.7500, 0.7500),
                     (4 / 63, 0.8125, 0.8125),
                     (5 / 63, 0.8750, 0.8750),
                     (6 / 63, 0.9375, 0.9375),
                     (7 / 63, 1.0, 1.0),
                     (8 / 63, 1.0, 1.0),
                     (9 / 63, 1.0, 1.0),
                     (10 / 63, 1.0, 1.0),
                     (11 / 63, 1.0, 1.0),
                     (12 / 63, 1.0, 1.0),
                     (13 / 63, 1.0, 1.0),
                     (14 / 63, 1.0, 1.0),
                     (15 / 63, 1.0, 1.0),
                     (16 / 63, 1.0, 1.0),
                     (17 / 63, 1.0, 1.0),
                     (18 / 63, 1.0, 1.0),
                     (19 / 63, 1.0, 1.0),
                     (20 / 63, 1.0, 1.0),
                     (21 / 63, 1.0, 1.0),
                     (22 / 63, 1.0, 1.0),
                     (23 / 63, 1.0, 1.0),
                     (24 / 63, 1.0, 1.0),
                     (25 / 63, 1.0, 1.0),
                     (26 / 63, 1.0, 1.0),
                     (27 / 63, 1.0, 1.0),
                     (28 / 63, 1.0, 1.0),
                     (29 / 63, 1.0, 1.0),
                     (30 / 63, 1.0, 1.0),
                     (31 / 63, 1.0, 1.0),
                     (32 / 63, 0.9375, 0.9375),
                     (33 / 63, 0.8750, 0.8750),
                     (34 / 63, 0.8125, 0.8125),
                     (35 / 63, 0.7500, 0.7500),
                     (36 / 63, 0.6875, 0.6875),
                     (37 / 63, 0.6250, 0.6250),
                     (38 / 63, 0.5625, 0.5625),
                     (39 / 63, 0.0, 0.0),
                     (40 / 63, 0.0, 0.0),
                     (41 / 63, 0.0, 0.0),
                     (42 / 63, 0.0, 0.0),
                     (43 / 63, 0.0, 0.0),
                     (44 / 63, 0.0, 0.0),
                     (45 / 63, 0.0, 0.0),
                     (46 / 63, 0.0, 0.0),
                     (47 / 63, 0.0, 0.0),
                     (48 / 63, 0.0, 0.0),
                     (49 / 63, 0.0, 0.0),
                     (50 / 63, 0.0, 0.0),
                     (51 / 63, 0.0, 0.0),
                     (52 / 63, 0.0, 0.0),
                     (53 / 63, 0.0, 0.0),
                     (54 / 63, 0.0, 0.0),
                     (55 / 63, 0.0, 0.0),
                     (56 / 63, 0.0, 0.0),
                     (57 / 63, 0.0, 0.0),
                     (58 / 63, 0.0, 0.0),
                     (59 / 63, 0.0, 0.0),
                     (60 / 63, 0.0, 0.0),
                     (61 / 63, 0.0, 0.0),
                     (62 / 63, 0.0, 0.0),
                     (63 / 63, 0.0, 0.0))
        }

        # mask magnitude
        cmap_custom2 = {
            'red': ((0.0, 1.0, 1.0),
                    (1 / 32, 1.0, 1.0),
                    (2 / 32, 1.0, 1.0),
                    (3 / 32, 1.0, 1.0),
                    (4 / 32, 1.0, 1.0),
                    (5 / 32, 1.0, 1.0),
                    (6 / 32, 1.0, 1.0),
                    (7 / 32, 1.0, 1.0),
                    (8 / 32, 1.0, 1.0),
                    (9 / 32, 1.0, 1.0),
                    (10 / 32, 1.0, 1.0),
                    (11 / 32, 1.0, 1.0),
                    (12 / 32, 1.0, 1.0),
                    (13 / 32, 1.0, 1.0),
                    (14 / 32, 1.0, 1.0),
                    (15 / 32, 1.0, 1.0),
                    (16 / 32, 1.0, 1.0),
                    (17 / 32, 1.0, 1.0),
                    (18 / 32, 1.0, 1.0),
                    (19 / 32, 1.0, 1.0),
                    (20 / 32, 1.0, 1.0),
                    (21 / 32, 1.0, 1.0),
                    (22 / 32, 1.0, 1.0),
                    (23 / 32, 1.0, 1.0),
                    (24 / 32, 1.0, 1.0),
                    (25 / 32, 0.9375, 0.9375),
                    (26 / 32, 0.8750, 0.8750),
                    (27 / 32, 0.8125, 0.8125),
                    (28 / 32, 0.7500, 0.7500),
                    (29 / 32, 0.6875, 0.6875),
                    (30 / 32, 0.6250, 0.6250),
                    (31 / 32, 0.5625, 0.5625),
                    (32 / 32, 0.5000, 0.5000)),
            'green': ((0.0, 1.0, 1.0),
                      (1 / 32, 1.0, 1.0),
                      (2 / 32, 1.0, 1.0),
                      (3 / 32, 1.0, 1.0),
                      (4 / 32, 1.0, 1.0),
                      (5 / 32, 1.0, 1.0),
                      (6 / 32, 1.0, 1.0),
                      (7 / 32, 1.0, 1.0),
                      (8 / 32, 1.0, 1.0),
                      (9 / 32, 0.9375, 0.9375),
                      (10 / 32, 0.8750, 0.8750),
                      (11 / 32, 0.8125, 0.8125),
                      (12 / 32, 0.7500, 0.7500),
                      (13 / 32, 0.6875, 0.6875),
                      (14 / 32, 0.6250, 0.6250),
                      (15 / 32, 0.5625, 0.5625),
                      (16 / 32, 0.5000, 0.5000),
                      (17 / 32, 0.4375, 0.4375),
                      (18 / 32, 0.3750, 0.3750),
                      (19 / 32, 0.3125, 0.3125),
                      (20 / 32, 0.2500, 0.2500),
                      (21 / 32, 0.1875, 0.1875),
                      (22 / 32, 0.1250, 0.1250),
                      (23 / 32, 0.0625, 0.0625),
                      (24 / 32, 0.0, 0.0),
                      (25 / 32, 0.0, 0.0),
                      (26 / 32, 0.0, 0.0),
                      (27 / 32, 0.0, 0.0),
                      (28 / 32, 0.0, 0.0),
                      (29 / 32, 0.0, 0.0),
                      (30 / 32, 0.0, 0.0),
                      (31 / 32, 0.0, 0.0),
                      (32 / 32, 0.0, 0.0)),
            'blue': ((0.0, 1.0, 1.0),
                     (1 / 32, 0.9375, 0.9375),
                     (2 / 32, 0.8750, 0.8750),
                     (3 / 32, 0.8125, 0.8125),
                     (4 / 32, 0.7500, 0.7500),
                     (5 / 32, 0.6875, 0.6875),
                     (6 / 32, 0.6250, 0.6250),
                     (7 / 32, 0.5625, 0.5625),
                     (8 / 32, 0.0, 0.0),
                     (9 / 32, 0.0, 0.0),
                     (10 / 32, 0.0, 0.0),
                     (11 / 32, 0.0, 0.0),
                     (12 / 32, 0.0, 0.0),
                     (13 / 32, 0.0, 0.0),
                     (14 / 32, 0.0, 0.0),
                     (15 / 32, 0.0, 0.0),
                     (16 / 32, 0.0, 0.0),
                     (17 / 32, 0.0, 0.0),
                     (18 / 32, 0.0, 0.0),
                     (19 / 32, 0.0, 0.0),
                     (20 / 32, 0.0, 0.0),
                     (21 / 32, 0.0, 0.0),
                     (22 / 32, 0.0, 0.0),
                     (23 / 32, 0.0, 0.0),
                     (24 / 32, 0.0, 0.0),
                     (25 / 32, 0.0, 0.0),
                     (26 / 32, 0.0, 0.0),
                     (27 / 32, 0.0, 0.0),
                     (28 / 32, 0.0, 0.0),
                     (29 / 32, 0.0, 0.0),
                     (30 / 32, 0.0, 0.0),
                     (31 / 32, 0.0, 0.0),
                     (32 / 32, 0.0, 0.0))
        }

        self.cmap_custom = matplotlib.colors.LinearSegmentedColormap('testCmap', segmentdata=cmap_custom, N=256)
        self.cmap_custom2 = matplotlib.colors.LinearSegmentedColormap('testCmap2', segmentdata=cmap_custom2, N=256)

    def log_loss(self, train_loss, vali_loss, step):
        self.add_scalar('train_loss', train_loss, step)
        self.add_scalar('vali_loss', vali_loss, step)

    def log_sub_loss(self, train_main_loss, train_sub_loss, vali_main_loss, vali_sub_loss, step):
        self.add_scalar('train_main_loss', train_main_loss, step)
        self.add_scalar('train_sub_loss', train_sub_loss, step)
        self.add_scalar('vali_main_loss', vali_main_loss, step)
        self.add_scalar('vali_sub_loss', vali_sub_loss, step)

    def log_score(self, vali_pesq, vali_stoi, step):
        self.add_scalar('vali_pesq', vali_pesq, step)
        self.add_scalar('vali_stoi', vali_stoi, step)

    def log_wav(self, mixed_wav, clean_wav, est_wav, step):
        # <Audio>
        self.add_audio('mixed_wav', mixed_wav, step, cfg.fs)
        self.add_audio('clean_target_wav', clean_wav, step, cfg.fs)
        self.add_audio('estimated_wav', est_wav, step, cfg.fs)

    def log_spectrogram(self, mixed_wav, clean_wav, noise_wav, est_wav, step):
        # <Data>
        self.add_image('data/mixed_spectrogram',
                       plot_spectrogram_to_numpy(mixed_wav, cfg.fs, cfg.win_len, int(cfg.ola_ratio),
                                                 None, [-150, -40], 'dB'), step,
                       dataformats='HWC')
        self.add_image('data/clean_spectrogram',
                       plot_spectrogram_to_numpy(clean_wav, cfg.fs, cfg.win_len, int(cfg.ola_ratio),
                                                 None, [-150, -40], 'dB'), step,
                       dataformats='HWC')
        self.add_image('data/noise_spectrogram',
                       plot_spectrogram_to_numpy(noise_wav, cfg.fs, cfg.win_len, int(cfg.ola_ratio),
                                                 None, [-150, -40], 'dB'), step,
                       dataformats='HWC')
        self.add_image('data/clean_unwrap_phase',
                       plot_spectrogram_to_numpy(clean_wav, cfg.fs, cfg.win_len, int(cfg.ola_ratio),
                                                 'phase', [-500, 500], None), step,
                       dataformats='HWC')

        # <Results>
        self.add_image('result/estimated_spectrogram',
                       plot_spectrogram_to_numpy(est_wav, cfg.fs, cfg.win_len, int(cfg.ola_ratio),
                                                 None, [-150, -40], 'dB'), step,
                       dataformats='HWC')
        self.add_image('result/estimated_unwrap_phase',
                       plot_spectrogram_to_numpy(est_wav, cfg.fs, cfg.win_len, int(cfg.ola_ratio),
                                                 'phase', [-500, 500], None), step,
                       dataformats='HWC')
        self.add_image('result/estimated_magnitude-clean_magnitude',
                       plot_spectrogram_to_numpy(est_wav - clean_wav, cfg.fs, cfg.win_len,
                                                 int(cfg.ola_ratio), None,
                                                 [-80, 80], 'dB'), step, dataformats='HWC')
        self.add_image('result/estimated_unwrap_phase-clean_unwrap_phase',
                       plot_spectrogram_to_numpy(est_wav - clean_wav, cfg.fs, cfg.win_len,
                                                 int(cfg.ola_ratio), 'phase',
                                                 [-500, 500], None), step, dataformats='HWC')

    def log_mask_spectrogram(self, est_mask_real, est_mask_imag, step):
        # <Data>
        self.add_image('result/estimated_mask_magnitude',
                       plot_mask_to_numpy(np.sqrt(est_mask_real ** 2 + est_mask_imag ** 2), cfg.fs, cfg.win_len,
                                          int(cfg.ola_ratio), 0, 2,
                                          cmap=self.cmap_custom2), step, dataformats='HWC')
        self.add_image('result/estimated_mask_real',
                       plot_mask_to_numpy(est_mask_real, cfg.fs, cfg.win_len, int(cfg.ola_ratio),
                                          -2, 2, cmap=self.cmap_custom), step, dataformats='HWC')
        self.add_image('result/estimated_mask_imag',
                       plot_mask_to_numpy(est_mask_imag, cfg.fs, cfg.win_len, int(cfg.ola_ratio),
                                          -2, 2, cmap=self.cmap_custom), step, dataformats='HWC')
