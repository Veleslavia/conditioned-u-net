import itertools

import os
import pathlib
import sys

import numpy as np

import torch
import torch.nn as nn
import torch.utils.data
import torch.optim as optim
from torch.utils.tensorboard import SummaryWriter

from configs import ex, logger
from datasets.data_utils import reconstruct_with_masks, \
    _get_spectrograms, _compute_masks, amplitude_to_db, rescale, add_noise, _log_resample

from utils.losses import MPELoss, MSLELoss, SISDRLoss, MSESDRLoss, EPS

from collections import namedtuple
AuxData = namedtuple('AuxData', ['stft', 'labels', 'filename', 'offset'], defaults=(None, None, None, None))

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

torch.manual_seed(0)
np.random.seed(0)


class ModelActionPipeline:

    def __init__(self, model, train_loader, val_loader, exp_config):

        self.train_loader = train_loader
        self.val_loader = val_loader
        self.writer = SummaryWriter(log_dir=f'./runs/{int(ex.current_run._id):05d}')
        self.save_cp = exp_config.save_cp
        self.sr = exp_config.expected_sr
        self.cp_path = os.path.join(exp_config.dir_checkpoint, f'{int(ex.current_run._id):05d}')
        if self.save_cp and not os.path.exists(self.cp_path):
            os.mkdir(self.cp_path)
        self.exp_config = exp_config

        self.model = model
        self.model.to(device)

        self.criterion = self.get_loss(exp_config.loss)
        self.metrics = SISDRLoss(reduction='sum')

        self.num_epochs = exp_config.num_epochs

        if self.exp_config.resume_training:
            self.model.load_state_dict(torch.load(os.path.join(exp_config.dir_checkpoint, exp_config.model_checkpoint)),
                                       strict=False)
        self.optimizer = optim.Adam(self.model.parameters(), lr=exp_config.init_lr, weight_decay=0.0005)
        self.scheduler = optim.lr_scheduler.ReduceLROnPlateau(self.optimizer,
                                                              factor=0.5,
                                                              mode='min',
                                                              patience=exp_config.patience,
                                                              verbose=True)
        self._set_train()

    def _is_training(self):
        return self.phase == 'train'

    def _set_train(self):
        self.model.train()
        self.phase = 'train'
        self.loader = self.train_loader

    def _set_eval(self):
        self.model.eval()
        self.phase = 'validation'
        self.loader = self.val_loader

    def _write_summary(self, iteration, loss, lr,
                       sdr=None, masks=None, audio=None,
                       aux_loss=None, aux_metrics=None):
        self.writer.add_scalar(f'loss/{self.phase}', loss, iteration)
        logger.info(f'Iteration #{iteration} loss: {loss:.4f}')
        self.writer.add_scalar(f'lr/{self.phase}', lr, iteration)
        self.writer.add_scalar(f'n_max_sources/{self.phase}', self.loader.dataset.n_mix_max, iteration)
        if sdr is not None:
            self.writer.add_scalar(f'sdr/{self.phase}', sdr, iteration)
        if masks is not None:
            gt_masks, predicted_masks = masks
            for ch_idx in range(gt_masks.shape[1]):
                self.writer.add_images(
                    f'gt_vs_predicted_mask_ch_{ch_idx+1}/{self.phase}',
                    torch.cat([gt_masks[:, ch_idx, :, :], predicted_masks[:, ch_idx, :, :]], dim=2).unsqueeze_(1))
        if audio is not None:
            for piece_idx, source_idx in itertools.product(range(audio.shape[0]), range(audio.shape[1])):
                self.writer.add_audio(
                    f'reconstructed_audio_{piece_idx}_{source_idx+1}/{self.phase}',
                    audio[piece_idx, source_idx],
                    sample_rate=self.exp_config.expected_sr)
        if aux_loss is not None:
            self.writer.add_scalar(f'aux_loss/{self.phase}', aux_loss, iteration)
            self.writer.add_scalar(f'aux_f1/{self.phase}', aux_metrics, iteration)

    def _get_test_segments(self):
        songs = dict()
        for sample_idx in range(len(self.val_loader.dataset)):
            inputs, gt_sources = self.val_loader.dataset[sample_idx]
            if self.exp_config.conditioning:
                mix = torch.Tensor(np.expand_dims(inputs[0], axis=0)).to(device, dtype=torch.float)
                labels = torch.Tensor(np.expand_dims(inputs[1], axis=0)).to(device, dtype=torch.float)
                output = np.array(self.model(mix, labels).detach().data.cpu()[0])
            else:
                mix = torch.Tensor(np.expand_dims(inputs, axis=0)).to(device, dtype=torch.float)
                output = np.array(self.model(mix).detach().data.cpu()[0])

            sample_datafile = self.val_loader.dataset.index_map[sample_idx].datafile
            sample_offset = self.val_loader.dataset.index_map[sample_idx].offset
            filenames = self.val_loader.dataset.metadata[sample_datafile][:, [sample_offset]][0][0]
            local_segment_idx = self.val_loader.dataset.metadata[sample_datafile][:, [sample_offset]][1][0]
            piece_name = filenames[0].split('/')[-2]
            if piece_name not in songs.keys():
                songs[piece_name] = [(source_filename.split('/')[-1], list()) for source_filename in filenames[1:]]
            for source_idx in range(len(output)):
                songs[piece_name][source_idx][1].append((local_segment_idx, output[source_idx][0]))
        return songs

    def _input_preprocessing(self, inputs, gt_data, aux_data):
        if self.model.conditioned:
            inputs, visual_or_labels = inputs[0].to(device, dtype=torch.float), inputs[1].to(device, dtype=torch.float)
        else:
            inputs, visual_or_labels = inputs.to(device, dtype=torch.float), None

        if aux_data and len(aux_data) > 1:
            source_indices = aux_data[1]
        else:
            source_indices = None
        if inputs.shape[-1] == self.exp_config.mock_spec_size[-1]:
            specs = inputs
        else:
            specs = _get_spectrograms(inputs,
                                      stft_frame=self.exp_config.stft_frame,
                                      mock_spec_size=self.exp_config.mock_spec_size,
                                      non_zero_ids=source_indices)

        if self.exp_config.log_resample:
            data = _log_resample(specs, log_sample_n=self.exp_config.log_sample_n)
            mix, gt_data = _compute_masks(data, self.exp_config.ratio_masks)
        else:
            mix, gt_data = _compute_masks(specs, self.exp_config.ratio_masks)

        # it was
        # mix = torch.log(mix)
        if self.exp_config.amp_to_db:
            mix = amplitude_to_db(mix)
        else:
            mix = torch.log(mix)
        if self.exp_config.spec_scale:
            mix = rescale(mix)
        if self.exp_config.add_noise:
            mix = add_noise(mix)

        aux_data = AuxData(specs, *aux_data)

        return mix, gt_data, visual_or_labels, aux_data

    def _get_estimations(self, est_data):
        est_masks, est_labels = est_data, None
        est_masks = (torch.sigmoid(est_masks) > 0.5).float() if not self.exp_config.ratio_masks else est_masks
        return est_masks, est_labels

    def _observe_iteration_results(self, num_it: int, epoch: int,
                                   loss, mix, est_data, gt_data,
                                   aux_data: AuxData, aux_loss=None):

        est_masks, est_labels = self._get_estimations(est_data)
        f1 = None # multi-task legacy

        iteration = num_it + (len(self.loader.dataset) // self.loader.batch_size) * epoch
        if not (iteration % self.exp_config.evaluation_steps):
            wav_predicted, wav_gt = reconstruct_with_masks(est_masks=est_masks, stft_all=aux_data.stft,
                                                           log_resample=self.exp_config.log_resample,
                                                           stft_frame=self.exp_config.stft_frame)
            sdr = -self.metrics(wav_predicted, wav_gt)
            masks = (gt_data, est_masks)
            logger.info(f'Iteration #{iteration} si-sdr: {sdr:.4f}')
        else:
            sdr, masks, wav_predicted = None, None, None

        self._write_summary(iteration, loss.item(), self.optimizer.param_groups[-1]['lr'],
                            sdr, masks, wav_predicted,
                            aux_loss=aux_loss, aux_metrics=f1)

    def get_loss(self, loss_key):
        if loss_key == 'log':
            return MSLELoss()
        elif loss_key == 'mpe':
            return MPELoss()
        elif loss_key == 'sdr' or loss_key == 'sisdr':
            return SISDRLoss()
        elif loss_key == 'msesdr':
            return MSESDRLoss()
        elif loss_key == 'bce':
            return nn.BCEWithLogitsLoss(pos_weight=torch.Tensor([self.exp_config.loss_pos_weight]).to(device))
        elif loss_key == 'l1':
            return nn.L1Loss()
        elif loss_key == 'smoothl1':
            return nn.SmoothL1Loss()
        elif loss_key == 'cce':
            return nn.CrossEntropyLoss()
        else:
            return nn.MSELoss()

    def run_phase(self, epoch: int):
        running_loss = 0.0
        aux_loss_weight = 1e+5

        for num_it, (inputs, gt_data, *aux_data) in enumerate(self.loader):
            # zero the parameter gradients
            self.optimizer.zero_grad()

            mix, gt_data, visual_or_labels, aux_data = self._input_preprocessing(inputs, gt_data, aux_data)

            # forward
            # track history if only in train
            with torch.set_grad_enabled(self._is_training()):
                outputs = self.model(mix, visual_or_labels) if self.model.conditioned else self.model(mix)
                loss = self.criterion(outputs, gt_data)
                aux_loss = None
                total_loss = loss

                # backward + optimize only if in training phase
                if self._is_training():
                    total_loss.backward()
                    if self.exp_config.with_lstm:
                        nn.utils.clip_grad_norm_(self.model.parameters(), 0.5)
                    self.optimizer.step()
                    if (epoch == 0) and (num_it == 0):
                        self.optimizer = optim.Adam(self.model.parameters(), lr=self.exp_config.init_lr,
                                                    weight_decay=0.0005)

                if aux_loss:
                    aux_loss = aux_loss / aux_loss_weight

            # statistics and outputs
            self._observe_iteration_results(num_it=num_it, epoch=epoch, loss=loss,
                                            mix=mix, est_data=outputs, gt_data=gt_data,
                                            aux_data=aux_data, aux_loss=aux_loss)
            running_loss += loss.item() * mix.size(0)

        epoch_loss = running_loss / len(self.loader.dataset)
        logger.info(f'{self.phase} loss: {epoch_loss:.4f}')

        return epoch_loss

    def train_model(self):
        """
        Full training pipeline function:
            - training
            - validation/validation with external metrics every 10 iteration
            - full tests every 100 epochs
        :return:
        """
        best_val_loss = np.inf
        curriculum_patience = 0

        for epoch in range(self.num_epochs):
            logger.info(f'Starting epoch {epoch}/{self.num_epochs}.')
            try:
                self._set_train()
                self.run_phase(epoch)

                self._set_eval()
                epoch_loss = self.run_phase(epoch)

                if epoch_loss < best_val_loss - 1e-4:
                    best_val_loss = epoch_loss
                    curriculum_patience = 0
                else:
                    curriculum_patience += 1

            except KeyboardInterrupt:
                torch.save(self.model.state_dict(), os.path.join(self.cp_path, 'INTERRUPTED.pth'))
                logger.info('Saved interrupt')
                sys.exit(0)

            if self.exp_config.curriculum_training:
                if curriculum_patience > self.exp_config.curriculum_patience:
                    train_inc = self.train_loader.dataset.increase_n_mix()
                    self.val_loader.dataset.increase_n_mix()
                    if train_inc:
                        logger.info('Increased number of sources in a mixture')
                        curriculum_patience = 0
                        best_val_loss = np.inf
                    else:
                        self.scheduler.step(epoch_loss)
            else:
                self.scheduler.step(epoch_loss)

            if self.save_cp and not (epoch % self.exp_config.save_cp_frequency):
                torch.save(self.model.state_dict(),
                           os.path.join(self.cp_path, f'CP{epoch:04d}.pth'))
                logger.info(f'Checkpoint {epoch} saved !')

    def load_state_dict(self, model_checkpoint):

        def new_key(key, model_keys):
            if key not in model_keys and 'encoder' in key:
                """ encoder.*.0.* -> encoder.*.conv.*
                    encoder.*.1.* -> encoder.*.bn.*
                """
                substitute = 'conv' if key[10] == '0' else 'bn'
                return key[:10] + substitute + key[11:]
            return key

        state_dict = torch.load(model_checkpoint)
        model_keys = self.model.state_dict().keys()
        # fix encoder
        new_state_dict = {new_key(k, model_keys): state_dict[k] for k in state_dict}
        return new_state_dict

    def test_model(self, model_checkpoint, output_dir):

        from utils.metrics import si_sd_sdr

        self._set_eval()
        self.model.to(device)

        all_metrics = list()
        mean_si_sdr, mean_sd_sdr, mean_pes, mean_eps = list(), list(), list(), list()

        for num_it, (inputs, gt_data, *aux_data) in enumerate(self.loader):

            mix, gt_data, visual_or_labels, aux_data = self._input_preprocessing(inputs, gt_data, aux_data)
            # zero the parameter gradients
            self.optimizer.zero_grad()
            outputs = self.model(mix, visual_or_labels) if self.model.conditioned else self.model(mix)

            if num_it == 0:
                # load film weights here
                state_dict = self.load_state_dict(model_checkpoint)
                self.model.load_state_dict(state_dict)
                outputs = self.model(mix, visual_or_labels) if self.model.conditioned else self.model(mix)

            est_masks, est_labels = self._get_estimations(outputs)
            wav_predicted, wav_gt = reconstruct_with_masks(est_masks=est_masks, stft_all=aux_data.stft,
                                                           log_resample=self.exp_config.log_resample,
                                                           stft_frame=self.exp_config.stft_frame,
                                                           norbert=self.exp_config.norbert)

            sample_metrics = list()
            for predicted, target in zip(wav_predicted[0], wav_gt[0]):

                si_sdr, sd_sdr = None, None
                predicted_energy_at_true_silence, true_energy_at_silent_prediction = None, None
                target_nz = torch.sum(torch.abs(target)) != 0
                predicted_nz = torch.sum(torch.abs(predicted)) != 0
                if target_nz and predicted_nz:
                    si_sdr, sd_sdr = si_sd_sdr(predicted, target)
                    si_sdr = si_sdr.detach().data.cpu().numpy()[0]
                    sd_sdr = sd_sdr.detach().data.cpu().numpy()[0]
                    mean_si_sdr.append(si_sdr)
                    mean_sd_sdr.append(sd_sdr)
                elif not target_nz and predicted_nz:
                    # compute Predicted Energy at Silence (PES)
                    predicted_energy_at_true_silence = 10 * torch.log10(sum(predicted ** 2) + EPS)
                    predicted_energy_at_true_silence = predicted_energy_at_true_silence.detach().data.cpu().item()
                    mean_pes.append(predicted_energy_at_true_silence)
                elif target_nz and not predicted_nz:
                    # compute Energy at Predicted Silence (EPS)
                    true_energy_at_silent_prediction = 10 * torch.log10(sum(target ** 2) + EPS)
                    true_energy_at_silent_prediction = true_energy_at_silent_prediction.detach().data.cpu().item()
                    mean_eps.append(true_energy_at_silent_prediction)

                sample_metrics.append([si_sdr, sd_sdr,
                                       predicted_energy_at_true_silence, true_energy_at_silent_prediction])

            probs = self.model.visual.probabilities if (self.exp_config.with_resnet and self.exp_config.conditioning == 'mult_final') else None

            self.loader.dataset.save_predicted(output_dir, wav_predicted[0].detach().data.cpu().numpy(), num_it, probs)

            print(f"mean si_sdr: {np.mean(mean_si_sdr)}, mean sd_sdr: {np.mean(mean_sd_sdr)}, "
                  f"mean pes: {np.mean(mean_pes)}, mean eps: {np.mean(mean_eps)}")

            all_metrics.append(sample_metrics)

        metrics_data_path = pathlib.Path(output_dir) / 'metrics.npy'
        np.save(metrics_data_path, np.array(all_metrics))
        print(f"Metrics saved at {metrics_data_path}")
