import importlib
import torch
from torch.nn.parallel import DataParallel, DistributedDataParallel
from collections import OrderedDict
from os import path as osp
from tqdm import tqdm
import math
from copy import deepcopy

from basicsr.archs import build_network
from basicsr.losses import build_loss
from basicsr.metrics import calculate_metric
from basicsr.utils import get_root_logger, imwrite, tensor2img
from basicsr.utils.registry import MODEL_REGISTRY
import torch.nn.functional as F
import torch.nn as nn
from .sr_model import SRModel
from .base_model import BaseModel
from basicsr.utils.sv_blur import BatchBlur_SV
from basicsr.utils.dist_util import get_dist_info

loss_module = importlib.import_module('basicsr.losses')
metric_module = importlib.import_module('basicsr.metrics')


@MODEL_REGISTRY.register()
class KernelDModel(BaseModel):
    """Base Deblur model for single image deblur."""

    def __init__(self, opt):
        super(KernelDModel, self).__init__(opt)

        # define network
        self.net_g = build_network(deepcopy(opt['network_g']))
        # self.net_g = self.model_to_device(self.net_g)
        # self.print_network(self.net_g)

        # # load pretrained models
        # load_path = self.opt['path'].get('pretrain_network_g', None)
        # if load_path is not None:
        #     # self.load_network(self.net_g, load_path,
        #     #                   self.opt['path'].get('strict_load_g', True), param_key=self.opt['path'].get('param_key', 'params'))
        #     self.load_network(self.net_g, load_path,
        #                       self.opt['path'].get('strict_load_g', True))

        dcnn_path = self.opt['path'].get('pretrain_network_dcnn', None)
        if dcnn_path is not None:
            from basicsr.archs.kernel_single_arch import KernelSingleArch
            checkpoint = torch.load(dcnn_path, map_location=lambda storage, loc: storage)
            cfg_vqgan = deepcopy(opt['network_g'])
            cfg_vqgan.pop('type')
            cfg_vqgan.pop('initialize')
            if cfg_vqgan.get('fix_modules', False):
                cfg_vqgan.pop('fix_modules')
            # cfg_vqgan.pop('fix_modules')
            cfg_vqgan['codebook_size'] = 1024
            new_model = KernelSingleArch(**cfg_vqgan)
            new_model.load_state_dict(checkpoint['params'])
            # model.dcnn = new_model.dcnn
            # model = new_model
            
            if isinstance(self.net_g, (DataParallel, DistributedDataParallel)):
                self.net_g.module.dcnn = new_model.dcnn
            else:
                self.net_g.dcnn = new_model.dcnn
            del new_model
        
        self.net_g = self.model_to_device(self.net_g)
        self.print_network(self.net_g)

        # load pretrained models
        load_path = self.opt['path'].get('pretrain_network_g', None)
        if load_path is not None:
            # self.load_network(self.net_g, load_path,
            #                   self.opt['path'].get('strict_load_g', True), param_key=self.opt['path'].get('param_key', 'params'))
            self.load_network(self.net_g, load_path,
                              self.opt['path'].get('strict_load_g', True))

        if self.is_train:
            self.init_training_settings()

        self.scale = int(opt['scale'])
        self.degradation_model = DegradationModel(kernel_size=19)
        self.degradation_model = nn.DataParallel(self.degradation_model, device_ids=[torch.cuda.current_device()])

    def init_training_settings(self):
        

        self.net_g.train()
        train_opt = self.opt['train']

        self.scale_adaptive_gan_weight = train_opt.get('scale_adaptive_gan_weight', 0.8)
        # define network net_d
        self.net_d = build_network(self.opt['network_d'])
        self.net_d = self.model_to_device(self.net_d)
        self.print_network(self.net_d)
        load_path = self.opt['path'].get('pretrain_network_d', None)
        if load_path is not None:
            self.load_network(self.net_d, load_path, self.opt['path'].get('strict_load_d', True))
        self.net_d.train()
        self.net_d_iters = train_opt.get('net_d_iters', 1)
        self.net_d_start_iter = train_opt.get('net_d_start_iter', 0)

        # define losses
        if train_opt.get('pixel_opt'):
            pixel_type = train_opt['pixel_opt'].pop('type')
            cri_pix_cls = getattr(loss_module, pixel_type)
            self.cri_pix = cri_pix_cls(**train_opt['pixel_opt']).to(
                self.device)
        else:
            self.cri_pix = None

        if train_opt.get('reblur_opt'):
            reblur_type = train_opt['reblur_opt'].pop('type')
            cri_reblur_cls = getattr(loss_module, reblur_type)
            self.cri_reblur = cri_reblur_cls(**train_opt['reblur_opt']).to(
                self.device)
            # self.reblur_weight = train_opt['reblur_opt'].pop('loss_weight')
        else:
            self.cri_reblur = None

        if train_opt.get('perceptual_opt'):
            percep_type = train_opt['perceptual_opt'].pop('type')
            cri_perceptual_cls = getattr(loss_module, percep_type)
            self.cri_perceptual = cri_perceptual_cls(
                **train_opt['perceptual_opt']).to(self.device)
        else:
            self.cri_perceptual = None
        
        if train_opt.get('codebook_opt'):
            self.l_weight_codebook = train_opt['codebook_opt'].get('loss_weight', 1.0)
        else:
            self.l_weight_codebook = 1.0
        
        if train_opt.get('shape_opt'):
            shape_type = train_opt['shape_opt'].pop('type')
            cri_shape_cls = getattr(loss_module, shape_type)
            self.cri_shape = cri_shape_cls(**train_opt['shape_opt']).to(
                self.device)
        else:
            self.cri_shape = None
        
        if train_opt.get('gan_opt'):
            gan_type = train_opt['gan_opt'].pop('type')
            cri_gan_cls = getattr(loss_module, gan_type)
            self.cri_gan = cri_gan_cls(**train_opt['gan_opt']).to(self.device)

        # if self.cri_pix is None and self.cri_perceptual is None:
        #     raise ValueError('Both pixel and perceptual losses are None.')

        # set up optimizers and schedulers
        self.setup_optimizers()
        self.setup_schedulers()

    def setup_optimizers(self):
        train_opt = self.opt['train']
        optim_params = []

        for k, v in self.net_g.named_parameters():
            if v.requires_grad:
        #         if k.startswith('module.offsets') or k.startswith('module.dcns'):
        #             optim_params_lowlr.append(v)
        #         else:
                optim_params.append(v)
            # else:
            #     logger = get_root_logger()
            #     logger.warning(f'Params {k} will not be optimized.')
        # print(optim_params)
        # ratio = 0.1

        optim_type = train_opt['optim_g'].pop('type')
        if optim_type == 'Adam':
            self.optimizer_g = torch.optim.Adam([{'params': optim_params}],
                                                **train_opt['optim_g'])
        elif optim_type == 'SGD':
            self.optimizer_g = torch.optim.SGD(optim_params,
                                               **train_opt['optim_g'])
        elif optim_type == 'AdamW':
            self.optimizer_g = torch.optim.AdamW([{'params': optim_params}],
                                                **train_opt['optim_g'])
            pass
        else:
            raise NotImplementedError(
                f'optimizer {optim_type} is not supperted yet.')
        self.optimizers.append(self.optimizer_g)
         # optimizer d
        optim_type = train_opt['optim_d'].pop('type')
        self.optimizer_d = self.get_optimizer(optim_type, self.net_d.parameters(), **train_opt['optim_d'])
        self.optimizers.append(self.optimizer_d)

    def feed_data(self, data, is_val=False):
        self.lq = data['lq'].to(self.device)
        if 'gt' in data:
            self.gt = data['gt'].to(self.device)

    def grids(self):
        b, c, h, w = self.gt.size()
        self.original_size = (b, c, h, w)

        assert b == 1
        if 'crop_size_h' in self.opt['val']:
            crop_size_h = self.opt['val']['crop_size_h']
        else:
            crop_size_h = int(self.opt['val'].get('crop_size_h_ratio') * h)

        if 'crop_size_w' in self.opt['val']:
            crop_size_w = self.opt['val'].get('crop_size_w')
        else:
            crop_size_w = int(self.opt['val'].get('crop_size_w_ratio') * w)


        crop_size_h, crop_size_w = crop_size_h // self.scale * self.scale, crop_size_w // self.scale * self.scale
        #adaptive step_i, step_j
        num_row = (h - 1) // crop_size_h + 1
        num_col = (w - 1) // crop_size_w + 1

        import math
        step_j = crop_size_w if num_col == 1 else math.ceil((w - crop_size_w) / (num_col - 1) - 1e-8)
        step_i = crop_size_h if num_row == 1 else math.ceil((h - crop_size_h) / (num_row - 1) - 1e-8)

        scale = self.scale
        step_i = step_i//scale*scale
        step_j = step_j//scale*scale

        parts = []
        idxes = []

        i = 0  # 0~h-1
        last_i = False
        while i < h and not last_i:
            j = 0
            if i + crop_size_h >= h:
                i = h - crop_size_h
                last_i = True

            last_j = False
            while j < w and not last_j:
                if j + crop_size_w >= w:
                    j = w - crop_size_w
                    last_j = True
                parts.append(self.lq[:, :, i // scale :(i + crop_size_h) // scale, j // scale:(j + crop_size_w) // scale])
                idxes.append({'i': i, 'j': j})
                j = j + step_j
            i = i + step_i

        self.origin_lq = self.lq
        self.lq = torch.cat(parts, dim=0)
        self.idxes = idxes

    def grids_inverse(self):
        preds = torch.zeros(self.original_size)
        b, c, h, w = self.original_size

        count_mt = torch.zeros((b, 1, h, w))
        if 'crop_size_h' in self.opt['val']:
            crop_size_h = self.opt['val']['crop_size_h']
        else:
            crop_size_h = int(self.opt['val'].get('crop_size_h_ratio') * h)

        if 'crop_size_w' in self.opt['val']:
            crop_size_w = self.opt['val'].get('crop_size_w')
        else:
            crop_size_w = int(self.opt['val'].get('crop_size_w_ratio') * w)

        crop_size_h, crop_size_w = crop_size_h // self.scale * self.scale, crop_size_w // self.scale * self.scale

        for cnt, each_idx in enumerate(self.idxes):
            i = each_idx['i']
            j = each_idx['j']
            preds[0, :, i: i + crop_size_h, j: j + crop_size_w] += self.outs[cnt]
            count_mt[0, 0, i: i + crop_size_h, j: j + crop_size_w] += 1.

        self.output = (preds / count_mt).to(self.device)
        self.lq = self.origin_lq

    def optimize_parameters(self, current_iter, tb_logger):

        # optimize net_g
        for p in self.net_d.parameters():
            p.requires_grad = False
        self.optimizer_g.zero_grad()

        if self.opt['train'].get('mixup', False):
            self.mixup_aug()
        if current_iter % self.net_d_iters == 0:
            preds, cb_loss = self.net_g(self.lq)

            cb_loss = cb_loss * self.l_weight_codebook
            # if not isinstance(preds, list):
            #     preds = [preds]

            # self.output = preds[-1]
            self.output = preds

            l_total = 0
            loss_dict = OrderedDict()
            # pixel loss
            if self.cri_pix:
                l_pix = 0.
                l_pix += self.cri_pix(self.output, self.gt)

                # print('l pix ... ', l_pix)
                l_total += l_pix
                loss_dict['l_pix'] = l_pix

            if self.cri_reblur:
                l_reblur = 0.
                self.fake_blur = self.degradation_model(self.gt, self.output)
                l_reblur += self.cri_reblur(self.fake_blur, self.lq)

                # print('l reblur ... ', l_reblur)
                l_total += l_reblur
                loss_dict['l_reblur'] = l_reblur
            
            if self.cri_shape:
                l_shape = 0.
                l_shape += self.cri_shape(self.output)
                l_total += l_shape
                loss_dict['l_shape'] = l_shape

            # perceptual loss
            if self.cri_perceptual:
                l_percep, l_style = self.cri_perceptual(self.output, self.gt)
            #
                if l_percep is not None:
                    l_total += l_percep
                    loss_dict['l_percep'] = l_percep
                if l_style is not None:
                    l_total += l_style
                    loss_dict['l_style'] = l_style


            if current_iter > self.net_d_start_iter:
                fake_g_pred = self.net_d(self.fake_blur)
                l_g_gan = self.cri_gan(fake_g_pred, True, is_disc=False)
                # recon_loss = l_pix + l_percep
                d_weight = 1.0
                # if not self.fix_generator:
                #     last_layer = self.net_g.module.generator.blocks[-1].weight
                #     d_weight = self.calculate_adaptive_weight(recon_loss, l_g_gan, last_layer, disc_weight_max=1.0)
                # else:
                #     largest_fuse_size = self.opt['network_g']['connect_list'][-1]
                #     last_layer = self.net_g.module.fuse_convs_dict[largest_fuse_size].shift[-1].weight
                #     d_weight = self.calculate_adaptive_weight(recon_loss, l_g_gan, last_layer, disc_weight_max=1.0)
                
                d_weight *= self.scale_adaptive_gan_weight # 0.8
                loss_dict['d_weight'] = d_weight
                l_g_total += d_weight * l_g_gan
                loss_dict['l_g_gan'] = d_weight * l_g_gan

            l_total += cb_loss
            loss_dict['l_codebook'] = cb_loss
            l_total = l_total + 0. * sum(p.sum() for p in self.net_g.parameters())

            l_total.backward()
            use_grad_clip = self.opt['train'].get('use_grad_clip', True)
            if use_grad_clip:
                torch.nn.utils.clip_grad_norm_(self.net_g.parameters(), 0.01)
            self.optimizer_g.step()
        # optimize net_d
        if  current_iter > self.net_d_start_iter:
            for p in self.net_d.parameters():
                p.requires_grad = True

            self.optimizer_d.zero_grad()
            # real
            real_d_pred = self.net_d(self.lq)
            l_d_real = self.cri_gan(real_d_pred, True, is_disc=True)
            loss_dict['l_d_real'] = l_d_real
            loss_dict['out_d_real'] = torch.mean(real_d_pred.detach())
            l_d_real.backward()
            # fake
            fake_d_pred = self.net_d(self.fake_blur.detach())
            l_d_fake = self.cri_gan(fake_d_pred, False, is_disc=True)
            loss_dict['l_d_fake'] = l_d_fake
            loss_dict['out_d_fake'] = torch.mean(fake_d_pred.detach())
            l_d_fake.backward()

            self.optimizer_d.step()

        self.log_dict = self.reduce_loss_dict(loss_dict)

    def test(self):
        self.net_g.eval()
        with torch.no_grad():
            n = len(self.lq)    # tensor[1, 3, H, W]
            outs = []
            m = self.opt['val'].get('max_minibatch', n)
            i = 0
            while i < n:
                j = i + m
                if j >= n:
                    j = n
                pred, cb_loss = self.net_g(self.lq[i:j])
                # if isinstance(pred, list):
                #     pred = pred[-1]
                outs.append(pred.detach().cpu())
                i = j

            self.output = torch.cat(outs, dim=0)  # tensor[1, 3, H, W]
        self.net_g.train()

    def test_crop9(self):
        self.net_g.eval()

        with torch.no_grad():
            N, C, H, W = self.lq.shape
            h, w = math.ceil(H / 3), math.ceil(W / 3)
            rf = 30
            imTL = self.net_g(self.lq[:, :, 0:h + rf, 0:w + rf])[:, :, 0:h, 0:w]
            imML = self.net_g(self.lq[:, :, h - rf:2 * h + rf, 0:w + rf])[:, :, rf:(rf + h), 0:w]
            imBL = self.net_g(self.lq[:, :, 2 * h - rf:, 0:w + rf])[:, :, rf:, 0:w]
            imTM = self.net_g(self.lq[:, :, 0:h + rf, w - rf:2 * w + rf])[:, :, 0:h, rf:(rf + w)]
            imMM = self.net_g(self.lq[:, :, h - rf:2 * h + rf, w - rf:2 * w + rf])[:, :, rf:(rf + h), rf:(rf + w)]
            imBM = self.net_g(self.lq[:, :, 2 * h - rf:, w - rf:2 * w + rf])[:, :, rf:, rf:(rf + w)]
            imTR = self.net_g(self.lq[:, :, 0:h + rf, 2 * w - rf:])[:, :, 0:h, rf:]
            imMR = self.net_g(self.lq[:, :, h - rf:2 * h + rf, 2 * w - rf:])[:, :, rf:(rf + h), rf:]
            imBR = self.net_g(self.lq[:, :, 2 * h - rf:, 2 * w - rf:])[:, :, rf:, rf:]

            imT = torch.cat((imTL, imTM, imTR), 3)
            imM = torch.cat((imML, imMM, imMR), 3)
            imB = torch.cat((imBL, imBM, imBR), 3)
            output_cat = torch.cat((imT, imM, imB), 2)

            self.output = output_cat
        self.net_g.train()


    def dist_validation(self, dataloader, current_iter, tb_logger, save_img, rgb2bgr=True, use_image=True):
        dataset_name = dataloader.dataset.opt['name']
        with_metrics = self.opt['val'].get('metrics') is not None
        if with_metrics:
            self.metric_results = {
                metric: 0
                for metric in self.opt['val']['metrics'].keys()
            }
            self.metric_results_gt = {
                metric: 0
                for metric in self.opt['val']['metrics'].keys()
            }

        rank, world_size = get_dist_info()
        if rank == 0:
            pbar = tqdm(total=len(dataloader), unit='image')

        cnt = 0

        for idx, val_data in enumerate(dataloader):
            if idx % world_size != rank:
                continue

            img_name = osp.splitext(osp.basename(val_data['lq_path'][0]))[0]

            self.feed_data(val_data, is_val=True)
            # if self.opt['val'].get('grids', False):
            #     self.grids()

            # self.test_crop9()
            self.test()

            self.fake_blur = self.degradation_model(self.gt, self.output)

            # if self.opt['val'].get('grids', False):
            #     self.grids_inverse()

            visuals = self.get_current_visuals()
            lq_img = tensor2img([visuals['lq']], rgb2bgr=rgb2bgr)
            sr_img = tensor2img([visuals['result']], rgb2bgr=rgb2bgr)
            if 'gt' in visuals:
                gt_img = tensor2img([visuals['gt']], rgb2bgr=rgb2bgr)
                del self.gt

            # tentative for out of GPU memory
            # del self.lq
            del self.output
            torch.cuda.empty_cache()

            # if save_img:
            #     if sr_img.shape[2] == 6:
            #         L_img = sr_img[:, :, :3]
            #         R_img = sr_img[:, :, 3:]

            #         # visual_dir = osp.join('visual_results', dataset_name, self.opt['name'])
            #         visual_dir = osp.join(self.opt['path']['visualization'], dataset_name)

            #         imwrite(L_img, osp.join(visual_dir, f'{img_name}_L.png'))
            #         imwrite(R_img, osp.join(visual_dir, f'{img_name}_R.png'))
            #     else:
            #         if self.opt['is_train']:

            #             save_img_path = osp.join(self.opt['path']['visualization'],
            #                                      img_name,
            #                                      f'{img_name}_{current_iter}.png')

            #             save_gt_img_path = osp.join(self.opt['path']['visualization'],
            #                                      img_name,
            #                                      f'{img_name}_{current_iter}_gt.png')
            #         else:
            #             save_img_path = osp.join(
            #                 self.opt['path']['visualization'], dataset_name,
            #                 f'{img_name}.png')
            #             save_gt_img_path = osp.join(
            #                 self.opt['path']['visualization'], dataset_name,
            #                 f'{img_name}_gt.png')

            #         imwrite(sr_img, save_img_path)
            #         # imwrite(gt_img, save_gt_img_path)

            if with_metrics:
                # calculate metrics
                opt_metric = deepcopy(self.opt['val']['metrics'])
                if use_image:
                    for name, opt_ in opt_metric.items():
                        metric_type = opt_.pop('type')
                        self.metric_results[name] += getattr(
                            metric_module, metric_type)(sr_img, lq_img, **opt_)
                        self.metric_results_gt[name] += getattr(
                            metric_module, metric_type)(sr_img, gt_img, **opt_)
                else:
                    for name, opt_ in opt_metric.items():
                        metric_type = opt_.pop('type')
                        self.metric_results[name] += getattr(
                            metric_module, metric_type)(visuals['result'], visuals['lq'], **opt_)
                        self.metric_results_gt[name] += getattr(
                            metric_module, metric_type)(visuals['result'], visuals['gt'], **opt_)

            cnt += 1
            if rank == 0:
                for _ in range(world_size):
                    pbar.update(1)
                    pbar.set_description(f'Test {img_name}')
        if rank == 0:
            pbar.close()

        # current_metric = 0.
        collected_metrics = OrderedDict()
        collected_metrics_gt = OrderedDict()
        if with_metrics:
            for metric in self.metric_results.keys():
                collected_metrics[metric] = torch.tensor(self.metric_results[metric]).float().to(self.device)
            collected_metrics['cnt'] = torch.tensor(cnt).float().to(self.device)

            self.collected_metrics = collected_metrics

            for metric in self.metric_results_gt.keys():
                collected_metrics_gt[metric] = torch.tensor(self.metric_results_gt[metric]).float().to(self.device)
            collected_metrics_gt['cnt'] = torch.tensor(cnt).float().to(self.device)

            self.collected_metrics_gt = collected_metrics_gt
        
        keys = []
        metrics = []
        for name, value in self.collected_metrics.items():
            keys.append(name)
            metrics.append(value)
        metrics = torch.stack(metrics, 0)
        # torch.distributed.reduce(metrics, dst=0)
        if self.opt['rank'] == 0:
            metrics_dict = {}
            cnt = 0
            for key, metric in zip(keys, metrics):
                if key == 'cnt':
                    cnt = float(metric)
                    continue
                metrics_dict[key] = float(metric)

            for key in metrics_dict:
                metrics_dict[key] /= cnt

            self._log_validation_metric_values(current_iter, dataloader.dataset.opt['name'],
                                               tb_logger, metrics_dict, is_gt=False)
        
        keys_gt = []
        metrics_gt = []
        for name, value in self.collected_metrics_gt.items():
            keys_gt.append(name)
            metrics_gt.append(value)
        metrics_gt = torch.stack(metrics_gt, 0)
        # torch.distributed.reduce(metrics, dst=0)
        if self.opt['rank'] == 0:
            metrics_dict_gt = {}
            cnt = 0
            for key, metric in zip(keys_gt, metrics_gt):
                if key == 'cnt':
                    cnt = float(metric)
                    continue
                metrics_dict_gt[key] = float(metric)

            for key in metrics_dict_gt:
                metrics_dict_gt[key] /= cnt

            self._log_validation_metric_values(current_iter, dataloader.dataset.opt['name'],
                                               tb_logger, metrics_dict_gt, is_gt=True)
        return 0.

    def nondist_validation(self, *args, **kwargs):
        logger = get_root_logger()
        logger.warning('nondist_validation is not implemented. Run dist_validation.')
        self.dist_validation(*args, **kwargs)


    def _log_validation_metric_values(self, current_iter, dataset_name,
                                      tb_logger, metric_dict, is_gt=False):
        log_str = f'Validation {dataset_name}, \t'
        for metric, value in metric_dict.items():
            log_str += f'\t # {metric}: {value:.4f}'
        if is_gt:
            log_str += ', is gt'
        else:
            log_str += ', is lq'
        logger = get_root_logger()
        logger.info(log_str)

        log_dict = OrderedDict()
        # for name, value in loss_dict.items():
        for metric, value in metric_dict.items():
            log_dict[f'm_{metric}'] = value

        self.log_dict = log_dict

    def get_current_visuals(self):
        out_dict = OrderedDict()
        out_dict['lq'] = self.lq.detach().cpu()
        out_dict['result'] = self.fake_blur.detach().cpu()
        if hasattr(self, 'gt'):
            out_dict['gt'] = self.gt.detach().cpu()
        return out_dict

    def save(self, epoch, current_iter):
        self.save_network(self.net_g, 'net_g', current_iter)
        self.save_network(self.net_d, 'net_d', current_iter)
        self.save_training_state(epoch, current_iter)
        


class DegradationModel(nn.Module):
    def __init__(self, kernel_size=19):
        super(DegradationModel, self).__init__()
        self.blur_layer = BatchBlur_SV(l=kernel_size, padmode='replication')

    def forward(self, image, kernel):
        return self.blur_layer(image, kernel)