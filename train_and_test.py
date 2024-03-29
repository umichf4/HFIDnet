# -*- coding: utf-8 -*-
# @Author: Brandon Han
# @Date:   2019-08-17 15:20:26
# @Last Modified by:   Brandon Han
# @Last Modified time: 2019-09-21 14:56:23

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
import visdom
import numpy as np
import os
import sys
current_dir = os.path.abspath(os.path.dirname(__file__))
sys.path.append(current_dir)
from torch.utils.data import DataLoader, TensorDataset
from net.ArbitraryShape_2 import GeneratorNet, SimulatorNet
from utils import *
from tqdm import tqdm
import cv2
import pytorch_ssim
from image_process import MetaShape


def train_generator(params):
    torch.set_default_tensor_type(torch.cuda.FloatTensor if params.cuda else torch.FloatTensor)
    # type_tensor = torch.cuda.FloatTensor if params.cuda else torch.FloatTensor
    # Device configuration
    device = torch.device('cuda:0' if params.cuda else 'cpu')
    # device = torch.device('cpu')
    print('Training GeneratorNet starts, using %s' % (device))

    # Visualization configuration
    make_figure_dir()
    viz = visdom.Visdom()
    cur_epoch_loss = None
    cur_epoch_loss_opts = {
        'title': 'Epoch Loss Trace',
        'xlabel': 'Epoch Number',
        'ylabel': 'Loss',
        'width': 1200,
        'height': 600,
        'showlegend': True,
    }

    # Data configuration
    if not (os.path.exists('data/all_ctrast.npy') and os.path.exists('data/all_gatk.npy') and
            os.path.exists('data/all_shape.npy') and os.path.exists('data/all_spec.npy')):
        data_pre_arbitrary(params.T_path)

    all_gatk = torch.from_numpy(np.load('data/all_gatk.npy')).float()
    all_spec = torch.from_numpy(np.load('data/all_spec.npy')).float()
    all_shape = torch.from_numpy(np.load('data/all_shape.npy')).float()
    all_ctrast = torch.from_numpy(np.load('data/all_ctrast.npy')).float()

    all_num = 5000
    permutation = np.random.permutation(all_num).tolist()
    all_gatk, all_spec, all_shape, all_ctrast = all_gatk[permutation],\
        all_spec[permutation, :], all_shape[permutation, :, :, :], all_ctrast[permutation, :]

    # train_gatk = all_gap[:int(all_num * params.ratio), :]
    # valid_gatk = all_gap[int(all_num * params.ratio):, :]

    train_spec = all_spec[:int(all_num * params.ratio), :]
    valid_spec = all_spec[int(all_num * params.ratio):, :]

    train_ctrast = all_ctrast[:int(all_num * params.ratio), :]
    valid_ctrast = all_ctrast[int(all_num * params.ratio):, :]

    train_shape = all_shape[:int(all_num * params.ratio), :, :, :]
    valid_shape = all_shape[int(all_num * params.ratio):, :, :, :]

    train_dataset = TensorDataset(train_spec, train_shape, train_ctrast)
    train_loader = DataLoader(dataset=train_dataset, batch_size=params.batch_size_g, shuffle=True)

    valid_dataset = TensorDataset(valid_spec, valid_shape, valid_ctrast)
    valid_loader = DataLoader(dataset=valid_dataset, batch_size=valid_spec.shape[0], shuffle=True)

    # Net configuration
    net = GeneratorNet(noise_dim=params.noise_dim, ctrast_dim=params.ctrast_dim, d=params.net_depth)
    net.weight_init(mean=0, std=0.02)

    optimizer = torch.optim.Adam(net.parameters(), lr=params.lr_g, betas=(0.5, 0.999))
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, params.step_size_g, params.gamma_g)

    simulator = SimulatorNet(spec_dim=params.spec_dim, d=params.net_depth)
    load_checkpoint('models/simulator_full_trained.pth', simulator, None)
    for param in simulator.parameters():
        param.requires_grad = False

    criterion_1 = nn.MSELoss()
    criterion_2 = pytorch_ssim.SSIM(window_size=11)
    criterion_3 = BiBinaryloss()

    train_loss_list, val_loss_list, epoch_list = [], [], []
    spec_loss, shape_loss = 0, 0

    if params.restore_from:
        load_checkpoint(params.restore_from, net, optimizer)

    net.to(device)
    simulator.to(device)
    simulator.eval()

    # Start training
    for k in range(params.epochs):
        epoch = k + 1
        epoch_list.append(epoch)

        # Train
        net.train()
        for i, data in enumerate(train_loader):

            inputs, labels, ctrasts = data
            inputs, labels, ctrasts = inputs.to(device), labels.to(device), ctrasts.to(device)
            noise = torch.rand(inputs.shape[0], params.noise_dim)

            optimizer.zero_grad()
            net.zero_grad()

            output_shapes, output_pairs = net(noise, ctrasts)
            binary_loss = criterion_3(output_shapes)
            shape_loss = 1 - criterion_2(output_shapes, labels)
            # output_shapes = binary(output_shapes)
            output_specs = simulator(output_shapes, output_pairs[:, 0], output_pairs[:, 1])
            spec_loss = criterion_1(output_specs, inputs)

            train_loss = spec_loss + shape_loss * params.alpha + binary_loss * params.beta
            train_loss.backward()
            optimizer.step()

        train_loss_list.append(train_loss)

        # Validation
        net.eval()
        val_loss = 0
        with torch.no_grad():
            for i, data in enumerate(valid_loader):
                inputs, labels, ctrasts = data
                inputs, labels, ctrasts = inputs.to(device), labels.to(device), ctrasts.to(device)
                noise = torch.rand(inputs.shape[0], params.noise_dim)

                output_shapes, output_pairs = net(noise, ctrasts)
                binary_loss = criterion_3(output_shapes)
                shape_loss = 1 - criterion_2(output_shapes, labels)
                # output_shapes = binary(output_shapes)
                output_specs = simulator(output_shapes, output_pairs[:, 0], output_pairs[:, 1])
                spec_loss = criterion_1(output_specs, inputs)
                val_loss += spec_loss + shape_loss * params.alpha + binary_loss * params.beta

        val_loss /= (i + 1)
        val_loss_list.append(val_loss)
        # with torch.no_grad():
        #     desire = [1, 1, 0.4, 0.1, 0.4, 1, 1, 1, 1, 0.4, 0.1, 0.4, 1, 1]
        #     ctrast = np.array(desire)
        #     spec = torch.from_numpy(ctrast).float().view(1, -1)
        #     noise = torch.rand(1, params.noise_dim)
        #     spec, noise = spec.to(device), noise.to(device)
        #     output_shapes, output_gaps = net(noise, spec)

        # # Save figures
        # out_img = output_shapes[0, :, :].view(64, 64).detach().cpu().numpy()
        # out_gap = int(np.rint(output_gaps[0, :].view(-1).detach().cpu().numpy() * 200 + 200))
        # shape_pred = MetaShape(out_gap)
        # shape_pred.img = np.uint8(out_img * 255)
        # shape_pred.save_polygon("results/" + str(epoch) + ".png")

        print('Epoch=%d train_loss: %.7f val_loss: %.7f spec_loss: %.7f shape_loss: %.7f binary_loss: %.7f lr: %.7f' %
              (epoch, train_loss, val_loss, spec_loss, shape_loss, binary_loss, scheduler.get_lr()[0]))

        scheduler.step()

        # Update Visualization
        if viz.check_connection():
            cur_epoch_loss = viz.line(torch.Tensor(train_loss_list), torch.Tensor(epoch_list),
                                      win=cur_epoch_loss, name='Train Loss',
                                      update=(None if cur_epoch_loss is None else 'replace'),
                                      opts=cur_epoch_loss_opts)
            cur_epoch_loss = viz.line(torch.Tensor(val_loss_list), torch.Tensor(epoch_list),
                                      win=cur_epoch_loss, name='Validation Loss',
                                      update=(None if cur_epoch_loss is None else 'replace'),
                                      opts=cur_epoch_loss_opts)

        if epoch % params.save_epoch == 0 and epoch != params.epochs:
            path = os.path.join(current_dir, params.save_model_dir)
            name = os.path.join(current_dir, params.save_model_dir, params.shape + '_Epoch' + str(epoch) + '.pth')
            save_checkpoint({'net_state_dict': net.state_dict(),
                             'optim_state_dict': optimizer.state_dict(),
                             },
                            path=path, name=name)

        if epoch == params.epochs:
            path = os.path.join(current_dir, params.save_model_dir)
            name = os.path.join(current_dir, params.save_model_dir, params.shape + '_Epoch' + str(epoch) + '_final.pth')
            save_checkpoint({'net_state_dict': net.state_dict(),
                             'optim_state_dict': optimizer.state_dict(),
                             },
                            path=path, name=name)

    print('Finished Training')


def test_generator(params):
    import matlab.engine
    eng = matlab.engine.start_matlab()
    eng.addpath(eng.genpath('matlab'))
    eng.addpath(eng.genpath('solvers'))
    # Device configuration
    device = torch.device('cuda:0' if params.cuda else 'cpu')
    # device = torch.device('cpu')
    print('Test starts, using %s' % (device))

    # Visualization configuration
    make_figure_dir()

    net = GeneratorNet(noise_dim=params.noise_dim, ctrast_dim=params.ctrast_dim, d=params.net_depth)
    if params.restore_from:
        load_checkpoint(os.path.join(current_dir, params.restore_from), net, None)

    net.to(device)
    net.eval()
    wavelength = np.linspace(400, 680, 29)
    # lucky = np.random.randint(low=int(5881 * params.ratio), high=5881)
    all_spec = np.load('data/all_spec.npy')
    all_ctrast = np.load('data/all_ctrast.npy')
    all_gap = np.load('data/all_gap.npy')
    all_shape = np.load('data/all_shape.npy')

    with torch.no_grad():
        # real_spec = all_spec[:10]
        ctrast = all_ctrast[:10]
        # desire = [1, 1, 0.4, 0.1, 0.4, 1, 1, 1, 1, 0.4, 0.1, 0.4, 1, 1]
        # ctrast = np.array(desire)
        spec = torch.from_numpy(ctrast).float().view(10, -1)
        noise = torch.rand(10, params.noise_dim)
        spec, noise = spec.to(device), noise.to(device)
        output_img, output_gap = net(noise, spec)
        for i in range(10):
            out_img = output_img[i, :, :].view(64, 64).detach().cpu().numpy()
            out_gap = int(np.rint(output_gap[i, :].view(-1).detach().cpu().numpy() * 200 + 200))
            # print(out_gap)
            shape_pred = MetaShape(out_gap)
            shape_pred.img = np.uint8(out_img * 255)
            # shape_pred.binary_polygon()
            # shape_pred.pad_boundary()
            # shape_pred.remove_small_twice()
            shape_pred.save_polygon("figures/test_output/dim/" + str(i + 10) + ".png")

        # spec_pred_TE, spec_pred_TM = RCWA_arbitrary(eng, gap=out_gap, img_path="figures/test_output/hhhh.png")
        # fake_TM = np.array(spec_pred_TM)
        # fake_TE = np.array(spec_pred_TE)
        # # plot_both_parts(wavelength, real_spec[0:29], fake_spec.squeeze(), "hhhh_result.png")
        # plot_both_parts_2(wavelength, fake_TM.squeeze(), ctrast[7:], "hhhh_result_TM.png")
        # plot_both_parts_2(wavelength, fake_TE.squeeze(), ctrast[:7], "hhhh_result_TE.png")

    print('Finished Testing \n')


def train_simulator(params):
    torch.set_default_tensor_type(torch.cuda.FloatTensor if params.cuda else torch.FloatTensor)
    # type_tensor = torch.cuda.FloatTensor if params.cuda else torch.FloatTensor
    # Device configuration
    device = torch.device('cuda:0' if params.cuda else 'cpu')
    # device = torch.device('cpu')
    print('Training SimulatorNet starts, using %s' % (device))

    # Visualization configuration
    make_figure_dir()
    viz = visdom.Visdom()
    cur_epoch_loss = None
    cur_epoch_loss_opts = {
        'title': 'Epoch Loss Trace',
        'xlabel': 'Epoch Number',
        'ylabel': 'Loss',
        'width': 1200,
        'height': 600,
        'showlegend': True,
    }

    # Data configuration
    if not (os.path.exists('data/all_ctrast.npy') and os.path.exists('data/all_gatk.npy') and
            os.path.exists('data/all_shape.npy') and os.path.exists('data/all_spec.npy')):
        data_pre_arbitrary(params.T_path)

    if not (os.path.exists('data/all_gatk_en.npy') and os.path.exists('data/all_shape_en.npy') and os.path.exists('data/all_spec_en.npy')):
        data_enhancement()

    all_gatk = torch.from_numpy(np.load('data/all_gatk_en.npy')).float()
    all_spec = torch.from_numpy(np.load('data/all_spec_en.npy')).float()
    all_shape = torch.from_numpy(np.load('data/all_shape_en.npy')).float()

    all_num = all_gatk.shape[0]
    permutation = np.random.permutation(all_num).tolist()
    all_gatk, all_spec, all_shape = all_gatk[permutation],\
        all_spec[permutation, :], all_shape[permutation, :, :, :]

    train_gatk = all_gatk[:int(all_num * params.ratio), :]
    valid_gatk = all_gatk[int(all_num * params.ratio):, :]

    train_spec = all_spec[:int(all_num * params.ratio), :]
    valid_spec = all_spec[int(all_num * params.ratio):, :]

    train_shape = all_shape[:int(all_num * params.ratio), :, :, :]
    valid_shape = all_shape[int(all_num * params.ratio):, :, :, :]

    train_dataset = TensorDataset(train_spec, train_shape, train_gatk)
    train_loader = DataLoader(dataset=train_dataset, batch_size=params.batch_size_s, shuffle=True)

    valid_dataset = TensorDataset(valid_spec, valid_shape, valid_gatk)
    valid_loader = DataLoader(dataset=valid_dataset, batch_size=valid_gatk.shape[0], shuffle=True)

    # Net configuration
    net = SimulatorNet(spec_dim=params.spec_dim, d=params.net_depth)
    net.weight_init(mean=0, std=0.02)

    optimizer = torch.optim.Adam(net.parameters(), lr=params.lr_s, betas=(0.5, 0.999))
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, params.step_size_s, params.gamma_s)

    criterion = nn.MSELoss()
    train_loss_list, val_loss_list, epoch_list = [], [], []

    if params.restore_from:
        load_checkpoint(params.restore_from, net, optimizer)

    net.to(device)

    # Start training
    for k in range(params.epochs):
        epoch = k + 1
        epoch_list.append(epoch)

        # Train
        net.train()
        for i, data in enumerate(train_loader):

            specs, shapes, pairs = data
            specs, shapes, pairs = specs.to(device), shapes.to(device), pairs.to(device)

            optimizer.zero_grad()
            net.zero_grad()

            outputs = net(shapes, pairs[:, 0], pairs[:, 1])
            train_loss = criterion(outputs, specs)
            train_loss.backward()
            optimizer.step()

        train_loss_list.append(train_loss)

        # Validation
        net.eval()
        val_loss = 0
        with torch.no_grad():
            for i, data in enumerate(valid_loader):
                specs, shapes, pairs = data
                specs, shapes, pairs = specs.to(device), shapes.to(device), pairs.to(device)

                optimizer.zero_grad()
                net.zero_grad()

                outputs = net(shapes, pairs[:, 0], pairs[:, 1])
                val_loss += criterion(outputs, specs)
        val_loss /= (i + 1)
        val_loss_list.append(val_loss)

        print('Epoch=%d train_loss: %.7f val_loss: %.7f lr: %.7f' %
              (epoch, train_loss, val_loss, scheduler.get_lr()[0]))

        scheduler.step()

        # Update Visualization
        if viz.check_connection():
            cur_epoch_loss = viz.line(torch.Tensor(train_loss_list), torch.Tensor(epoch_list),
                                      win=cur_epoch_loss, name='Train Loss',
                                      update=(None if cur_epoch_loss is None else 'replace'),
                                      opts=cur_epoch_loss_opts)
            cur_epoch_loss = viz.line(torch.Tensor(val_loss_list), torch.Tensor(epoch_list),
                                      win=cur_epoch_loss, name='Validation Loss',
                                      update=(None if cur_epoch_loss is None else 'replace'),
                                      opts=cur_epoch_loss_opts)

        if epoch % params.save_epoch == 0 and epoch != params.epochs:
            path = os.path.join(current_dir, params.save_model_dir)
            name = os.path.join(current_dir, params.save_model_dir, params.shape + '_Epoch' + str(epoch) + '.pth')
            save_checkpoint({'net_state_dict': net.state_dict(),
                             'optim_state_dict': optimizer.state_dict(),
                             },
                            path=path, name=name)

        if epoch == params.epochs:
            path = os.path.join(current_dir, params.save_model_dir)
            name = os.path.join(current_dir, params.save_model_dir, params.shape + '_Epoch' + str(epoch) + '_final.pth')
            save_checkpoint({'net_state_dict': net.state_dict(),
                             'optim_state_dict': optimizer.state_dict(),
                             },
                            path=path, name=name)

    print('Finished Training')


def test_simulator(params):
    import matlab.engine
    from image_process import MetaShape
    eng = matlab.engine.start_matlab()
    eng.addpath(eng.genpath('matlab'))
    eng.addpath(eng.genpath('solvers'))
    # Device configuration
    device = torch.device('cuda:0' if params.cuda else 'cpu')
    # device = torch.device('cpu')
    print('Test starts, using %s' % (device))

    # Visualization configuration
    make_figure_dir()

    net = SimulatorNet(spec_dim=params.spec_dim, d=params.net_depth)
    if params.restore_from:
        load_checkpoint(os.path.join(current_dir, params.restore_from), net, None)

    net.to(device)
    net.eval()
    wavelength = np.linspace(400, 680, 29)
    # lucky = np.random.randint(low=0, high=6500)
    all_spec = np.load('data/all_spec.npy')
    all_gap = np.load('data/all_gap.npy')
    # all_shape = np.load('data/all_shape.npy')

    # with torch.no_grad():
    #     real_spec = all_spec[int(lucky)]
    #     gap = all_gap[int(lucky)]
    #     img = all_shape[int(lucky)]
    #     # cv2.imwrite("hhh.png", img.reshape(64, 64) * 255)
    #     spec = torch.from_numpy(real_spec).float().view(1, -1)
    #     img = torch.from_numpy(img).float().view(1, 1, 64, 64)
    #     gap = torch.from_numpy(np.array(gap)).float().view(1, 1)

    #     output = net(img, gap)
    #     fake_spec = output.view(-1).detach().cpu().numpy()
    #     plot_both_parts(wavelength, real_spec[:29], fake_spec[:29], "hhhh_result_TE.png")
    #     plot_both_parts(wavelength, real_spec[29:], fake_spec[29:], "hhhh_result_TM.png")
    #     loss = F.mse_loss(output, spec)
    #     print(lucky, loss)
    # with torch.no_grad():
    #     real_TE, real_TM = RCWA_arbitrary(eng, gap=342, img_path="figures/test_output/2_error.png")
    #     test_img = cv2.imread("figures/test_output/2_error.png", cv2.IMREAD_GRAYSCALE)
    #     img = torch.from_numpy(test_img / 255).float().view(1, 1, 64, 64)
    #     gap = torch.from_numpy(np.array((342 - 200) / 200)).float().view(1, 1)
    #     output = net(img, gap)
    #     fake_spec = output.view(-1).detach().cpu().numpy()
    #     plot_both_parts(wavelength, np.array(real_TE).squeeze(), fake_spec[:29], "hhhh_result_TE.png")
    #     plot_both_parts(wavelength, np.array(real_TM).squeeze(), fake_spec[29:], "hhhh_result_TM.png")
    with torch.no_grad():
        real_TE, _ = RCWA_arbitrary(eng, gap=220, img_path="figures/test_output/dim/1.png")
        dim_img = cv2.imread("figures/test_output/dim/11.png", cv2.IMREAD_GRAYSCALE)
        dim_img = torch.from_numpy(dim_img / 255).float().view(1, 1, 64, 64)
        gap = torch.from_numpy(np.array((220 - 200) / 200)).float().view(1, 1)
        output = net(dim_img, gap)
        dim_spec = output.view(-1).detach().cpu().numpy()
        clear_img = cv2.imread("figures/test_output/dim/1.png", cv2.IMREAD_GRAYSCALE)
        clear_img = torch.from_numpy(clear_img / 255).float().view(1, 1, 64, 64)
        output = net(clear_img, gap)
        clear_spec = output.view(-1).detach().cpu().numpy()
        plot_triple_parts(wavelength, clear_spec[:29], dim_spec[:29], np.array(real_TE).squeeze(), 'dim_vs_clear.png')

    print('Finished Testing \n')
