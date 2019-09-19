# -*- coding: utf-8 -*-
# @Author: Brandon Han
# @Date:   2019-08-17 15:20:26
# @Last Modified by:   Brandon Han
# @Last Modified time: 2019-09-16 19:45:20
import torch
import os
import json
import matplotlib.pyplot as plt
import cmath
import numpy as np
from scipy import interpolate
import scipy.io as scio
import imageio
import cv2
from tqdm import tqdm
from scipy.optimize import curve_fit
from imutils import rotate_bound
import warnings
warnings.filterwarnings('ignore')

current_dir = os.path.abspath(os.path.dirname(__file__))
Tensor = torch.cuda.FloatTensor if torch.cuda.is_available() else torch.FloatTensor


class Params():
    """Class that loads hyperparameters from a json file.

    Example:
    ```
    params = Params(json_path)
    print(params.learning_rate)
    params.learning_rate = 0.5  # change the value of learning_rate in params
    ```
    """

    def __init__(self, json_path):
        self.update(json_path)

    def save(self, json_path):
        """Saves parameters to json file"""
        with open(json_path, 'w') as f:
            json.dump(self.__dict__, f, indent=4)

    def update(self, json_path):
        """Loads parameters from json file"""
        with open(json_path) as f:
            params = json.load(f)
            self.__dict__.update(params)

    @property
    def dict(self):
        """Gives dict-like access to Params instance by `params.dict['learning_rate']`"""
        return self.__dict__


def save_checkpoint(state, path, name):
    if not os.path.exists(path):
        print("Checkpoint Directory does not exist! Making directory {}".format(path))
        os.mkdir(path)

    torch.save(state, name)

    print('Model saved')


def load_checkpoint(path, net, optimizer):
    if not os.path.exists(path):
        raise("File doesn't exist {}".format(path))

    if torch.cuda.is_available():
        state = torch.load(path, map_location='cuda:0')
    else:
        state = torch.load(path, map_location='cpu')
    net.load_state_dict(state['net_state_dict'])
    if optimizer is not None:
        optimizer.load_state_dict(state['optim_state_dict'])

    print('Model loaded')


def interploate(org_data, points=1000):
    org = np.linspace(400, 680, len(org_data))
    new = np.linspace(400, 680, points)
    inter_func = interpolate.interp1d(org, org_data, kind='cubic')
    return inter_func(new)


def make_figure_dir():
    os.makedirs(current_dir + '/figures/loss_curves', exist_ok=True)
    os.makedirs(current_dir + '/figures/test_output', exist_ok=True)


def plot_single_part(wavelength, spectrum, name, legend='spectrum', interpolate=True):
    save_dir = os.path.join(current_dir, 'figures/test_output', name)
    plt.figure()
    plt.plot(wavelength, spectrum, 'ob')
    plt.grid()
    if interpolate:
        new_spectrum = interploate(spectrum)
        new_wavelength = interploate(wavelength)
        plt.plot(new_wavelength, new_spectrum, '-b')
    plt.title(legend)
    plt.xlabel('Wavelength (nm)')
    plt.ylabel(legend)
    plt.ylim((0, 1))
    plt.savefig(save_dir)
    plt.close()


def plot_triple_parts(wavelength, clear, dim, real, name, interpolate=True):
    save_dir = os.path.join(current_dir, 'figures/test_output', name)
    plt.figure()
    plt.plot(wavelength, real, 'ob')
    plt.plot(wavelength, clear, 'or')
    plt.plot(wavelength, dim, 'og')
    plt.grid()
    if interpolate:
        new_real = interploate(real)
        new_clear = interploate(clear)
        new_dim = interploate(dim)
        new_wavelength = interploate(wavelength)
        plt.plot(new_wavelength, new_real, '-b', label='RCWA Ground Truth')
        plt.plot(new_wavelength, new_clear, '-r', label='Simulator Clear')
        plt.plot(new_wavelength, new_dim, '-g', label='Simulator Dim')
    plt.title('Comparison of inputs are binary or not')
    plt.legend()
    plt.xlabel('Wavelength (nm)')
    plt.ylabel('Transimttance')
    plt.ylim((0, 1))
    plt.savefig(save_dir)
    plt.close()


def plot_both_parts(wavelength, real, fake, name, legend='Real and Fake', interpolate=True):

    color_left = 'blue'
    color_right = 'red'
    save_dir = os.path.join(current_dir, 'figures/test_output', name)

    fig, ax1 = plt.subplots()

    ax1.set_xlabel('Wavelength (nm)')
    ax1.set_ylabel('Real', color=color_left)
    ax1.plot(wavelength, real, 'o', color=color_left, label='Real')
    if interpolate:
        new_real = interploate(real)
        new_wavelength = interploate(wavelength)
        ax1.plot(new_wavelength, new_real, color=color_left)
    ax1.legend(loc='upper left')
    ax1.tick_params(axis='y', labelcolor=color_left)
    ax1.grid()
    plt.ylim((0, 1))

    ax2 = ax1.twinx()  # instantiate a second axes that shares the same x-axis
    ax2.set_ylabel('Fake', color=color_right)  # we already handled the x-label with ax1
    ax2.plot(wavelength, fake, 'o', color=color_right, label='Fake')
    if interpolate:
        new_fake = interploate(fake)
        ax2.plot(new_wavelength, new_fake, color=color_right)
    ax2.legend(loc='upper right')
    ax2.tick_params(axis='y', labelcolor=color_right)
    ax2.spines['left'].set_color(color_left)
    ax2.spines['right'].set_color(color_right)
    plt.ylim((0, 1))
    plt.title(legend)

    fig.tight_layout()  # otherwise the right y-label is slightly clipped
    plt.savefig(save_dir)


def plot_both_parts_2(wavelength, real, cv, name, legend='Spectrum and Contrast Vector', interpolate=True):

    color_left = 'blue'
    color_right = 'red'
    save_dir = os.path.join(current_dir, 'figures/test_output', name)

    fig, ax1 = plt.subplots()

    ax1.set_xlabel('Wavelength (nm)')
    ax1.set_ylabel('Transimttance', color=color_left)
    ax1.plot(wavelength, real, 'o', color=color_left, label='Spectrum')
    if interpolate:
        new_real = interploate(real)
        new_wavelength = interploate(wavelength)
        ax1.plot(new_wavelength, new_real, color=color_left)
    # ax1.legend()
    ax1.tick_params(axis='y', labelcolor=color_left)
    ax1.grid()
    plt.ylim((0, 1))

    ax2 = ax1.twinx()  # instantiate a second axes that shares the same x-axis
    ax2.set_ylabel('Contrast', color=color_right)  # we already handled the x-label with ax1
    ax2.step(np.linspace(400, 680, 8), np.append(cv, cv[-1]), where='post', color=color_right, label='Contrast Vector')
    # ax2.legend()
    ax2.tick_params(axis='y', labelcolor=color_right)
    ax2.spines['left'].set_color(color_left)
    ax2.spines['right'].set_color(color_right)
    # plt.ylim((0, 1))
    plt.title(legend)

    fig.tight_layout()  # otherwise the right y-label is slightly clipped
    plt.savefig(save_dir)


def make_gif(path):
    images = []
    filenames = sorted((fn for fn in os.listdir(path) if fn.endswith('.png')))
    for filename in filenames:
        images.append(imageio.imread(os.path.join(path, filename)))
    imageio.mimsave('tendency.gif', images, duration=0.1)


def rename_all_files(path):
    filelist = os.listdir(path)
    count = 0
    for file in filelist:
        print(file)
    for file in filelist:
        Olddir = os.path.join(path, file)
        if os.path.isdir(Olddir):
            rename_all_files(Olddir)
            continue
        filetype = os.path.splitext(file)[1]
        filename = os.path.splitext(file)[0]
        Newdir = os.path.join(path, str(int(filename)).zfill(4) + filetype)
        os.rename(Olddir, Newdir)
        count += 1


def load_mat(path):
    variables = scio.whosmat(path)
    target = variables[0][0]
    data = scio.loadmat(path)
    TT_array = data[target]
    TT_list = TT_array.tolist()
    return TT_list, TT_array


def RCWA_arbitrary(eng, gap, img_path, thickness, acc=5):
    import matlab.engine
    gap = matlab.double([gap])
    acc = matlab.double([acc])
    thick = matlab.double([thickness])
    spec = eng.cal_spec(gap, thick, acc, img_path, nargout=2)
    spec_TE, spec_TM = spec
    return spec_TE, spec_TM


def normalization(data):
    _range = np.max(data) - np.min(data)
    return (data - np.min(data)) / _range


def keep_range(data, low=0, high=1):
    index = np.where(data > high)
    data[index] = high
    index = np.where(data < low)
    data[index] = low

    return data


def cal_contrast(wavelength, spec, spec_start, spec_end):
    spec_range_in = spec[np.argwhere((wavelength <= spec_end) & (wavelength >= spec_start))]
    sepc_range_out = spec[np.argwhere((wavelength > spec_end) | (wavelength < spec_start))]
    contrast = np.max(spec_range_in) / np.max(sepc_range_out)
    return contrast


def cal_contrast_vector(spec):
    wavelength = np.linspace(400, 680, 29)
    contrast_vector = np.zeros(7)
    for i in range(len(contrast_vector)):
        contrast_vector[i] = cal_contrast(wavelength, spec, 400 + i * 40, 440 + i * 40)
    return contrast_vector


def plot_possible_spec(spec):
    # import copy
    # temp = copy.copy(spec)
    # min_index_1 = np.argmin(temp, axis=1)
    # temp[min_index_1] = 2
    # min_index_2 = np.argmin(temp, axis=1)
    # temp[min_index_2] = 2
    # min_index_3 = np.argmin(temp, axis=1)
    # temp[min_index_3] = 2
    # min_index_4 = np.argmin(temp, axis=1)
    # temp[min_index_4] = 2
    # min_index_5 = np.argmin(temp, axis=1)
    # temp[min_index_5] = 2
    # min_index_6 = np.argmin(temp, axis=1)
    # min_sort = np.lexsort((min_index_6, min_index_5, min_index_4, min_index_3, min_index_2, min_index_1))
    # TE_spec = spec[min_sort, :]
    # max_index = np.argmax(spec, axis=1)
    # min_index = np.argmin(spec, axis=1)
    # sort = np.lexsort((max_index[::-1], min_index))
    # TE_spec = spec[sort, :]
    mean = np.mean(spec, axis=1)
    min_index_0 = np.argsort(mean)
    min_index_1 = np.argmin(spec[:, 15:], axis=1)
    sort = np.lexsort((min_index_0, min_index_1))
    TE_spec = spec[sort, :]

    wavelength = np.linspace(400, 680, 29)
    TE_spec = cv2.resize(src=TE_spec, dsize=(1000, 1881), interpolation=cv2.INTER_CUBIC)

    plt.figure()
    plt.pcolor(TE_spec, cmap=plt.cm.jet)
    plt.xlabel('Wavelength (nm)')
    # plt.xlabel('Index of elements')
    plt.ylabel('Index of Devices')
    plt.title('Possible Contrast Distribution (TE)')
    # plt.title('Gaussian Amplitude after Decomposition')
    # plt.title('Possible Spectrums of Arbitrary Shapes (' + title + ')')
    # plt.title(r'Possible Spectrums of Square Shape ($T_iO_2$)')
    plt.xticks(np.arange(len(wavelength), step=4), np.uint16(wavelength[::4]))
    # plt.xticks(np.arange(8), np.uint16(wavelength[::4]))
    plt.yticks([])
    cb = plt.colorbar()
    cb.ax.set_ylabel('Contrast')
    plt.show()


def data_pre_arbitrary(T_path):
    print("Waiting for Data Preparation...")
    _, TT_array = load_mat(T_path)
    all_num = TT_array.shape[0]
    all_name_np = TT_array[:, 0]
    all_gap_np = (TT_array[:, 1] - 200) / 200
    all_thk_np = (TT_array[:, 2] - 200) / 250
    all_spec_np = TT_array[:, 3:]
    all_shape_np = np.zeros((all_num, 1, 64, 64))
    all_ctrast_np = np.zeros((all_num, 14))
    with tqdm(total=all_num, ncols=70) as t:
        delete_list = []
        for i in range(all_num):
            # shape
            find = False
            name = str(int(all_name_np[i]))
            filelist = os.listdir('polygon')
            for file in filelist:
                if name == str(int(file.split('_')[0])):
                    img_np = cv2.imread('polygon/' + file, cv2.IMREAD_GRAYSCALE)
                    all_shape_np[i, 0, :, :] = img_np / 255
                    find = True
                    break
                else:
                    continue
            if not find:
                print("NO match with " + str(i) + ", it will be deleted later!")
                delete_list.append(i)
            # calculate contrast
            all_ctrast_np[i, :] = np.concatenate((cal_contrast_vector(
                all_spec_np[i, :29]), cal_contrast_vector(all_spec_np[i, 29:])))
            t.update()
    # delete error guys
    all_name_np = np.delete(all_name_np, delete_list, axis=0)
    all_gap_np = np.delete(all_gap_np, delete_list, axis=0)
    all_thk_np = np.delete(all_thk_np, delete_list, axis=0)
    all_spec_np = np.delete(all_spec_np, delete_list, axis=0)
    all_shape_np = np.delete(all_shape_np, delete_list, axis=0)
    all_ctrast_np = np.delete(all_ctrast_np, delete_list, axis=0)
    all_gatk_np = np.concatenate((all_gap_np, all_thk_np[:, 1]), axis=0)
    np.save('data/all_gatk.npy', all_gatk_np)
    np.save('data/all_spec.npy', all_spec_np)
    np.save('data/all_shape.npy', all_shape_np)
    np.save('data/all_ctrast.npy', all_ctrast_np)
    print("Data Preparation Done! All get {} elements!".format(all_num - len(delete_list)))


def data_enhancement():
    print("Waiting for Data Enhancement...")
    all_gatk_org = np.load('data/all_gatk.npy')
    all_spec_org = np.load('data/all_spec.npy')
    all_shape_org = np.load('data/all_shape.npy')
    all_spec_90_270 = np.zeros_like(all_spec_org)
    all_shape_90, all_shape_270, all_shape_180 = np.zeros_like(
        all_shape_org), np.zeros_like(all_shape_org), np.zeros_like(all_shape_org)
    with tqdm(total=all_gatk_org.shape[0], ncols=70) as t:
        for i in range(all_gatk_org.shape[0]):
            all_spec_90_270[i, :] = np.concatenate((all_spec_org[i, 29:], all_spec_org[i, :29]))
            all_shape_90[i, 0, :, :] = rotate_bound(all_shape_org[i, 0, :, :], 90)
            all_shape_180[i, 0, :, :] = rotate_bound(all_shape_org[i, 0, :, :], 180)
            all_shape_270[i, 0, :, :] = rotate_bound(all_shape_org[i, 0, :, :], 270)
            t.update()
    all_gatk_en = np.concatenate((all_gatk_org, all_gatk_org, all_gatk_org, all_gatk_org), axis=0)
    all_spec_en = np.concatenate((all_spec_org, all_spec_90_270, all_spec_org, all_spec_90_270), axis=0)
    all_shape_en = np.concatenate((all_shape_org, all_shape_90, all_shape_180, all_shape_270), axis=0)
    np.save('data/all_gatk_en.npy', all_gatk_en)
    np.save('data/all_spec_en.npy', all_spec_en)
    np.save('data/all_shape_en.npy', all_shape_en)
    print('Data Enhancement Done!')


def binary(output_shapes_org):
    mask_1 = output_shapes_org > 0.5
    mask_0 = output_shapes_org <= 0.5
    output_shapes_new = output_shapes_org.masked_fill(mask_1, 1)
    output_shapes_org = output_shapes_new.masked_fill(mask_0, 0)
    return output_shapes_org


if __name__ == '__main__':
    _, spec = load_mat('data/shape_spec_5000.mat')
    plot_possible_spec(spec[:, 32:])
