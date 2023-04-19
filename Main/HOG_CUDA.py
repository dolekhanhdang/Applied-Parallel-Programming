
from numba import cuda

import numpy as np

import matplotlib.image as mpimg
import matplotlib.pyplot as plt

from PIL import Image
import cv2
import math
import operator

class HOG_CUDA:
    def __init__(self, blockSize, cellSize, nbins, sbins, threadsperblock):
        self.blockSize       = blockSize
        self.cellSize        = cellSize
        self.nbins           = nbins
        self.sbins           = sbins
        self.threadsperblock = threadsperblock
    
    @staticmethod
    @cuda.jit
    def __gray_kernel(input, width, height, channel, gray):
        row, col = cuda.grid(2)
        if row >= height or col >= width or channel != 3:
            return
        rgb = input[row][col]
        gray[row][col] = 0.299*rgb[0] + 0.587*rgb[1] + 0.114*rgb[2]
    
    def __gray(self):
        picture = self.picture_array
        # Memory Allocation
        blockspergrid_x = math.ceil(self.picture_array.shape[0] / self.threadsperblock[0])
        blockspergrid_y = math.ceil(self.picture_array.shape[1] / self.threadsperblock[1])
        blockspergrid = (blockspergrid_x, blockspergrid_y)
        gray_dev   = np.empty([self.height, self.width],dtype = float)
        input_dev   = cuda.to_device(self.picture_array)
        gray_device = cuda.device_array_like(gray_dev)
        kernel = self.__gray_kernel
        kernel[blockspergrid, self.threadsperblock](input_dev, self.width, self.height, self.channel, gray_device)
        gray_dev = gray_device.copy_to_host()
        self.gray = gray_dev
    
    @staticmethod
    @cuda.jit
    def __calc_gradient_kernel(input, width, height, output_x, output_y):
        row, col = cuda.grid(2)
        if (row>=height) or (col>=width):
            return
        for i in range(-1,2):
            pixel_r = row + i
            pixel_r = min(max(0, pixel_r), height - 1)
            output_y[row, col] += input[pixel_r,col] * i

            pixel_c = col + i
            pixel_c = min(max(0, pixel_c), width - 1)
            output_x[row,col] += input[row,pixel_c] * i
        
    def __calc_gradient(self):
        blockspergrid_x = math.ceil(self.picture_array.shape[0] / self.threadsperblock[0])
        blockspergrid_y = math.ceil(self.picture_array.shape[1] / self.threadsperblock[1])
        blockspergrid = (blockspergrid_x, blockspergrid_y)
        gradient_x_dev = np.zeros((self.height, self.width))
        gradient_y_dev = np.zeros((self.height, self.width))

        gradient_x_device = cuda.to_device(gradient_x_dev)
        gradient_y_device = cuda.to_device(gradient_y_dev)
        gray_device       = cuda.to_device(self.gray)
        self.__calc_gradient_kernel[blockspergrid, self.threadsperblock]\
                (gray_device, self.width, self.height, gradient_x_device, gradient_y_device)

        gradient_x_dev = gradient_x_device.copy_to_host()
        gradient_y_dev = gradient_y_device.copy_to_host()
        return gradient_x_dev, gradient_y_dev
    
    def __calc_direc_mag(self):
        self.__gray()
        gradient_x, gradient_y = self.__calc_gradient()
        self.magnitude = np.sqrt(np.square(gradient_x)+np.square(gradient_y))
        self.direction = np.mod(np.add(360, np.rad2deg(np.arctan2(np.array(gradient_y), np.array(gradient_x)))), 360)
    
    @staticmethod
    @cuda.jit
    def __hist_kernel(direction, magnitude, width, height, sbin, cell_size, result_out):    
        cur_r, cur_c  = cuda.grid(2)

        idy   = int(cur_r//cell_size[0])
        idx   = int(cur_c//cell_size[1])
        # kiểm tra
        if cur_r>=height or cur_c>= width:
            return 
        thread_direction = direction[cur_r][cur_c]
        thread_mag       = magnitude[cur_r][cur_c]
        # chia lấy phần nguyên và phần dư
        quotient  = int(thread_direction//sbin)
        remainder =     thread_direction % sbin

        if remainder==0:
            cuda.atomic.add(result_out, (idy, idx, quotient), thread_mag)
        else:
            first_bin = quotient

            second_bin   = first_bin+1
            need_to_add    = thread_mag*((second_bin*sbin - thread_direction)/(second_bin*sbin - first_bin*sbin))
            cuda.atomic.add(result_out, (idy, idx, first_bin), need_to_add)

            second_bin_idx = second_bin
            if second_bin > 8:
                second_bin_idx = 0
            need_to_add_2  = thread_mag*((thread_direction - first_bin*sbin)/(second_bin*sbin - first_bin*sbin))   
            cuda.atomic.add(result_out, (idy, idx, second_bin_idx), need_to_add_2)
    
    def __all_hist(self):
        blockspergrid_x = math.ceil(self.picture_array.shape[0] / self.threadsperblock[0])
        blockspergrid_y = math.ceil(self.picture_array.shape[1] / self.threadsperblock[1])
        blockspergrid = (blockspergrid_x, blockspergrid_y)
        hist_dev    = np.empty([self.n_cell[0], self.n_cell[1], self.nbins],dtype = np.float64)
        d_direction = cuda.to_device(self.direction)
        d_magnitude = cuda.to_device(self.magnitude)
        d_cell_size = cuda.to_device(self.cellSize)
        hist_device = cuda.device_array_like(hist_dev)
        kernel = self.__hist_kernel
        kernel[blockspergrid, self.threadsperblock]\
                            (d_direction, d_magnitude, self.width, self.height, self.sbins, d_cell_size, hist_device)
        self.hist = hist_device.copy_to_host()
    
    @staticmethod
    @cuda.jit()
    def __normalize_kernel(hist, n_block, block_size, l2, normed):
        row, col  = cuda.grid(2)
        if row >= n_block[0] or col >= n_block[1]:
            return
        for y in range(block_size[0]):
            for x in range(block_size[1]):
                for i in range(9):
                    normed[row][col][y][x][i] = hist[row + y][col + x][i]/(l2[row][col]+1)
        
    def compute_HOG(self, picture):
        
        self.picture_array = picture
        self.height, self.width, self.channel = self.picture_array.shape
        self.n_cell  = (self.height//self.cellSize[0], self.width//self.cellSize[1])
        self.n_block = (self.n_cell[0] - self.blockSize[0] + 1, self.n_cell[1] - self.blockSize[1] + 1)
        
        self.__calc_direc_mag()
        self.__all_hist()
        
        norm_array_size = self.n_block[0] * self.n_block[1] * self.blockSize[0] * self.blockSize[1] * self.nbins
        l2 = np.empty(self.n_block)
        
        for i in range(self.n_block[0]):
            for j in range(self.n_block[1]):
                l2[i][j] = math.sqrt(np.sum(np.square(self.hist[i:i+2, j:j+2])))
                
        norm_block    = np.zeros((self.n_block[0], self.n_block[1], self.blockSize[0], self.blockSize[1], self.nbins))
        normed_device = cuda.to_device(norm_block)
        l2_dev        = cuda.to_device(l2)
        hist_device   = cuda.to_device(self.hist)
        n_block_device       = cuda.to_device(self.n_block)
        
        blockspergrid_x = math.ceil(self.picture_array.shape[0] / self.blockSize[0])
        blockspergrid_y = math.ceil(self.picture_array.shape[1] / self.blockSize[1])
        blockspergrid = (blockspergrid_x, blockspergrid_y)
        
        self.__normalize_kernel[blockspergrid, self.threadsperblock] \
                                    (hist_device, n_block_device, self.blockSize, l2_dev, normed_device)
        norm_block = normed_device.copy_to_host()
        
        self.HOG        = norm_block.flatten()
        self.norm_block = norm_block
        return self.HOG
