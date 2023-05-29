
from numba import cuda,float64

import numpy as np

import matplotlib.image as mpimg
import matplotlib.pyplot as plt

from PIL import Image
import cv2
import math
import operator

class HOG:
    def __init__(self, blockSize, cellSize, nbins, sbins, threadsperblock):
        self.blockSize       = blockSize
        self.cellSize        = cellSize
        self.nbins           = nbins
        self.sbins           = sbins
        self.threadsperblock = threadsperblock
        self.x_Dim   = self.threadsperblock[0]//self.cellSize[0]
        self.y_Dim   = self.threadsperblock[1]//self.cellSize[1]
    
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
        self.gray = gray_device
    
    @staticmethod
    @cuda.jit
    def __calc_gradient_kernel_2(input, width, height, magnitude, direction):
        row, col = cuda.grid(2)
        y = 0.0
        x = 0.0
        if (row>=height) or (col>=width):
            return
        for i in range(-1,2):
            pixel_r = row + i
            pixel_r = min(max(0, pixel_r), height - 1)
            y += input[pixel_r,col] * i

            pixel_c = col + i
            pixel_c = min(max(0, pixel_c), width - 1)
            x += input[row,pixel_c] * i

        magnitude[row, col] = math.sqrt(y**2+x**2)
        direction[row, col] = operator.mod((360 + math.atan2(y,x)*180/math.pi), 360)
        
    def __calc_gradient(self):
        blockspergrid_x = math.ceil(self.picture_array.shape[0] / self.threadsperblock[0])
        blockspergrid_y = math.ceil(self.picture_array.shape[1] / self.threadsperblock[1])
        blockspergrid   = (blockspergrid_x, blockspergrid_y)

        mag_dev            = np.zeros((self.height, self.width))
        dir_dev            = np.zeros((self.height, self.width))


        mag_device            = cuda.to_device(mag_dev)
        dir_device            = cuda.to_device(dir_dev)
        
        self.__calc_gradient_kernel_2[blockspergrid, self.threadsperblock]\
                (self.gray, self.width, self.height,mag_device, dir_device)

        return mag_device,dir_device
    
    def __calc_direc_mag(self):
        self.__gray()
        mag_device, dir_device = self.__calc_gradient()
        self.magnitude = mag_device
        self.direction = dir_device
    

    @staticmethod
    @cuda.jit()
    def __hist_kernel(direction, magnitude, cell_size, d_sbin, result_out,out_l2,x_Dim,y_Dim,nbins):    
        cur_r, cur_c  = cuda.grid(2)
        height, width = direction.shape

        idy   = int(cur_r//cell_size[0])
        idx   = int(cur_c//cell_size[1])
        sbin  = d_sbin

        # Creat 3D shared array
        shared     = cuda.shared.array(shape=(4, 4, 9), dtype=float64)
        l2         = cuda.shared.array(shape=(4, 4)   , dtype=float64)
        idy_shared = idy%x_Dim
        idx_shared = idx%y_Dim

        condition_1 = cur_r%cell_size[0]
        condition_2 = int(cur_c  %(cell_size[1]))
        if  condition_1 == 0:
            check   = condition_2 
            if check < 8:
                shared[idy_shared][idx_shared][check] = 0.0
        elif  condition_1 == 1:
            check   = condition_2 
            if check == 0:
                shared[idy_shared][idx_shared][8] = 0.0
                l2[idy_shared][idx_shared] = 0.0

        cuda.syncthreads()
        # kiểm tra
        if cur_r>=height or cur_c>= width:
            return
        thread_direction = direction[cur_r][cur_c]
        thread_mag       = magnitude[cur_r][cur_c]
        # chia lấy phần nguyên và phần dư
        quotient  = int(thread_direction//sbin)
        remainder =     thread_direction % sbin

        if remainder==0:
            cuda.atomic.add(shared, (idy_shared, idx_shared, quotient), thread_mag)

        else:
            first_bin = quotient
            second_bin     = first_bin+1
            need_to_add    = thread_mag*((second_bin*sbin - thread_direction)/sbin)
            cuda.atomic.add(shared, (idy_shared, idx_shared, first_bin), need_to_add)


            second_bin_idx = second_bin
            if second_bin > 8:
                second_bin_idx = 0
            need_to_add_2  = thread_mag*((thread_direction - first_bin*sbin)/sbin)   

            cuda.atomic.add(shared, (idy_shared, idx_shared, second_bin_idx), need_to_add_2)

        cuda.syncthreads()

        if condition_1 == 0:
            check   = condition_2
            if check < 8:
                result_out[idy][idx][check]     = shared[idy_shared][idx_shared][check]
                cuda.atomic.add(l2, (idy_shared, idx_shared), shared[idy_shared][idx_shared][check]**2)
        elif  condition_1 == 1:
            if check == 0:
                cuda.atomic.add(l2, (idy_shared, idx_shared), shared[idy_shared][idx_shared][8]**2)
                result_out[idy][idx][8]            = shared[idy_shared][idx_shared][8]
                cuda.syncthreads()
                out_l2[idy][idx]                   = l2[idy_shared][idx_shared]
        else:
            return


    
    def __all_hist(self):
        blockspergrid_x = math.ceil(self.direction.shape[0] / self.threadsperblock[0])
        blockspergrid_y = math.ceil(self.direction.shape[1] / self.threadsperblock[1])
        blockspergrid = (blockspergrid_x, blockspergrid_y)
        
        hist_dev           = np.empty([self.n_cell[0], self.n_cell[1], self.nbins],dtype = np.float64)
        l2_dev             = np.empty([self.n_cell[0], self.n_cell[1]],dtype = np.float64)
        d_cell_size        = cuda.to_device(self.cellSize)
        hist_device        = cuda.device_array_like(hist_dev)
        l2_device          = cuda.device_array_like(l2_dev)
        kernel = self.__hist_kernel
        kernel[blockspergrid, self.threadsperblock]\
                            (self.direction, self.magnitude,d_cell_size, self.sbins, hist_device,l2_device,\
                                     self.x_Dim,self.y_Dim,self.nbins)
        self.hist        = hist_device
        self.test        = hist_device.copy_to_host()
        self.l2_sum      = l2_device
    @staticmethod
    @cuda.jit()
    def __l2_kernel(l2_hist,n_block,final_l2):
        row, col  = cuda.grid(2)
        if row >= n_block[0] or col >= n_block[1]:
            return
        need_to_sum = l2_hist[row:row+2, col:col+2]
        sum_result = 0.0
        for i in range(2):
            for j in range(2):
                sum_result += need_to_sum[i][j] 
        final_l2[row][col] = math.sqrt(sum_result)
    
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
        
        blockspergrid_x = math.ceil(self.n_block[0] /  self.threadsperblock[0])
        blockspergrid_y = math.ceil(self.n_block[1] /  self.threadsperblock[1])
        blockspergrid   = (blockspergrid_x, blockspergrid_y)
                
        norm_array_size = self.n_block[0] * self.n_block[1] * self.blockSize[0] * self.blockSize[1] * self.nbins
        final_l2   = np.empty([self.n_block[0],self.n_block[1]],dtype = np.float64)
        device_l2 = cuda.device_array_like(final_l2)
        self.__l2_kernel[blockspergrid, self.threadsperblock](self.l2_sum,self.n_block,device_l2)
        self.l2 = device_l2
        

        norm_block    = np.zeros((self.n_block[0], self.n_block[1], self.blockSize[0], self.blockSize[1], self.nbins))
        normed_device = cuda.to_device(norm_block)
            
        self.__normalize_kernel[blockspergrid, self.threadsperblock] \
                                    (self.hist, self.n_block, self.blockSize, self.l2, normed_device)
        norm_block = normed_device.copy_to_host()
        
        self.HOG        = norm_block.reshape(norm_array_size)
        self.norm_block = norm_block
        return self.HOG
