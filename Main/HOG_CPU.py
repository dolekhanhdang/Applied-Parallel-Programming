
from numba import cuda

import numpy as np

import matplotlib.image as mpimg
import matplotlib.pyplot as plt

from PIL import Image
import cv2
import math
import operator
class HOG_CPU:
    def __init__(self, blockSize, cellSize, nbins, sbins):
        self.blockSize     = blockSize
        self.cellSize      = cellSize
        self.nbins         = nbins
        self.sbins         = sbins
    def __gray(self):
        picture = self.picture_array
        self.gray = 0.299*picture[:,:,0]+0.587*picture[:,:,1]+0.114*picture[:,:,2]
    def __calc_gradient(self):
        input_   = self.gray
        output_x = np.zeros((self.height,self.width))
        output_y = np.zeros((self.height,self.width))

        for r in range(self.height):
            for c in range(self.width):

                for i in range(-1,2):
                    pixel_r = r + i
                    pixel_r = min(max(0, pixel_r), self.height - 1)
                    output_y[r,c] += input_[pixel_r,c] * i

                    pixel_c = c + i
                    pixel_c = min(max(0, pixel_c), self.width - 1)
                    output_x[r,c] += input_[r,pixel_c] * i
        return output_x, output_y
    def __calc_direc_mag(self):
        self.__gray()
        gradient_x, gradient_y = self.__calc_gradient()
        self.magnitude = np.sqrt(np.square(gradient_x)+np.square(gradient_y))
        self.direction = np.mod(np.add(360, np.rad2deg(np.arctan2(np.array(gradient_y), np.array(gradient_x)))), 360)
    def __cell_hist(self,idx, idy):
        output = np.zeros(9) # output = [0,0,0,0,0,0,0,0,0]

        # duyệt qua kích thước cell theo chiều cao
        for r in range(self.cellSize[0]):
          # duyệt qua kích thước cell theo chiều rộng
            for c in range(self.cellSize[1]):
              # cột và dòng hiện tại trong ảnh
                cur_r = idy*self.cellSize[0] + r
                cur_c = idx*self.cellSize[1] + c
                # kiểm tra
                if cur_r>=self.height or cur_c >= self.width:
                    break

                # chia lấy phần nguyên và phần dư
                quotient = int(self.direction[cur_r][cur_c]//self.sbins)
                remainder = self.direction[cur_r][cur_c] % self.sbins

                if remainder==0:
                    output[quotient] += self.magnitude[cur_r][cur_c]
                else:
                    first_bin = quotient

                    second_bin = first_bin+1

                    output[first_bin] += self.magnitude[cur_r][cur_c]*\
                        ((second_bin*self.sbins - self.direction[cur_r][cur_c])/ \
                                         (second_bin*self.sbins - first_bin*self.sbins))

                    second_bin_idx = second_bin
                    if second_bin > 8:
                        second_bin_idx = 0
                    output[second_bin_idx] += self.magnitude[cur_r][cur_c]*\
                        ((self.direction[cur_r][cur_c] - first_bin*self.sbins)/(second_bin*self.sbins - first_bin*self.sbins))
        return output

    def __all_hist(self):
        hist = []
        for y in range(0,self.n_cell[0]):
            row = []
            for x in range(0,self.n_cell[1]):
                output = self.__cell_hist(x,y)
                row.append(output)
            hist.append(row)
        self.hist = np.array(hist)

        
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
        norm_block = np.zeros((self.n_block[0], self.n_block[1], self.blockSize[0], self.blockSize[1], self.nbins))
        for y in range(self.n_block[0]):
            for x in range(self.n_block[1]):
                out = self.hist[y: y + self.blockSize[0], x: x + self.blockSize[1]] / (l2[y][x] + 1)
                norm_block[y][x] = out
        self.HOG        = norm_block.flatten()
        self.norm_block = norm_block
        return self.HOG
