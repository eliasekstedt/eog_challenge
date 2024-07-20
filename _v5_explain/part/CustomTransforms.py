#import torch
#import torchvision.transforms as transforms
from PIL import ImageDraw #, Image
import random

class CenterBlock(object):
    def __init__(self, block_chance, image_size):
        self.block_size = (int(75/256*image_size[0]), int(75/256*image_size[0]))
        self.block_chance = block_chance

    def __call__(self, img):
        if random.random() < self.block_chance:
            draw = ImageDraw.Draw(img)
            w, h = img.size
            x1 = (w - self.block_size[0]) // 2
            y1 = (h - self.block_size[1]) // 2
            x2 = x1 + self.block_size[0]
            y2 = y1 + self.block_size[1]
            draw.rectangle([x1, y1, x2, y2], fill="black")
        return img

class SegmentBlock(object):
    def __init__(self, block_chance=0.5):
        segment_side_len=32
        self.segment_side_len = segment_side_len
        self.block_chance = block_chance

    def __call__(self, img):
        assert img.size[0] == img.size[1], "image does not fit criteria: nr row pixles == nr col pixels"
        _, image_side_len = img.size
        draw = ImageDraw.Draw(img)
        for i in range(0, image_side_len, self.segment_side_len):
            for j in range(0, image_side_len, self.segment_side_len):
                if random.random() < self.block_chance:
                    draw.rectangle([i, j, i+self.segment_side_len, j+self.segment_side_len], fill="black")
        return img




"""
class SegmentBlock_0(object):
    def __init__(self, segment_side_len=32, block_chance=0.5):
        self.segment_side_len = segment_side_len
        self.block_chance = block_chance

    def __call__(self, img):
        assert img.size[0] == img.size[1], "image does not fit criteria: nr row pixles == nr col pixels"
        _, image_side_len = img.size
        draw = ImageDraw.Draw(img)
        for i in range(0, image_side_len, self.segment_side_len):
            for j in range(0, image_side_len, self.segment_side_len):
                if random.random() < self.block_chance and self.block_condition(i, j, image_side_len, self.segment_side_len):
                    draw.rectangle([i, j, i+self.segment_side_len, j+self.segment_side_len], fill="black")
        return img

    def block_condition(self, i, j, image_side_len, segment_side_len):
        region_criteria = lambda x: x < 3*segment_side_len or x >= image_side_len-3*segment_side_len
        return region_criteria(i) or region_criteria(j)
"""