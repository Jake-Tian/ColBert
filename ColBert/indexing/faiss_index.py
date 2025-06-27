import sys
import time
import math
import faiss
import torch
import numpy as np

from utils import print_message, grouper

class FaissIndex():
    def __init__(self, dim, partitions):
        self.dim = dim
        self.partitions = partitions
        self.quantizer, self.index = self._create_index()
        self.offset = 0

    def _create_index(self):
        """ Create a FAISS index with the specified dimensions and partitions."""

        # A flat index for L2 distance is used as the quantizer.
        quantizer = faiss.IndexFlatL2(self.dim)  # faiss.IndexHNSWFlat(dim, 32)

        # FAISS inverted file index with Product Quantization.
        # Partitions the embedding space into P (partitions) clusters.
        # To improve memory efficiency, every embedding is divided into 16 subvectors, each of size dim/16.
        # Each subvector is quantized separately using 8 bits per subvector.
        index = faiss.IndexIVFPQ(quantizer, self.dim, self.partitions, 16, 8)

        return quantizer, index

    def train(self, train_data):
        """ Train the FAISS index with the provided training data.
            1. Clustering: Find the centroids of the clusters.
            2. Product Quantization: Learn the quantization of each subvector.

            Input: 
                train_data (num_samples, dim) - document embeddings."""
        
        # TODO: move the index to GPU 
        s = time.time()
        self.index.train(train_data)
        print(time.time() - s)

    def add(self, data):
        """ Assigning data to its nearest cluster based on the quantizer,
            and adding it to the index"""
        
        print_message(f"Add data with shape {data.shape} (offset = {self.offset})..")
        self.index.add(data)
        self.offset += data.shape[0]

    def save(self, output_path):
        
        print_message(f"Writing index to {output_path} ...")
        self.index.nprobe = 10  # just a default
        faiss.write_index(self.index, output_path)
