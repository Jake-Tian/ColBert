import os
import math
import faiss
import torch
import numpy as np

import threading
import queue

from utils import print_message, grouper
from faiss_index import FaissIndex


def get_faiss_index_name(args, offset=None, endpos=None):
    partitions_info = '' if args.partitions is None else f'.{args.partitions}'
    range_info = '' if offset is None else f'.{offset}-{endpos}'

    return f'ivfpq{partitions_info}{range_info}.faiss'

def load_index_part(filename, verbose=True):
    # part: (N, dim)
    part = torch.load(filename)

    if type(part) == list:  # for backward compatibility
        part = torch.cat(part)

    return part

def load_sample(samples_paths, sample_fraction=None):
    sample = []

    for filename in samples_paths:
        print_message(f"#> Loading {filename} ...")
        part = load_index_part(filename)
        if sample_fraction:
            # Randomly sample a fraction of the part.
            part = part[torch.randint(0, high=part.size(0), size=(int(part.size(0) * sample_fraction),))]
        sample.append(part)

    sample = torch.cat(sample).float().numpy()

    print("#> Sample has shape", sample.shape)

    return sample


def prepare_faiss_index(slice_samples_paths, partitions, sample_fraction=None):
    training_sample = load_sample(slice_samples_paths, sample_fraction=sample_fraction)

    dim = training_sample.shape[-1]
    # Create a FAISS index with the specified dimensions and partitions.
    index = FaissIndex(dim, partitions)

    print_message("#> Training with the vectors...")

    # Train the index with the provided training sample.
    index.train(training_sample)

    print_message("Done training!\n")

    return index


SPAN = 3

def get_parts(directory):
    """ Get all parts in the directory, sorted by their integer prefix.
        Returns:
            parts: List of integers representing the part indices.
            parts_paths: List of paths to the part files.
            samples_paths: List of paths to the sample files."""
    
    extension = '.pt'

    parts = sorted([int(filename[: -1 * len(extension)]) for filename in os.listdir(directory)
                    if filename.endswith(extension)])

    assert list(range(len(parts))) == parts, parts

    # Integer-sortedness matters.
    parts_paths = [os.path.join(directory, '{}{}'.format(filename, extension)) for filename in parts]
    samples_paths = [os.path.join(directory, '{}.sample'.format(filename)) for filename in parts]

    return parts, parts_paths, samples_paths

def index_faiss(args):
    print_message("#> Starting..")

    parts, parts_paths, samples_paths = get_parts(args.index_path)

    if args.sample is not None:
        assert args.sample, args.sample
        print_message(f"#> Training with {round(args.sample * 100.0, 1)}% of *all* embeddings (provided --sample).")
        samples_paths = parts_paths

    num_parts_per_slice = math.ceil(len(parts) / args.slices)

    for slice_idx, part_offset in enumerate(range(0, len(parts), num_parts_per_slice)):
        part_endpos = min(part_offset + num_parts_per_slice, len(parts))

        slice_parts_paths = parts_paths[part_offset:part_endpos]
        slice_samples_paths = samples_paths[part_offset:part_endpos]

        if args.slices == 1:
            faiss_index_name = get_faiss_index_name(args)
        else:
            faiss_index_name = get_faiss_index_name(args, offset=part_offset, endpos=part_endpos)

        output_path = os.path.join(args.index_path, faiss_index_name)
        print_message(f"#> Processing slice #{slice_idx+1} of {args.slices} (range {part_offset}..{part_endpos}).")
        print_message(f"#> Will write to {output_path}.")

        assert not os.path.exists(output_path), output_path

        index = prepare_faiss_index(slice_samples_paths, args.partitions, args.sample)

        loaded_parts = queue.Queue(maxsize=1)

        def _loader_thread(thread_parts_paths):
            for filenames in grouper(thread_parts_paths, SPAN, fillvalue=None):
                sub_collection = [load_index_part(filename) for filename in filenames if filename is not None]
                sub_collection = torch.cat(sub_collection)
                sub_collection = sub_collection.float().numpy()
                loaded_parts.put(sub_collection)

        thread = threading.Thread(target=_loader_thread, args=(slice_parts_paths,))
        thread.start()

        print_message("#> Indexing the vectors...")

        for filenames in grouper(slice_parts_paths, SPAN, fillvalue=None):
            print_message("#> Loading", filenames, "(from queue)...")
            sub_collection = loaded_parts.get()

            print_message("#> Processing a sub_collection with shape", sub_collection.shape)
            index.add(sub_collection)

        print_message("Done indexing!")

        index.save(output_path)

        print_message(f"\n\nDone! All complete (for slice #{slice_idx+1} of {args.slices})!")

        thread.join()
