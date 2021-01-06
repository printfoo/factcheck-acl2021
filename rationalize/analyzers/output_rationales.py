# coding: utf-8


import torch
from torch.autograd import Variable

import os
import numpy as np


def output(ckpt_path, out_path, data, args):

    model = torch.load(ckpt_path)  # Load model from checkpoint.    

    if not os.path.exists(out_path):
        os.mkdir(out_path)
    
    model.eval()  # Set model to eval mode.
    
    set_names = ["train", "dev", "test"]  # Analyze all.
    for set_name in set_names:
        f = open(os.path.join(out_path, set_name + ".tsv"), "w")
        f.write("index\trationale_true\trationale_pred\tmask\n")
        
        instance_count = data.data_sets[set_name].size()
        for start in range(instance_count // args.batch_size + 1):

            # Get a batch.
            batch_idx = range(start * args.batch_size,
                              min((start + 1) * args.batch_size, instance_count))
            samples = data.get_batch(set_name, batch_idx=batch_idx, sort=True, return_id=True)
            x, y, m, r, s, d, ids = samples

            # Save values to torch tensors.
            x = Variable(torch.from_numpy(x))
            y = Variable(torch.from_numpy(y))
            m = Variable(torch.from_numpy(m)).float()
            r = Variable(torch.from_numpy(r)).float()
            s = Variable(torch.from_numpy(s)).float()
            d = Variable(torch.from_numpy(d)).float()
            if args.cuda:
                x = x.cuda()
                y = y.cuda()
                m = m.cuda()
                r = r.cuda()
                s = s.cuda()
                d = d.cuda()

            # Get soft or hard rationales, (batch_size, seq_len).
            _, _, z, _, _ = model(x, m)

            # Get writable text from list.
            z = [" ".join([str(z__) for z__ in z_]) for z_ in z.tolist()]
            r = [" ".join([str(r__) for r__ in r_]) for r_ in r.tolist()]
            m = [" ".join([str(m__) for m__ in m_]) for m_ in m.tolist()]
            for id_, z_, r_, m_ in zip(ids, z, r, m):
                line = str(id_) + "\t" + r_ + "\t" + z_ + "\t" + m_ + "\n"
                f.write(line)
        
        f.close()
