# Copyright (c) 2022, salesforce.com, inc.
# All rights reserved.
# SPDX-License-Identifier: BSD-3-Clause
# For full license text, see the LICENSE file in the repo root or https://opensource.org/licenses/BSD-3-Clause

import os
import time
import random
import argparse

import torch

from tokenizers import Tokenizer
from models.progen.modeling_progen import ProGenForCausalLM

# Global variables for model and tokenizer
model = None
tokenizer = None
device = None
fp16_global = True  # Default, will be set during loading

########################################################################
# util


class print_time:
    def __init__(self, desc):
        self.desc = desc

    def __enter__(self):
        print(self.desc)
        self.t = time.time()

    def __exit__(self, type, value, traceback):
        print(f'{self.desc} took {time.time()-self.t:.02f}s')


def set_env():
    os.environ['TOKENIZERS_PARALLELISM'] = 'false'


def set_seed(seed, deterministic=True):
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.backends.cudnn.deterministic = deterministic
        torch.backends.cudnn.benchmark = not deterministic


########################################################################
# model loading and setup


def create_model(ckpt, fp16=True):
    if fp16:
        return ProGenForCausalLM.from_pretrained(ckpt, revision='float16', torch_dtype=torch.float16, low_cpu_mem_usage=True)
    else:
        return ProGenForCausalLM.from_pretrained(ckpt)


def create_tokenizer_custom(file):
    with open(file, 'r') as f:
        return Tokenizer.from_str(f.read())


def load_model_and_tokenizer(model_name='progen2-large', use_fp16=True, device_name='cuda:0'):
    """Loads the ProGen model and tokenizer."""
    global model, tokenizer, device, fp16_global

    if model is not None and tokenizer is not None:
        print("Model and tokenizer already loaded.")
        return model, tokenizer, device

    set_env()

    if not torch.cuda.is_available() and device_name != 'cpu':
        print('CUDA not available, falling back to cpu')
        device_name = 'cpu'

    current_device = torch.device(device_name)
    fp16_global = use_fp16 if current_device.type != 'cpu' else False

    if current_device.type == 'cpu' and use_fp16:
        print('CPU selected, falling back to fp32')
        fp16_global = False

    ckpt = f'./checkpoints/{model_name}'

    with print_time(f'loading parameters for {model_name}'):
        model = create_model(ckpt=ckpt, fp16=fp16_global).to(current_device)

    with print_time('loading tokenizer'):
        tokenizer = create_tokenizer_custom(file='tokenizer.json')

    device = current_device  # Store the device globally

    return model, tokenizer, device


########################################################################
# sample


def truncate(sample, terminals):
    pos = []
    for terminal in terminals:
        find_pos = sample.find(terminal, 1)
        if find_pos != -1:
            pos.append(find_pos)
    if len(pos) > 0:
        return sample[:(min(pos)+1)]
    else:
        return sample


def cross_entropy(logits, target, reduction='mean'):
    return torch.nn.functional.cross_entropy(input=logits, target=target, weight=None, size_average=None, reduce=None, reduction=reduction)


def generate_sequences(context, max_length=256, num_return_sequences=1, top_p=0.95, temp=0.2):
    """Generates sequences using the loaded model."""
    global model, tokenizer, device  # Use globally loaded components

    if model is None or tokenizer is None or device is None:
        raise RuntimeError("Model and tokenizer not loaded. Call load_model_and_tokenizer first.")

    pad_token_id = tokenizer.encode('<|pad|>').ids[0]

    with print_time('sampling'):
        with torch.no_grad():
            input_ids = torch.tensor(tokenizer.encode(context).ids).view([1, -1]).to(device)
            with torch.cuda.amp.autocast(enabled=(device.type == 'cuda' and fp16_global)):
                tokens_batch = model.generate(input_ids, do_sample=True, temperature=temp, max_length=max_length, top_p=top_p, num_return_sequences=num_return_sequences, pad_token_id=pad_token_id)

            as_lists = lambda batch: [batch[i, ...].detach().cpu().numpy().tolist() for i in range(batch.shape[0])]
            completions = tokenizer.decode_batch(as_lists(tokens_batch))
            truncations = [truncate(completion, terminals=['1', '2']) for completion in completions]
            return truncations


########################################################################
# sanity check


def run_sanity_check(model_to_check, tokenizer_to_check, device_to_check, model_name, use_fp16):
    """Performs the sanity cross-entropy check."""
    with print_time('sanity cross-entropy'):
        def ce(tokens):
            with torch.no_grad():
                with torch.cuda.amp.autocast(enabled=use_fp16):
                    target = torch.tensor(tokenizer_to_check.encode(tokens).ids).to(device_to_check)
                    logits = model_to_check(target, labels=target).logits

                    # shift
                    logits = logits[:-1, ...]
                    target = target[1:]

                    return cross_entropy(logits=logits, target=target).item()

        x_uniref90bfd30 = '2GFLPFRGADEGLAAREAATLAARGTAARAYREDSWAVPVPRGLLGDLTARVAALGAASPPPADPLAVTLDLHHVTAEVALTTVLDAATLVHGQTRVLSAEDAAEAATAAAAATEAYLERLQDFVLFMSASVRVWRRGNAAGATGPEWDQWYTVADRDALGSAPTHLAVLGRQADALCHFVLDRVAWGTCGTPLWSGDEDLGNVVATFAGYADRLATAPRDLIM1'
        x_oas = '1EVQLVESGGGLVQPGGSLRLSCAASGFTFSSYAMHWVRQAPWKGLEYVSAISSNGGSTYYANSVKGRFTISRDNSKNTLYLQMGSLRAEDMAVYYCARDESGYSYGWGYYFDYWGQGTLVTVSS2'
        x_bfd90 = '1TAPRSTRASGSEGSRPPGIPAKGRRCLPSRAGSVTPRFRHARQGTATVAKEQGRKLIASNRKARHDYHIEDTFEAGLVLTGTEVKSLRMGRASLIDGYAVFYGEELWLEGVHIPEYLNGNWTNHTPRRRRKLLLNRSELTKLAHKTSESGHTIVPLALYFKDGRAKVEIAVAKGKKAYDKRHALRERQDQREV2'

        checkpoint_x_ce = {
            'progen2-small': (x_uniref90bfd30, 2.4),
            'progen2-medium': (x_uniref90bfd30, 1.9),
            'progen2-base': (x_uniref90bfd30, 1.9),
            'progen2-large': (x_uniref90bfd30, 1.8),
            'progen2-xlarge': (x_uniref90bfd30, 1.0),
            'progen2-oas': (x_oas, 0.3),
            'progen2-BFD90': (x_bfd90, 1.3),
        }

        if model_name not in checkpoint_x_ce:
            print(f"Warning: No sanity check data for model {model_name}")
            return

        ce_eval = ce(checkpoint_x_ce[model_name][0])
        ce_target = checkpoint_x_ce[model_name][1]

        print(f"Sanity Check ({model_name}): Target CE={ce_target:.4f}, Eval CE={ce_eval:.4f}, Diff={abs(ce_eval - ce_target):.4f}")

        if abs(ce_eval - ce_target) >= 0.15:
            print(f"Warning: Sanity check failed for {model_name}. Difference {abs(ce_eval - ce_target):.4f} >= 0.15")


########################################################################

# main (for command-line usage)


def main():

    # (0) constants

    models_151M = ['progen2-small']
    models_754M = ['progen2-medium', 'progen2-oas', 'progen2-base']
    models_2B = ['progen2-large', 'progen2-BFD90']
    models_6B = ['progen2-xlarge']
    models = models_151M + models_754M + models_2B + models_6B

    # (1) params

    parser = argparse.ArgumentParser()
    parser.add_argument('--model', type=str, choices=models, default='progen2-small')
    parser.add_argument('--device', type=str, default='cuda:0')
    parser.add_argument('--rng-seed', type=int, default=42)
    parser.add_argument('--rng-deterministic', default=True, type=lambda x: (str(x).lower() == 'true'))
    parser.add_argument('--p', type=float, default=0.95)
    parser.add_argument('--t', type=float, default=0.2)
    parser.add_argument('--max-length', type=int, default=256)
    parser.add_argument('--num-samples', type=int, default=1)
    parser.add_argument('--fp16', default=True, type=lambda x: (str(x).lower() == 'true'))
    parser.add_argument('--context', type=str, default='1')
    parser.add_argument('--sanity', default=True, type=lambda x: (str(x).lower() == 'true'))
    args = parser.parse_args()

    # (2) preamble & load model/tokenizer
    set_seed(args.rng_seed, deterministic=args.rng_deterministic)
    loaded_model, loaded_tokenizer, loaded_device = load_model_and_tokenizer(
        model_name=args.model,
        use_fp16=args.fp16,
        device_name=args.device
    )

    # (3) Sanity Check (if requested via CLI)
    if args.sanity:
        run_sanity_check(loaded_model, loaded_tokenizer, loaded_device, args.model, fp16_global)

    # (4) Generate sequences using the new function
    truncations = generate_sequences(
        context=args.context,
        max_length=args.max_length,
        num_return_sequences=args.num_samples,
        temp=args.t,
        top_p=args.p
    )

    # (5) Print results
    print("\n--- Generated Sequences ---")
    print(f"Context: {args.context}")
    for i, truncation in enumerate(truncations):
        print(f"\nSample {i+1}:")
        print(truncation)


if __name__ == '__main__':
    main()
    print('done.')
