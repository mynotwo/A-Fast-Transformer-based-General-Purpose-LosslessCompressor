import torch
import os

from collections.abc import Iterable
from collections import Counter

def get_static_features(data):
    print("Seq Length {}".format(len(data)))
    #vals = list(set(data))
    #vals.sort()

    counts = Counter(data)
    counts = dict(counts.most_common(len(data)))
    v = []
    for k in counts:
      counts[k] = counts[k]/len(data)
      v.append(counts[k])
    for i in range(len(v), 256):
      v.append(0)
    print(len(v))

    char2id_dict = {c: i for (i,c) in enumerate(counts)}
    id2char_dict = {i: c for (i,c) in enumerate(counts)}
    print(counts)
    
    return v, char2id_dict, id2char_dict

def decode_token(token):
    return str(chr(max(32, token)))

def decode_tokens(tokens):
    return ''.join(list(map(decode_token, tokens)))
 
def set_freeze_by_names(model, layer_names=None, freeze=True):
    if not isinstance(layer_names, Iterable):
        layer_names = model.state_dict().keys()
    for name, param in model.named_parameters():
        if name not in layer_names:
            continue
        param.requires_grad = not freeze

def check_model_grad(model, layer_names=None):
    # if not given, check all layer
    if not isinstance(layer_names, Iterable):
        layer_names = model.state_dict().keys()
    
    print('==============================================')
    for name, param in model.named_parameters():
        if name not in layer_names:
            continue
        print(name, 'require_grad:', param.requires_grad)
    print('==============================================')
 
def freeze_by_names(model, layer_names=None):
    set_freeze_by_names(model, layer_names, True)
 
 
def unfreeze_by_names(model, layer_names):
    set_freeze_by_names(model, layer_names, False)
 
 
def set_freeze_by_idxs(model, idxs, freeze=True):
    if not isinstance(idxs, Iterable):
        idxs = [idxs]
    num_child = len(list(model.children()))
    idxs = tuple(map(lambda idx: num_child + idx if idx < 0 else idx, idxs))
    for idx, child in enumerate(model.children()):
        if idx not in idxs:
            continue
        for param in child.parameters():
            param.requires_grad = not freeze
 
 
def freeze_by_idxs(model, idxs):
    set_freeze_by_idxs(model, idxs, True)
 
 
def unfreeze_by_idxs(model, idxs):
    set_freeze_by_idxs(model, idxs, False)

def average_model(model_list):
    #model_list = get_model_list(model_path, iter_list)
    print(model_list)
    model = torch.load(model_list[0])
    new_weights = dict(model.state_dict())
    for name in model_list[1:]:
        next_model = torch.load(name)
        next_weights = next_model.state_dict()
        for key in next_weights:
            new_weights[key] = (new_weights[key] + next_weights[key])/2

        del next_model, next_weights

    model.load_state_dict(new_weights)
    del new_weights

    return model

def get_model_list(model_path, iter_list):
    return [os.path.join(model_path, f) for f in os.listdir(model_path) if int(f.split("_")[-1].split(".")[0]) in iter_list]

def average_output(model_path, iter_list, inp):
    model_list = get_model_list(model_path, iter_list)
    out = []
    for name in model_list:
        model = torch.load(name).cuda().eval()
        out.append(model.full_forward(inp))

    out_tensor = torch.Tensor(len(out), len(out[0]))
    torch.cat(out, out_tensor)
    print(out.shape)
    return torch.mean(out, dim=0)

def loading_time(dataset, num_workers, pin_memory):
    kwargs = {'num_workers': num_workers, 'pin_memory': pin_memory} if use_cuda else {}
    #train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=args.batch_size, shuffle=(train_sampler is None),sampler=train_sampler, **kwargs)
    train_loader = torch.utils.data.DataLoader(dataset, batch_size=batch_size, shuffle=False, num_workers=4, pin_memory=True)
    start = time.time()
    for epoch in range(4):
        for batch_idx, (data, target) in enumerate(train_loader):
            if batch_idx == 15:
                break
                pass
    end = time.time()
    print(" Used {} second with num_workers = {}".format(end-start,num_workers))
    return end-start

def get_best_loader(dataset):
    for pin_memory in [False, True]:
        print("While pin_memory =",pin_memory)
        for num_workers in range(0, core_number*2+1, 4): 
            current_time = loading_time(dataset, num_workers, pin_memory)
            if current_time < best_time[pin_memory]:
                best_time[pin_memory] = current_time
                best_num_worker[pin_memory] = num_workers
            else: # assuming its a convex function 
                if best_num_worker[pin_memory] == 0:
                    the_range = []
                else:
                    the_range = list(range(best_num_worker[pin_memory]-3, best_num_worker[pin_memory]))
                    for num_workers in (the_range + list(range(best_num_worker[pin_memory] + 1,best_num_worker[pin_memory] + 4))): 
                        current_time = loading_time(dataset, num_workers, pin_memory)
                        if current_time < best_time[pin_memory]:
                            best_time[pin_memory] = current_time
                            best_num_worker[pin_memory] = num_workers
                            break

                    if best_time[0] < best_time[1]:
                        print("Best num_workers =", best_num_worker[0], "with pin_memory = False")
                    else:
                        print("Best num_workers =", best_num_worker[1], "with pin_memory = True")


