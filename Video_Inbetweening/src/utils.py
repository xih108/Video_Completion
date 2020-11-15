"""
Some codes from https://github.com/Newmu/dcgan_code
"""
import random

import cv2
import imageio
import numpy as np
import scipy.misc


def transform(image):
    return image / 127.5 - 1.


def inverse_transform(images):
    return (images + 1.) / 2.


def save_images(images, size, image_path):
    return imsave(inverse_transform(images) * 255., size, image_path)


def merge(images, size):
    h, w = images.shape[1], images.shape[2]
    img = np.zeros((h * size[0], w * size[1], 3))

    for idx, image in enumerate(images):
        i = idx % size[1]
        j = idx / size[1]
        img[j * h:j * h + h, i * w:i * w + w, :] = image

    return img


def imsave(images, size, path):
    return scipy.misc.imsave(path, merge(images, size))


def get_minibatches_idx(n, minibatch_size, shuffle=False):
    """
    Used to shuffle the dataset at each iteration.
    """

    idx_list = np.arange(n, dtype="int32")

    if shuffle:
        random.shuffle(idx_list)

    minibatches = []
    minibatch_start = 0
    for i in range(n // minibatch_size):
        minibatches.append(idx_list[minibatch_start:
                                    minibatch_start + minibatch_size])
        minibatch_start += minibatch_size

    if (minibatch_start != n):
        # Make a minibatch out of what is left
        minibatches.append(idx_list[minibatch_start:])

    return zip(range(len(minibatches)), minibatches)


def draw_frame(img, is_input):
    if img.shape[2] == 1:
        img = np.repeat(img, [3], axis=2)

    if is_input:
        img[:2, :, 0] = img[:2, :, 2] = 0
        img[:, :2, 0] = img[:, :2, 2] = 0
        img[-2:, :, 0] = img[-2:, :, 2] = 0
        img[:, -2:, 0] = img[:, -2:, 2] = 0
        img[:2, :, 1] = 255
        img[:, :2, 1] = 255
        img[-2:, :, 1] = 255
        img[:, -2:, 1] = 255
    else:
        img[:2, :, 0] = img[:2, :, 1] = 0
        img[:, :2, 0] = img[:, :2, 2] = 0
        img[-2:, :, 0] = img[-2:, :, 1] = 0
        img[:, -2:, 0] = img[:, -2:, 1] = 0
        img[:2, :, 2] = 255
        img[:, :2, 2] = 255
        img[-2:, :, 2] = 255
        img[:, -2:, 2] = 255

    return img


def load_data(f_name, data_path, image_size, K, T, data_name='KTH'):
    flip = np.random.binomial(1, .5, 1)[0]
    if data_name == 'KTH':
        tokens = f_name.split()
        vid_path = data_path + tokens[0] + "_uncomp.avi"
    elif data_name == 'MITD':
        vid_path = (data_path + f_name).replace('\n', '')
    print(">>>vid_path", vid_path)

    # raise ValueError()
    while True:
        try:
            vid = imageio.get_reader(vid_path, "ffmpeg")
            break
        except Exception as e:
            print(e)
            print(vid_path)
            print("imageio failed loading frames, retrying")

    if data_name == 'KTH':
        low = int(tokens[1])
        high = int(np.min([int(tokens[2]), vid.count_frames()]) - K - T + 1)
    elif data_name == 'MITD':
        low = 1
        high = 2
        for i in range(5): # try 5 times
            try:
                high = vid.count_frames() - K - T + 1
                break
            except:
                print(f"Could not get number of frames: {i}")


    # vid = imageio.get_reader(vid_path, "ffmpeg")
    if low == high:
        stidx = 0
    else:
        if low >= high: print(vid_path)
        stidx = np.random.randint(low=low, high=high)
    seq = np.zeros((image_size, image_size, K + T, 1), dtype="float32")

    
    print("-------print vid_path------") ##adl
    import alog ##adl
    from pprint import pprint ##adl
    alog.info("vid_path") ##adl
    print(">>> type(vid_path) = ", type(vid_path)) ##adl
    if hasattr(vid_path, "shape"): ##adl
        print(">>> vid_path.shape", vid_path.shape) ##adl
    if type(vid_path) is list: ##adl
        print(">>> len(vid_path) = ", len(vid_path)) ##adl
        pprint(vid_path) ##adl
    else: ##adl
        pprint(vid_path) ##adl
    print("------------------------\n") ##adl
    
    
    print("-------print vid.count_frames()------") ##adl
    import alog ##adl
    from pprint import pprint ##adl
    alog.info("vid.count_frames()") ##adl
    print(">>> type(vid.count_frames()) = ", type(vid.count_frames())) ##adl
    if hasattr(vid.count_frames(), "shape"): ##adl
        print(">>> vid.count_frames().shape", vid.count_frames().shape) ##adl
    if type(vid.count_frames()) is list: ##adl
        print(">>> len(vid.count_frames()) = ", len(vid.count_frames())) ##adl
        pprint(vid.count_frames()) ##adl
    else: ##adl
        pprint(vid.count_frames()) ##adl
    print("------------------------\n") ##adl
    
    """
    var_list = [low, high, stidx] ##adl
    for idx,x in enumerate(var_list): ##adl
        var_names = "low, high, stidx".split(",") ##adl
        cur_name = var_names[idx] ##adl
        print("-------print "+ cur_name + "------") ##adl
        import alog ##adl
        from pprint import pprint ##adl
        alog.info(cur_name) ##adl
        print(">>> type(x) = ", type(x)) ##adl
        if hasattr(x, "shape"): ##adl
            print(">>> " + cur_name + ".shape", x.shape) ##adl
        if type(x) is list: ##adl
            print(">>> len(" + cur_name + ") = ", len(x)) ##adl
            pprint(x) ##adl
        else: ##adl
            pprint(x) ##adl
            pass ##adl
        print("------------------------\n") ##adl
    """
    
    
    for t in range(K + T):
        img = cv2.cvtColor(cv2.resize(vid.get_data(stidx + t),
                                      (image_size, image_size)),
                           cv2.COLOR_RGB2GRAY)
        seq[:, :, t] = transform(img[:, :, None])

    if flip == 1:
        seq = seq[:, ::-1]

    diff = np.zeros((image_size, image_size, K - 1, 1), dtype="float32")
    for t in range(1, K):
        prev = inverse_transform(seq[:, :, t - 1])
        next = inverse_transform(seq[:, :, t])
        diff[:, :, t - 1] = next.astype("float32") - prev.astype("float32")
    return seq, diff


def load_s1m_data(f_name, data_path, trainlist, K, T):
    flip = np.random.binomial(1, .5, 1)[0]
    vid_path = data_path + f_name
    img_size = [240, 320]

    while True:
        try:
            vid = imageio.get_reader(vid_path, "ffmpeg")
            low = 1
            high = vid.count_frames() - K - T + 1
            if low == high:
                stidx = 0
            else:
                stidx = np.random.randint(low=low, high=high)
            seq = np.zeros((img_size[0], img_size[1], K + T, 3),
                           dtype="float32")
            for t in range(K + T):
                img = cv2.resize(vid.get_data(stidx + t),
                                 (img_size[1], img_size[0]))[:, :, ::-1]
                seq[:, :, t] = transform(img)

            if flip == 1:
                seq = seq[:, ::-1]

            diff = np.zeros((img_size[0], img_size[1], K - 1, 1),
                            dtype="float32")
            for t in range(1, K):
                prev = inverse_transform(seq[:, :, t - 1]) * 255
                prev = cv2.cvtColor(prev.astype("uint8"), cv2.COLOR_BGR2GRAY)
                next = inverse_transform(seq[:, :, t]) * 255
                next = cv2.cvtColor(next.astype("uint8"), cv2.COLOR_BGR2GRAY)
                diff[:, :, t - 1, 0] = (next.astype("float32") - prev.astype(
                    "float32")) / 255.
            break
        except Exception:
            # In case the current video is bad load a random one
            rep_idx = np.random.randint(low=0, high=len(trainlist))
            f_name = trainlist[rep_idx]
            vid_path = data_path + f_name

    return seq, diff
