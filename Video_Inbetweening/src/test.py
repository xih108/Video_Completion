import alog
import os
from argparse import ArgumentParser
from os import makedirs, system
from os.path import exists

import skimage.measure as measure
import ssim
import tensorflow as tf
from PIL import Image
from joblib import Parallel, delayed

from midnet import MCNET
from utils import *
from pprint import pprint


def main(prefix, image_size, K, T, gpu, p):
    data_path = "../data/MITD/"
    f = open(data_path + "skiing_val.txt", "r")
    testfiles = f.readlines()

    batch_size = 1
    num_gpus = 1
    checkpoint_dir = "../models/" + prefix + "/"
    test_model = "MCNET.model-" + str(p)

    with tf.device("/gpu:%d" % gpu[0]):
        model = MCNET(image_size=[image_size, image_size], c_dim=1, K=K,
                      batch_size=batch_size, T=T, checkpoint_dir=checkpoint_dir)
    
    gpu_options = tf.GPUOptions(per_process_gpu_memory_fraction=1.0)
    config = tf.ConfigProto(allow_soft_placement=True,
                            log_device_placement=False,
                            gpu_options=gpu_options)
    config.gpu_options.allow_growth = True  
    
    ### Added for initializing the saver
    _diff_in = tf.placeholder(tf.float32,
                              [batch_size * num_gpus, image_size,
                               image_size, K - 1, 1])
    _xt = tf.placeholder(tf.float32,
                         [batch_size * num_gpus, image_size, image_size,
                          model.c_dim])
    _target = tf.placeholder(tf.float32,
                             [batch_size * num_gpus, image_size, image_size,
                              K + T, model.c_dim])
    ### Added for initializing the saver

    ### Load Model
    with tf.Session(config=config) as sess:
        
        ### Added for initializing the saver
        model.build_model(
            _diff_in[:batch_size, :, :,
            :, :],
            _xt[:batch_size, :, :, :],
            _target[:batch_size, :, :,
            :, :])
        ### Added for initializing the saver
            
        tf.global_variables_initializer().run()
        loaded, model_name = model.load(sess, checkpoint_dir, test_model)
        if loaded:
            print(" [*] Load SUCCESS")
        else:
            print(" [!] Load failed... exitting")
            return
            
        
        gts = np.zeros((len(testfiles), 16, image_size, image_size))
        preds = np.zeros((len(testfiles), 16, image_size, image_size))

        vid_names = []
        psnr_err = np.zeros((0, T))
        ssim_err = np.zeros((0, T))
        for i in range(len(testfiles)):
            with Parallel(n_jobs=batch_size * num_gpus) as parallel:
                batchidx = np.array([i,])
                if len(batchidx) == batch_size * num_gpus:
                    seq_batch = np.zeros(
                        (batch_size * num_gpus, image_size, image_size,
                            K + T, 1), dtype="float32")
                    diff_batch = np.zeros(
                        (batch_size * num_gpus, image_size, image_size,
                            K - 1, 1), dtype="float32")
                    Ts = np.repeat(np.array([T]), batch_size * num_gpus,
                                    axis=0)
                    Ks = np.repeat(np.array([K]), batch_size * num_gpus,
                                    axis=0)
                    paths = np.repeat(data_path, batch_size * num_gpus,
                                        axis=0)
                    tfiles = np.array(testfiles)[batchidx]
                    shapes = np.repeat(np.array([image_size]),
                                        batch_size * num_gpus, axis=0)

                    try:
                        output = parallel(
                            delayed(load_data)(f, p, img_sze, k, t, 'MITD')
                            for f, p, img_sze, k, t in
                            zip(tfiles, paths, shapes, Ks, Ts))
                    except:
                        continue
                        print("-------------- Skip file --------------")
                        print(sys.exc_info()[0])
                        print(tfiles)
                        print("---------------------------------------")

                    seq_batch[0] = output[0][0]
                    diff_batch[0] = output[0][1]

                    out = sess.run(model.E_pred,
                                    feed_dict={
                                        _diff_in: diff_batch,
                                        _xt: seq_batch[:, :,
                                                :, K - 1],
                                        _target: seq_batch
                                    }) # (1, 16, 64, 64, 1)

                    print("-------------- Saving --------------")
                    savedir = "../tests/images/skiing/" + prefix + "/"
                    if not os.path.exists(savedir):
                        os.makedirs(savedir)

                    # print(out[0, 0, :, :, 0])
                    for idx in range(0, 16):
                        show_pred = out[0, idx, :, :, 0]
                        show_target = seq_batch[0, :, :, idx, 0]
                        show_pred = (inverse_transform(
                            show_pred) * 255).astype(
                            "uint8")
                        show_target = (inverse_transform(
                            show_target) * 255).astype(
                            "uint8")

                        gts[i, idx] = show_target
                        preds[i, idx] = show_pred
                        
                        cv2.imwrite(
                            savedir + "pred_" + str(i) + "_" + "{0:04d}".format(
                                idx) + ".png", show_pred)
                        cv2.imwrite(
                            savedir + "gt_" + str(i) + "_" + "{0:04d}".format(idx) + ".png",
                            show_target)

        import pickle
        pickle.dump( gts, open( savedir + "gts.p", "wb" ) )
        pickle.dump( preds, open( savedir + "preds.p", "wb" ) )
                            
    """

            action = vid_path.split("_")[1]
            if action in ["running", "jogging"]:
                n_skip = 3
            else:
                n_skip = T

            for j in range(int(tokens[1]), int(tokens[2]) - K - T - 1, n_skip):
                print("Video " + str(i) + "/" + str(
                    len(testfiles)) + ". Index " + str(j) +
                      "/" + str(vid.get_length() - T - 1))

                folder_pref = vid_path.split("/")[-1].split(".")[0]
                folder_name = folder_pref + "." + str(j) + "-" + str(j + T)
                vid_names.append(folder_name)
                savedir = "../test/images/" + prefix + "/" + folder_name

                seq_batch = np.zeros((1, image_size, image_size, K + T, c_dim),
                                     dtype="float32")
                diff_batch = np.zeros((1, image_size, image_size, K - 1, 1),
                                      dtype="float32")
                for t in range(K + T):
                    # imageio fails randomly sometimes
                    while True:
                        try:
                            img = cv2.resize(vid.get_data(j + t),
                                             (image_size, image_size))
                            break
                        except Exception:
                            print("imageio failed loading frames, retrying")

                    img = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
                    seq_batch[0, :, :, t] = transform(img[:, :, None])

                for t in range(1, K):
                    prev = inverse_transform(seq_batch[0, :, :, t - 1])
                    next = inverse_transform(seq_batch[0, :, :, t])
                    diff = next.astype("float32") - prev.astype("float32")
                    diff_batch[0, :, :, t - 1] = diff

                true_data = seq_batch[:, :, :, K:, :].copy()
                pred_data = np.zeros(true_data.shape, dtype="float32")
                xt = seq_batch[:, :, :, K - 1]
                pred_data[0] = sess.run(model.G,
                                        feed_dict={model.diff_in: diff_batch,
                                                   model.xt: xt})

                if not os.path.exists(savedir):
                    os.makedirs(savedir)

                cpsnr = np.zeros((K + T,))
                cssim = np.zeros((K + T,))
                pred_data = np.concatenate((seq_batch[:, :, :, :K], pred_data),
                                           axis=3)
                true_data = np.concatenate((seq_batch[:, :, :, :K], true_data),
                                           axis=3)
                for t in range(K + T):
                    pred = (inverse_transform(
                        pred_data[0, :, :, t]) * 255).astype("uint8")
                    target = (inverse_transform(
                        true_data[0, :, :, t]) * 255).astype("uint8")

                    cpsnr[t] = measure.compare_psnr(pred, target)
                    cssim[t] = ssim.compute_ssim(Image.fromarray(
                        cv2.cvtColor(target, cv2.COLOR_GRAY2BGR)),
                                                 Image.fromarray(
                                                     cv2.cvtColor(pred,
                                                                  cv2.COLOR_GRAY2BGR)))

                    # ================================== Produce Samples ======================================
                    pred = draw_frame(pred, t < K)
                    target = draw_frame(target, t < K)

                    cv2.imwrite(
                        savedir + "/pred_" + "{0:04d}".format(t) + ".png", pred)
                    cv2.imwrite(savedir + "/gt_" + "{0:04d}".format(t) + ".png",
                                target)

                cmd1 = "rm " + savedir + "/pred.gif"
                cmd2 = ("ffmpeg -f image2 -framerate 7 -i " + savedir +
                        "/pred_%04d.png " + savedir + "/pred.gif")
                cmd3 = "rm " + savedir + "/pred*.png"

                system(cmd1)
                system(cmd2)
                system(cmd3)

                cmd1 = "rm " + savedir + "/gt.gif"
                cmd2 = ("ffmpeg -f image2 -framerate 7 -i " + savedir +
                        "/gt_%04d.png " + savedir + "/gt.gif")
                cmd3 = "rm " + savedir + "/gt*.png"

                system(cmd1)
                system(cmd2)
                system(cmd3)

                psnr_err = np.concatenate((psnr_err, cpsnr[None, K:]), axis=0)
                ssim_err = np.concatenate((ssim_err, cssim[None, K:]), axis=0)

        np.savez(save_path, psnr=psnr_err, ssim=ssim_err)
        print("Results saved to " + save_path)
    print("Done.")
    """


if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_argument("--prefix", type=str, dest="prefix",
                        help="Prefix for log/snapshot")
    parser.add_argument("--image_size", type=int, dest="image_size",
                        default=64, help="Mini-batch size")
    parser.add_argument("--K", type=int, dest="K",
                        default=2,
                        help="Number of steps to observe from the past")
    parser.add_argument("--T", type=int, dest="T",
                        default=14, help="Number of steps into the future")
    parser.add_argument("--gpu", type=int, nargs="+", dest="gpu", required=True,
                        help="GPU device id")
    parser.add_argument("--p", type=int, dest="p", help="checkpoint index")

    args = parser.parse_args()
    main(**vars(args))
