from tensorpack import *

from RBMCell import RBM
from ops import *
from utils import *
import  os

# https://arxiv.org/pdf/1905.10240.pdf

class MCNET(object):
    def __init__(self, image_size, batch_size=32, c_dim=3,
                 K=10, T=10, checkpoint_dir=None, is_train=True):

        self.batch_size = batch_size
        self.image_size = image_size
        self.is_train = is_train
        self.gf_dim = 64
        self.df_dim = 64

        self.dim_C = 64
        self.dim_D = 128

        self.c_dim = c_dim
        self.K = K
        self.T = T
        self.diff_shape = [batch_size, self.image_size[0], self.image_size[1],
                           K - 1, 1]
        self.xt_shape = [batch_size, self.image_size[0], self.image_size[1],
                         c_dim]
        self.target_shape = [batch_size, self.image_size[0], self.image_size[1],
                             K + T, c_dim]

        self.num_layers = 15
        self.hidden_dims = [256, 256, 256, 256, 256, 256, 256, 256, 256, 256,
                            256, 256, 256, 256, 256]

    def build_model(self, diff_in, xt, target):
        # xt: [2, 64, 64, 1]
        # target: [2, 64, 64, 16, 1]
        [n_bz, n_h, n_w, n_dep, n_c] = target.get_shape().as_list()
        target = tf.transpose(target,[0, 3, 1, 2, 4])

        E_xs = target[:, 0, :, :, :]
        E_xe = target[:, -1, :, :, :]
        E_xs = self.image_encoder(E_xs, reuse=False)
        E_xe = self.image_encoder(E_xe, reuse=True)
        E_xs = tf.expand_dims(E_xs, 1)
        E_xe = tf.expand_dims(E_xe, 1)

        u = tf.get_variable('u', [self.dim_D, 1],
                            initializer=tf.contrib.layers.xavier_initializer())
        u_l = self.latent_gen(E_xs, E_xe, u, reuse=False) #[3, 16, 8, 8, 64]
        E_pred = self.video_generator(u_l, reuse=False, dim_c=1) # [3, 16, 64, 64, 1]


        self.diff_in = diff_in
        self.xt = xt
        self.target = target

        self.E_pred = E_pred
        # cell = RBM([self.image_size[0] // 8, self.image_size[1] // 8], [3, 3],
        #            15, self.hidden_dims, 5, 0.5)

        # pred = self.forward(self.diff_in, self.xt, cell)

        # self.G = tf.concat(axis=3, values=pred)
        if self.is_train:
            # true_sim = inverse_transform(self.target[:, :, :, self.K:, :])
            # if self.c_dim == 1: true_sim = tf.tile(true_sim, [1, 1, 1, 1, 3])
            # true_sim = tf.reshape(tf.transpose(true_sim, [0, 3, 1, 2, 4]),
            #                       [-1, self.image_size[0],
            #                        self.image_size[1], 3])
            # gen_sim = inverse_transform(self.G)
            # if self.c_dim == 1: gen_sim = tf.tile(gen_sim, [1, 1, 1, 1, 3])
            # gen_sim = tf.reshape(tf.transpose(gen_sim, [0, 3, 1, 2, 4]),
            #                      [-1, self.image_size[0],
            #                       self.image_size[1], 3])

            true_sim = inverse_transform(target)
            if self.c_dim == 1: true_sim = tf.tile(true_sim, [1, 1, 1, 1, 3])
            true_sim = tf.reshape(true_sim, [-1, self.image_size[0], self.image_size[1], 3])
            gen_sim = inverse_transform(E_pred)
            if self.c_dim == 1: gen_sim = tf.tile(gen_sim, [1, 1, 1, 1, 3])
            gen_sim = tf.reshape(gen_sim,  [-1, self.image_size[0], self.image_size[1], 3])

            # binput = tf.reshape(self.target[:, :, :, :self.K, :],
            #                     [self.batch_size, self.image_size[0],
            #                      self.image_size[1], -1])
            # btarget = tf.reshape(self.target[:, :, :, self.K:, :],
            #                      [self.batch_size, self.image_size[0],
            #                       self.image_size[1], -1])
            # bgen = tf.reshape(self.G, [self.batch_size,
            #                            self.image_size[0],
            #                            self.image_size[1], -1])
            #
            # good_data = tf.concat(axis=3, values=[binput, btarget])
            # gen_data = tf.concat(axis=3, values=[binput, bgen])

            self.D2 = []
            self.D2_logits = []
            with tf.variable_scope("DIS", reuse=False):
                # self.D, self.D_logits = self.discriminator(good_data)
                self.D, self.D_logits = self.video_discriminator(target, reuse=False)
                #
                n_bz = target.shape[0]

                # for i in range(1, n_bz-1):
                #     img = target[:, i, :, :, :]
                #     if i == 1:
                #         re = False
                #     else:
                #         re = True
                img = target[:, 1, :, :, :]
                d, d_log = self.image_discriminator(img, reuse=False, dim_c=1)
                self.D2.append(d)
                self.D2_logits.append(d_log)

            with tf.variable_scope("DIS", reuse=True):
                for i in range(2, n_dep-1):
                    
                    print("-------print target.shape------") ##adl
                    import alog ##adl
                    from pprint import pprint ##adl
                    alog.info("target.shape") ##adl
                    print(">>> type(target.shape) = ", type(target.shape)) ##adl
                    if hasattr(target.shape, "shape"): ##adl
                        print(">>> target.shape.shape", target.shape.shape) ##adl
                    if type(target.shape) is list: ##adl
                        print(">>> len(target.shape) = ", len(target.shape)) ##adl
                        pprint(target.shape) ##adl
                    else: ##adl
                        pprint(target.shape) ##adl
                    print("------------------------\n") ##adl
                    
                    img = target[:, i, :, :, :]
                    d, d_log = self.image_discriminator(img, reuse=True, dim_c=1)
                    self.D2.append(d)
                    self.D2_logits.append(d_log)

            self.D2 = tf.convert_to_tensor(self.D2)
            self.D2_logits = tf.convert_to_tensor(self.D2_logits)

            with tf.variable_scope("DIS", reuse=True):
                # self.D_, self.D_logits_ = self.discriminator(gen_data)
                self.D_, self.D_logits_ = self.video_discriminator(E_pred, reuse=True)
                n_bz = target.shape[0]
                # self.D2_ = tf.zeros(n_bz - 2)
                # self.D_logits2_ = tf.zeros(n_bz - 2)
                self.D2_ = []
                self.D2_logits_ = []
                for i in range(1, n_dep - 1):
                    img = E_pred[:, i, :, :, :]
                    d, d_log = self.image_discriminator(img, reuse=True, dim_c=1)
                    self.D2_.append(d)
                    self.D2_logits_.append(d_log)
                self.D2_ = tf.convert_to_tensor(self.D2_)
                self.D2_logits_ = tf.convert_to_tensor(self.D2_logits_)

            self.L_p = tf.reduce_mean(
                # tf.square(self.G - self.target[:, :, :, self.K:, :])
                tf.square(target[:, 1:-1, :, :, :] - E_pred[:, 1:-1, :, :, :])
            )

            self.L_gdl = gdl(gen_sim, true_sim, 1.)
            self.L_img = self.L_p + self.L_gdl

            self.d_loss_real = tf.reduce_mean(
                tf.nn.sigmoid_cross_entropy_with_logits(
                    logits=self.D_logits, labels=tf.ones_like(self.D)
                )
            )

            self.d_loss_real2 = tf.reduce_mean(
                tf.nn.sigmoid_cross_entropy_with_logits(
                    logits=self.D2_logits, labels=tf.ones_like(self.D2)
                )
            )

            self.d_loss_fake = tf.reduce_mean(
                tf.nn.sigmoid_cross_entropy_with_logits(
                    logits=self.D_logits_, labels=tf.zeros_like(self.D_)
                )
            )

            self.d_loss_fake2 = tf.reduce_mean(
                tf.nn.sigmoid_cross_entropy_with_logits(
                    logits=self.D2_logits_, labels=tf.zeros_like(self.D2_)
                )
            )

            self.d_loss = self.d_loss_real + self.d_loss_fake + self.d_loss_real2 + self.d_loss_fake2

            self.L_GAN = tf.reduce_mean(
                tf.nn.sigmoid_cross_entropy_with_logits(
                    logits=self.D_logits_, labels=tf.ones_like(self.D_)
                )
            ) +  tf.reduce_mean(
                tf.nn.sigmoid_cross_entropy_with_logits(
                    logits=self.D2_logits_, labels=tf.ones_like(self.D2_)
                )
            )

            self.loss_sum = tf.summary.scalar("L_img", self.L_img)
            self.L_p_sum = tf.summary.scalar("L_p", self.L_p)
            self.L_gdl_sum = tf.summary.scalar("L_gdl", self.L_gdl)
            self.L_GAN_sum = tf.summary.scalar("L_GAN", self.L_GAN)

            self.d_loss_sum = tf.summary.scalar("d_loss", self.d_loss)
            self.d_loss_real_sum = tf.summary.scalar("d_loss_real",
                                                     self.d_loss_real)
            self.d_loss_fake_sum = tf.summary.scalar("d_loss_fake",
                                                     self.d_loss_fake)

            self.t_vars = tf.trainable_variables()
            self.g_vars = [var for var in self.t_vars if 'DIS' not in var.name]
            self.d_vars = [var for var in self.t_vars if 'DIS' in var.name]
            num_param = 0.0
            for var in self.g_vars:
                num_param += int(np.prod(var.get_shape()))
            print("Number of parameters: %d" % num_param)

        self.saver = tf.train.Saver(max_to_keep=10)

    def forward(self, diff_in, xt, cell):
        # Initial state
        state_list = [None] * self.num_layers
        for i in range(self.num_layers):
            state_list[i] = tf.zeros([self.batch_size, self.image_size[0] // 8,
                                      self.image_size[1] // 8,
                                      self.hidden_dims[i]])
        reuse = False
        # Encoder
        for t in range(self.K - 1):
            enc_h, res_m = self.motion_enc(diff_in[:, :, :, t, :], reuse=reuse)
            h_dyn, state_list = cell(enc_h, state_list, scope='rbm',
                                     reuse=reuse)
            reuse = True

        pred = []
        # Decoder
        for t in range(self.T):
            if t == 0:
                h_cont, res_c = self.content_enc(xt, reuse=False)
                h_tp1 = self.comb_layers(h_dyn, h_cont, reuse=False)
                res_connect = self.residual(res_m, res_c, reuse=False)
                x_hat = self.dec_cnn(h_tp1, res_connect, reuse=False)
            else:
                enc_h, res_m = self.motion_enc(diff_in, reuse=True)
                h_dyn, state_list = cell(enc_h, state_list, scope='rbm',
                                         reuse=True)
                h_cont, res_c = self.content_enc(xt, reuse=reuse)
                h_tp1 = self.comb_layers(h_dyn, h_cont, reuse=True)
                res_connect = self.residual(res_m, res_c, reuse=True)
                x_hat = self.dec_cnn(h_tp1, res_connect, reuse=True)

            if self.c_dim == 3:
                # Network outputs are BGR so they need to be reversed to use
                # rgb_to_grayscale
                x_hat_rgb = tf.concat(axis=3,
                                      values=[x_hat[:, :, :, 2:3],
                                              x_hat[:, :, :, 1:2],
                                              x_hat[:, :, :, 0:1]])
                xt_rgb = tf.concat(axis=3,
                                   values=[xt[:, :, :, 2:3], xt[:, :, :, 1:2],
                                           xt[:, :, :, 0:1]])

                x_hat_gray = 1. / 255. * tf.image.rgb_to_grayscale(
                    inverse_transform(x_hat_rgb) * 255.
                )
                xt_gray = 1. / 255. * tf.image.rgb_to_grayscale(
                    inverse_transform(xt_rgb) * 255.
                )
            else:
                x_hat_gray = inverse_transform(x_hat)
                xt_gray = inverse_transform(xt)

            diff_in = x_hat_gray - xt_gray
            xt = x_hat

            pred.append(tf.reshape(x_hat, [self.batch_size, self.image_size[0],
                                           self.image_size[1], 1, self.c_dim]))

        return pred

    def image_encoder(self, xt, reuse):
        # xt: H0 x W0 x {1, 3}
        L1 = relu(batch_norm(conv2d(xt, output_dim=64, k_h=4, k_w=4, d_h=2, d_w=2,
                    name='image_conv1', reuse=reuse), name="bn1", reuse=reuse))
        L2 = relu(batch_norm(conv2d(L1, output_dim=64, k_h=3, k_w=3, d_h=1, d_w=1,
                    name='image_conv2', reuse=reuse), name="bn2", reuse=reuse))
        L3 = relu(batch_norm(conv2d(L2, output_dim=128, k_h=4,k_w=4, d_h=2, d_w=2,
                    name='image_conv3', reuse=reuse), name="bn3", reuse=reuse))
        L4 = relu(batch_norm(conv2d(L3, output_dim=128, k_h=3, k_w=3, d_h=1, d_w=1,
                    name='image_conv4', reuse=reuse), name="bn4", reuse=reuse))
        L5 = relu(batch_norm(conv2d(L4, output_dim=256, k_h=4, k_w=4, d_h=2, d_w=2,
                    name='image_conv5', reuse=reuse), name="bn5", reuse=reuse))
        L6 = relu(batch_norm(conv2d(L5, output_dim=256, k_h=3, k_w=3, d_h=1, d_w=1,
                    name='image_conv6', reuse=reuse), name="bn6", reuse=reuse))
        L7 = relu(batch_norm(conv2d(L6, output_dim=64, k_h=3, k_w=3, d_h=1, d_w=1,
                    name='image_conv7', reuse=reuse), name="bn7", reuse=reuse))
        return L7  # H0/8 × W0/8 × 64

    def video_generator(self, xt, reuse, dim_c):
        # xt 16 × H0/8 × W0/8 × 64
        # [3, 16, 8, 8, 64]
        L1 = lrelu(batch_norm(transConv3d(xt, output_dim=256, k_h=3, k_w=3, k_dep=3, d_h=1,
                         d_w=1, d_dep=1,
                         name='video_transconv3d1', reuse=reuse), name="gen_bn1"))
        L2 = lrelu(batch_norm(transConv3d(L1, output_dim=256, k_h=3, k_w=3, k_dep=3, d_h=1,
                         d_w=1, d_dep=1,
                         name='video_transconv3d2', reuse=reuse), name="gen_bn2"))
        L3 = lrelu(batch_norm(transConv3d(L2, output_dim=128, k_h=4, k_w=4, k_dep=3, d_h=2,
                         d_w=2, d_dep=1,
                         name='video_transconv3d3', reuse=reuse), name="gen_bn3"))
        L4 = lrelu(batch_norm(transConv3d(L3, output_dim=128, k_h=3, k_w=3, k_dep=3, d_h=1,
                         d_w=1, d_dep=1,
                         name='video_transconv3d4', reuse=reuse), name="gen_bn4"))
        L5 = lrelu(batch_norm(transConv3d(L4, output_dim=64, k_h=4, k_w=4, k_dep=3, d_h=2, d_w=2,
                         d_dep=1,
                         name='video_transconv3d5', reuse=reuse), name="gen_bn5"))
        L6 = lrelu(batch_norm(transConv3d(L5, output_dim=64, k_h=3, k_w=3, k_dep=3, d_h=1, d_w=1,
                         d_dep=1,
                         name='video_transconv3d6', reuse=reuse), name="gen_bn6"))
        L7 = lrelu(batch_norm(transConv3d(L6, output_dim=dim_c, k_h=4, k_w=4, k_dep=3, d_h=2,
                         d_w=2, d_dep=1,
                         name='video_transconv3d7', reuse=reuse), name="gen_bn7"))
        return L7  # 16 × H0 × W0 × {1, 3}

    def video_discriminator(self, xt, reuse):
        # 16 × H0 × W0 × {1, 3}
        L1 = lrelu(layer_norm(conv3d(xt, output_dim=64, k_h=4, k_w=4, k_dep=4, d_h=2,
                    d_w=2, d_dep=1,
                    name='video_conv3d1', reuse=reuse), name="video_ln1"))
        L2 = lrelu(layer_norm(conv3d(L1, output_dim=128, k_h=4, k_w=4, k_dep=4, d_h=2,
                    d_w=2, d_dep=1,
                    name='video_conv3d2', reuse=reuse), name="video_ln2"))
        L3 = lrelu(layer_norm(conv3d(L2, output_dim=256, k_h=4, k_w=4, k_dep=4, d_h=2,
                    d_w=2, d_dep=1,
                    name='video_conv3d3', reuse=reuse), name="video_ln3"))
        L4 = lrelu(layer_norm(conv3d(L3, output_dim=512, k_h=4, k_w=4, k_dep=4, d_h=2,
                    d_w=2, d_dep=1,
                    name='video_conv3d4', reuse=reuse), name="video_ln4"))

        h = linear(tf.reshape(L4, [L4.shape[0], -1]), 1, 'dis_video')

        return tf.sigmoid(h), h

    def short_cut(self, xt, C, reuse, name):
        l1 = AvgPooling("avg_pool", xt, [2, 2])
        l2 = lrelu(layer_norm(conv2d(l1, C, k_h=1, k_w=1, d_h=1, d_w=1, name=name, reuse=reuse), name=name, reuse=reuse))
        return l2

    def image_discriminator(self, xt, reuse, dim_c):
        # xt H0 × W0 × {1, 3}
        L1 = lrelu(layer_norm(conv2d(xt, dim_c, k_h=3, k_w=3,
                    d_h=1, d_w=1, name='image_disconv1', reuse=reuse), name="image_ln1"))
        
        x2 = self.short_cut(xt, 64, reuse, name="shortcut1")
        L2 = lrelu(layer_norm(conv2d(L1, 64, k_h=4, k_w=4,
                    d_h=2, d_w=2, name='image_disconv2',
                    reuse=reuse), name="image_ln2")) + x2
        L3 = lrelu(layer_norm(conv2d(L2, 64, k_h=3, k_w=3,
                    d_h=1, d_w=1, name='image_disconv3', reuse=reuse), name="image_ln3"))

        x4 = self.short_cut(x2, 128, reuse, name="shortcut2")
        L4 = lrelu(layer_norm(conv2d(L3, 128, k_h=4, k_w=4, d_h=2, d_w=2, name='image_disconv4',
                    reuse=reuse), name="image_ln4")) + x4

        L5 = lrelu(layer_norm(conv2d(L4, 128, k_h=3, k_w=3, d_h=1, d_w=1, name='image_disconv5',
                    reuse=reuse), name="image_ln5"))
        x6 = self.short_cut(x4, 256, reuse, name="shortcut3")
        L6 = lrelu(layer_norm(conv2d(L5, 256, k_h=4, k_w=4, d_h=2, d_w=2, name='image_disconv6',
                    reuse=reuse), name="image_ln6")) + x6
        L7 = lrelu(layer_norm(conv2d(L6, 256, k_h=3, k_w=3, d_h=1, d_w=1, name='image_disconv7',
                    reuse=reuse), name="image_ln7"))
        x8 = self.short_cut(x6, 512, reuse, name="shortcut4")
        L8 = lrelu(layer_norm(conv2d(L7, 512, k_h=4, k_w=4, d_h=2, d_w=2, name='image_disconv8',
                    reuse=reuse), name="image_ln8")) + x8
        h = linear(tf.reshape(L8, [L8.shape[0], -1]), 1, 'dis_image', reuse=reuse)
        return tf.sigmoid(h), h


    def get_layer_u(self, u, l_id, reuse, n_bz):
        T_l = [4, 4, 4, 4, 4, 4, 4, 4,
               8, 8, 8, 8, 8, 8, 8, 8,
               16, 16, 16, 16, 16, 16, 16, 16]
        #
        # print("-------print u.shape------") ##adl
        # import alog ##adl
        # from pprint import pprint ##adl
        # alog.info("u.shape") ##adl
        # print(">>> type(u.shape) = ", type(u.shape)) ##adl
        # if hasattr(u.shape, "shape"): ##adl
        #     print(">>> u.shape.shape", u.shape.shape) ##adl
        # if type(u.shape) is list: ##adl
        #     print(">>> len(u.shape) = ", len(u.shape)) ##adl
        #     pprint(u.shape) ##adl
        # else: ##adl
        #     pprint(u.shape) ##adl
        # print("------------------------\n") ##adl
        # raise ValueError
        
        with tf.variable_scope("layer_u" + str(l_id), reuse=reuse):
            w = tf.get_variable('w',
                                [T_l[l_id] * self.dim_C, self.dim_D],
                                initializer=tf.contrib.layers.xavier_initializer())
            biases = tf.get_variable('biases', [T_l[l_id] * self.dim_C, 1],
                                     initializer=tf.constant_initializer(0.0))
            
            var_list = [w.shape, u.shape] ##adl
            for idx,x in enumerate(var_list): ##adl
                var_names = "w.shape, u.shape".split(",") ##adl
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
            
            ul = tf.matmul(w, u) + biases

            ul2 = tf.reshape(ul, [T_l[l_id], self.dim_C])
            return ul2


    def _latent_generator(self, z_pre, E_xs, E_xe, u, l_id, reuse):
        # u: Dx1
        # z_pre: N x T2 x H x W x C
        # E_xs, E_xe: N x 1 x H × W × C, N x 1 x H × W × C
        [N, T2, H, W, C]= z_pre.get_shape().as_list()
        with tf.variable_scope("latent_layer" + str(l_id), reuse=reuse):
            u_l = self.get_layer_u(u, l_id, reuse, z_pre.shape[0]) # [4,64]
            u_l = tf.expand_dims(u_l, 0)
            print("-------print u_l.shape------") ##adl
            import alog ##adl
            from pprint import pprint ##adl
            alog.info("u_l.shape") ##adl
            print(">>> type(u_l.shape) = ", type(u_l.shape)) ##adl
            if hasattr(u_l.shape, "shape"): ##adl
                print(">>> u_l.shape.shape", u_l.shape.shape) ##adl
            if type(u_l.shape) is list: ##adl
                print(">>> len(u_l.shape) = ", len(u_l.shape)) ##adl
                pprint(u_l.shape) ##adl
            else: ##adl
                pprint(u_l.shape) ##adl
            print("------------------------\n") ##adl
            
            out = conv1d(u_l, self.dim_C * 3, k=3, d=1, name="g_s_e_3conv")
            _g_s, _g_e, n_l = tf.split(out, 3, axis=-1)
            g_s = tf.sigmoid(_g_s)
            g_e = tf.sigmoid(_g_e)
            var_list = [g_s.shape, E_xs.shape, z_pre.shape, n_l.shape] ##adl
            for idx,x in enumerate(var_list): ##adl
                var_names = "g_s.shape, E_xs.shape, z_pre.shape, n_l.shape".split(",") ##adl
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

            # g_s [1, 2, 64]
            g_s = tf.reshape(g_s, [1, T2, 1, 1, C])
            g_e = tf.reshape(g_e, [1, T2, 1, 1, C])
            n_l = tf.reshape(n_l, [1, T2, 1, 1, C])
            # raise ValueError
            z_in = g_s * E_xs + g_e * E_xe + tf.maximum(1 - g_s - g_e, 0) * z_pre + n_l
            # raise ValueError
            z_in2 = conv3d(z_in, self.dim_C,  k_h=3, k_w=3, k_dep=3, name="conv3d1")
            z_in2 = tf.nn.leaky_relu(z_in2)
            z_in3 = conv3d(z_in2, self.dim_C,  k_h=3, k_w=3, k_dep=3, name="conv3d2")
            z_l = tf.nn.leaky_relu(z_pre + z_in3)

            return z_l

    def latent_gen(self, E_xs, E_xe, u, reuse):
        # u: Dx1
        # E_xs, E_xe: N x T x H × W × C, N x T x H × W × C

        print("-------print E_xs.shape------") ##adl
        import alog ##adl
        from pprint import pprint ##adl
        alog.info("E_xs.shape") ##adl
        print(">>> type(E_xs.shape) = ", type(E_xs.shape)) ##adl
        if hasattr(E_xs.shape, "shape"): ##adl
            print(">>> E_xs.shape.shape", E_xs.shape.shape) ##adl
        if type(E_xs.shape) is list: ##adl
            print(">>> len(E_xs.shape) = ", len(E_xs.shape)) ##adl
            pprint(E_xs.shape) ##adl
        else: ##adl
            pprint(E_xs.shape) ##adl
        print("------------------------\n") ##adl
        
        
        print("-------print E_xe.shape------") ##adl
        import alog ##adl
        from pprint import pprint ##adl
        alog.info("E_xe.shape") ##adl
        print(">>> type(E_xe.shape) = ", type(E_xe.shape)) ##adl
        if hasattr(E_xe.shape, "shape"): ##adl
            print(">>> E_xe.shape.shape", E_xe.shape.shape) ##adl
        if type(E_xe.shape) is list: ##adl
            print(">>> len(E_xe.shape) = ", len(E_xe.shape)) ##adl
            pprint(E_xe.shape) ##adl
        else: ##adl
            pprint(E_xe.shape) ##adl
        print("------------------------\n") ##adl
        
        
        z_0 = tf.concat([E_xs, E_xe], 1)
        var_list = [z_0.shape] ##adl
        for idx,x in enumerate(var_list): ##adl
            var_names = "z_0.shape".split(",") ##adl
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
        
        z_pre = z_0 #[10, 2, 64, 64, 1]
        for l_id in range(0, 24):
            if (l_id + 1) in [1, 9, 17]:
                # upsampling z_pre
                up_sampling = tf.keras.layers.UpSampling1D(2)
                [n_bz, n_t, n_h, n_w, n_c] = z_pre.get_shape().as_list()
                z_pre0 = tf.reshape(z_pre, [n_bz, n_t, -1])
                z_pre1 = up_sampling(z_pre0)
                n_t_new = z_pre1.shape[1]
                z_pre2 = tf.reshape(z_pre1,[n_bz, n_t_new, n_h, n_w, n_c])
                z_pre = z_pre2

            z = self._latent_generator(z_pre, E_xs, E_xe, u, l_id, reuse)
            z_pre = z
        return z


    def motion_enc(self, diff_in, reuse):
        res_in = []
        conv1 = relu(conv2d(diff_in, output_dim=self.gf_dim, k_h=5, k_w=5,
                            d_h=1, d_w=1, name='dyn_conv1', reuse=reuse))
        res_in.append(conv1)
        pool1 = MaxPooling(conv1, [2, 2])

        conv2 = relu(conv2d(pool1, output_dim=self.gf_dim * 2, k_h=5, k_w=5,
                            d_h=1, d_w=1, name='dyn_conv2', reuse=reuse))
        res_in.append(conv2)
        pool2 = MaxPooling(conv2, [2, 2])

        conv3 = relu(conv2d(pool2, output_dim=self.gf_dim * 4, k_h=7, k_w=7,
                            d_h=1, d_w=1, name='dyn_conv3', reuse=reuse))
        res_in.append(conv3)
        pool3 = MaxPooling(conv3, [2, 2])

        return pool3, res_in

    def content_enc(self, xt, reuse):
        res_in = []
        conv1_1 = relu(conv2d(xt, output_dim=self.gf_dim, k_h=3, k_w=3,
                              d_h=1, d_w=1, name='cont_conv1_1', reuse=reuse))
        conv1_2 = relu(conv2d(conv1_1, output_dim=self.gf_dim, k_h=3, k_w=3,
                              d_h=1, d_w=1, name='cont_conv1_2', reuse=reuse))
        res_in.append(conv1_2)
        pool1 = MaxPooling(conv1_2, [2, 2])

        conv2_1 = relu(conv2d(pool1, output_dim=self.gf_dim * 2, k_h=3, k_w=3,
                              d_h=1, d_w=1, name='cont_conv2_1', reuse=reuse))
        conv2_2 = relu(conv2d(conv2_1, output_dim=self.gf_dim * 2, k_h=3, k_w=3,
                              d_h=1, d_w=1, name='cont_conv2_2', reuse=reuse))
        res_in.append(conv2_2)
        pool2 = MaxPooling(conv2_2, [2, 2])

        conv3_1 = relu(conv2d(pool2, output_dim=self.gf_dim * 4, k_h=3, k_w=3,
                              d_h=1, d_w=1, name='cont_conv3_1', reuse=reuse))
        conv3_2 = relu(conv2d(conv3_1, output_dim=self.gf_dim * 4, k_h=3, k_w=3,
                              d_h=1, d_w=1, name='cont_conv3_2', reuse=reuse))
        conv3_3 = relu(conv2d(conv3_2, output_dim=self.gf_dim * 4, k_h=3, k_w=3,
                              d_h=1, d_w=1, name='cont_conv3_3', reuse=reuse))
        res_in.append(conv3_3)
        pool3 = MaxPooling(conv3_3, [2, 2])
        return pool3, res_in

    def comb_layers(self, h_dyn, h_cont, reuse=False):
        comb1 = relu(conv2d(tf.concat(axis=3, values=[h_dyn, h_cont]),
                            output_dim=self.gf_dim * 4, k_h=3, k_w=3,
                            d_h=1, d_w=1, name='comb1', reuse=reuse))
        comb2 = relu(conv2d(comb1, output_dim=self.gf_dim * 2, k_h=3, k_w=3,
                            d_h=1, d_w=1, name='comb2', reuse=reuse))
        h_comb = relu(conv2d(comb2, output_dim=self.gf_dim * 4, k_h=3, k_w=3,
                             d_h=1, d_w=1, name='h_comb', reuse=reuse))
        return h_comb

    def residual(self, input_dyn, input_cont, reuse=False):
        n_layers = len(input_dyn)
        res_out = []
        for l in range(n_layers):
            input_ = tf.concat(axis=3, values=[input_dyn[l], input_cont[l]])
            out_dim = input_cont[l].get_shape()[3]
            res1 = relu(conv2d(input_, output_dim=out_dim,
                               k_h=3, k_w=3, d_h=1, d_w=1,
                               name='res' + str(l) + '_1', reuse=reuse))
            res2 = conv2d(res1, output_dim=out_dim, k_h=3, k_w=3,
                          d_h=1, d_w=1, name='res' + str(l) + '_2', reuse=reuse)
            res_out.append(res2)
        return res_out

    def dec_cnn(self, h_comb, res_connect, reuse=False):
        shapel3 = [self.batch_size, self.image_size[0] // 4,
                   self.image_size[1] // 4, self.gf_dim * 4]
        shapeout3 = [self.batch_size, self.image_size[0] // 4,
                     self.image_size[1] // 4, self.gf_dim * 2]
        depool3 = FixedUnPooling(h_comb, [2, 2])
        deconv3_3 = relu(deconv2d(relu(tf.add(depool3, res_connect[2])),
                                  output_shape=shapel3, k_h=3, k_w=3,
                                  d_h=1, d_w=1, name='dec_deconv3_3',
                                  reuse=reuse))
        deconv3_2 = relu(deconv2d(deconv3_3, output_shape=shapel3, k_h=3, k_w=3,
                                  d_h=1, d_w=1, name='dec_deconv3_2',
                                  reuse=reuse))
        deconv3_1 = relu(
            deconv2d(deconv3_2, output_shape=shapeout3, k_h=3, k_w=3,
                     d_h=1, d_w=1, name='dec_deconv3_1', reuse=reuse))

        shapel2 = [self.batch_size, self.image_size[0] // 2,
                   self.image_size[1] // 2, self.gf_dim * 2]
        shapeout3 = [self.batch_size, self.image_size[0] // 2,
                     self.image_size[1] // 2, self.gf_dim]
        depool2 = FixedUnPooling(deconv3_1, [2, 2])
        deconv2_2 = relu(deconv2d(relu(tf.add(depool2, res_connect[1])),
                                  output_shape=shapel2, k_h=3, k_w=3,
                                  d_h=1, d_w=1, name='dec_deconv2_2',
                                  reuse=reuse))
        deconv2_1 = relu(
            deconv2d(deconv2_2, output_shape=shapeout3, k_h=3, k_w=3,
                     d_h=1, d_w=1, name='dec_deconv2_1', reuse=reuse))

        shapel1 = [self.batch_size, self.image_size[0],
                   self.image_size[1], self.gf_dim]
        shapeout1 = [self.batch_size, self.image_size[0],
                     self.image_size[1], self.c_dim]
        depool1 = FixedUnPooling(deconv2_1, [2, 2])
        deconv1_2 = relu(deconv2d(relu(tf.add(depool1, res_connect[0])),
                                  output_shape=shapel1, k_h=3, k_w=3, d_h=1,
                                  d_w=1,
                                  name='dec_deconv1_2', reuse=reuse))
        xtp1 = tanh(deconv2d(deconv1_2, output_shape=shapeout1, k_h=3, k_w=3,
                             d_h=1, d_w=1, name='dec_deconv1_1', reuse=reuse))
        return xtp1

    def discriminator(self, image):
        h0 = lrelu(conv2d(image, self.df_dim, name='dis_h0_conv'))
        h1 = lrelu(batch_norm(conv2d(h0, self.df_dim * 2, name='dis_h1_conv'),
                              "bn1"))
        h2 = lrelu(batch_norm(conv2d(h1, self.df_dim * 4, name='dis_h2_conv'),
                              "bn2"))
        h3 = lrelu(batch_norm(conv2d(h2, self.df_dim * 8, name='dis_h3_conv'),
                              "bn3"))
        h = linear(tf.reshape(h3, [self.batch_size, -1]), 1, 'dis_h3_lin')

        return tf.nn.sigmoid(h), h

    def save(self, sess, checkpoint_dir, step):
        model_name = "MCNET.model"

        if not os.path.exists(checkpoint_dir):
            os.makedirs(checkpoint_dir)

        self.saver.save(sess, os.path.join(checkpoint_dir, model_name),
                        global_step=step)

    def load(self, sess, checkpoint_dir, model_name=None):
        print(" [*] Reading checkpoints...")

        ckpt = tf.train.get_checkpoint_state(checkpoint_dir)
        if ckpt and ckpt.model_checkpoint_path:
            ckpt_name = os.path.basename(ckpt.model_checkpoint_path)
            if model_name is None: model_name = ckpt_name
            self.saver.restore(sess, os.path.join(checkpoint_dir, model_name))
            print("     Loaded model: " + str(model_name))
            return True, model_name
        else:
            return False, None
