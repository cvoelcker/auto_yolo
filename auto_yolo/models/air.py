# Copyright 2016 The TensorFlow Authors. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ==============================================================================
import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as patches

from dps import cfg
from dps.utils import Param
from dps.utils.tf import build_scheduled_value, RenderHook, tf_mean_sum

from auto_yolo.models.core import (
    VariationalAutoencoder, normal_vae, mAP, xent_loss,
    concrete_binary_pre_sigmoid_sample, concrete_binary_sample_kl)


# ------ transformer.py -------


def transformer(U, theta, out_size, name='SpatialTransformer', **kwargs):
    """Spatial Transformer Layer

    Implements a spatial transformer layer as described in [1]_.
    Based on [2]_ and edited by David Dao for Tensorflow.
    Parameters
    ----------
    U : float
        The output of a convolutional net should have the
        shape [num_batch, height, width, num_channels].
    theta: float
        The output of the
        localisation network should be [num_batch, 6].
    out_size: tuple of two ints
        The size of the output of the network (height, width)
    References
    ----------
    .. [1]  Spatial Transformer Networks
            Max Jaderberg, Karen Simonyan, Andrew Zisserman, Koray Kavukcuoglu
            Submitted on 5 Jun 2015
    .. [2]  https://github.com/skaae/transformer_network/blob/master/transformerlayer.py
    Notes
    -----
    To initialize the network to the identity transform init
    ``theta`` to :
        identity = np.array([[1., 0., 0.],
                             [0., 1., 0.]])
        identity = identity.flatten()
        theta = tf.Variable(initial_value=identity)
    """

    def _repeat(x, n_repeats):
        with tf.variable_scope('_repeat'):
            rep = tf.transpose(
                tf.expand_dims(tf.ones(shape=tf.stack([n_repeats, ])), 1), [1, 0])
            rep = tf.cast(rep, 'int32')
            x = tf.matmul(tf.reshape(x, (-1, 1)), rep)
            return tf.reshape(x, [-1])

    def _interpolate(im, x, y, out_size):
        with tf.variable_scope('_interpolate'):
            # constants
            num_batch = tf.shape(im)[0]
            height = tf.shape(im)[1]
            width = tf.shape(im)[2]
            channels = tf.shape(im)[3]

            x = tf.cast(x, 'float32')
            y = tf.cast(y, 'float32')
            height_f = tf.cast(height, 'float32')
            width_f = tf.cast(width, 'float32')
            out_height = out_size[0]
            out_width = out_size[1]
            zero = tf.zeros([], dtype='int32')
            max_y = tf.cast(tf.shape(im)[1] - 1, 'int32')
            max_x = tf.cast(tf.shape(im)[2] - 1, 'int32')

            # scale indices from [-1, 1] to [0, width/height]
            x = (x + 1.0)*(width_f-1.001) / 2.0
            y = (y + 1.0)*(height_f-1.001) / 2.0

            # do sampling
            x0 = tf.cast(tf.floor(x), 'int32')
            x1 = x0 + 1
            y0 = tf.cast(tf.floor(y), 'int32')
            y1 = y0 + 1

            x0 = tf.clip_by_value(x0, zero, max_x)
            x1 = tf.clip_by_value(x1, zero, max_x)
            y0 = tf.clip_by_value(y0, zero, max_y)
            y1 = tf.clip_by_value(y1, zero, max_y)
            dim2 = width
            dim1 = width*height
            base = _repeat(tf.range(num_batch)*dim1, out_height*out_width)
            base_y0 = base + y0*dim2
            base_y1 = base + y1*dim2
            idx_a = base_y0 + x0
            idx_b = base_y1 + x0
            idx_c = base_y0 + x1
            idx_d = base_y1 + x1

            # use indices to lookup pixels in the flat image and restore
            # channels dim
            im_flat = tf.reshape(im, tf.stack([-1, channels]))
            im_flat = tf.cast(im_flat, 'float32')
            Ia = tf.gather(im_flat, idx_a)
            Ib = tf.gather(im_flat, idx_b)
            Ic = tf.gather(im_flat, idx_c)
            Id = tf.gather(im_flat, idx_d)

            # and finally calculate interpolated values
            x0_f = tf.cast(x0, 'float32')
            x1_f = tf.cast(x1, 'float32')
            y0_f = tf.cast(y0, 'float32')
            y1_f = tf.cast(y1, 'float32')
            wa = tf.expand_dims(((x1_f-x) * (y1_f-y)), 1)
            wb = tf.expand_dims(((x1_f-x) * (y-y0_f)), 1)
            wc = tf.expand_dims(((x-x0_f) * (y1_f-y)), 1)
            wd = tf.expand_dims(((x-x0_f) * (y-y0_f)), 1)
            output = tf.add_n([wa*Ia, wb*Ib, wc*Ic, wd*Id])
            return output

    def _meshgrid(height, width):
        with tf.variable_scope('_meshgrid'):
            # This should be equivalent to:
            #  x_t, y_t = np.meshgrid(np.linspace(-1, 1, width),
            #                         np.linspace(-1, 1, height))
            #  ones = np.ones(np.prod(x_t.shape))
            #  grid = np.vstack([x_t.flatten(), y_t.flatten(), ones])
            x_t = tf.matmul(tf.ones(shape=tf.stack([height, 1])),
                            tf.transpose(tf.expand_dims(tf.linspace(-1.0, 1.0, width), 1), [1, 0]))
            y_t = tf.matmul(tf.expand_dims(tf.linspace(-1.0, 1.0, height), 1),
                            tf.ones(shape=tf.stack([1, width])))

            x_t_flat = tf.reshape(x_t, (1, -1))
            y_t_flat = tf.reshape(y_t, (1, -1))

            ones = tf.ones_like(x_t_flat)
            grid = tf.concat(axis=0, values=[x_t_flat, y_t_flat, ones])
            return grid

    def _transform(theta, input_dim, out_size):
        with tf.variable_scope('_transform'):
            num_batch = tf.shape(input_dim)[0]
            num_channels = tf.shape(input_dim)[3]
            theta = tf.reshape(theta, (-1, 2, 3))
            theta = tf.cast(theta, 'float32')

            # grid of (x_t, y_t, 1), eq (1) in ref [1]
            out_height = out_size[0]
            out_width = out_size[1]
            grid = _meshgrid(out_height, out_width)
            grid = tf.expand_dims(grid, 0)
            grid = tf.reshape(grid, [-1])
            grid = tf.tile(grid, tf.stack([num_batch]))
            grid = tf.reshape(grid, tf.stack([num_batch, 3, -1]))

            # Transform A x (x_t, y_t, 1)^T -> (x_s, y_s)
            T_g = tf.matmul(theta, grid)
            x_s = tf.slice(T_g, [0, 0, 0], [-1, 1, -1])
            y_s = tf.slice(T_g, [0, 1, 0], [-1, 1, -1])
            x_s_flat = tf.reshape(x_s, [-1])
            y_s_flat = tf.reshape(y_s, [-1])

            input_transformed = _interpolate(
                input_dim, x_s_flat, y_s_flat,
                out_size)

            output = tf.reshape(
                input_transformed, tf.stack([num_batch, out_height, out_width, num_channels]))
            return output

    with tf.variable_scope(name):
        output = _transform(theta, U, out_size)
        return output


class AIR_AP(object):
    keys_accessed = "scale shift predicted_n_digits annotations n_annotations"

    def __init__(self, iou_threshold=None):
        if iou_threshold is not None:
            try:
                iou_threshold = list(iou_threshold)
            except (TypeError, ValueError):
                iou_threshold = [float(iou_threshold)]
        self.iou_threshold = iou_threshold

    def __call__(self, _tensors, updater):
        network = updater.network
        w, h = np.split(_tensors['scale'], 2, axis=2)
        x, y = np.split(_tensors['shift'], 2, axis=2)
        predicted_n_digits = _tensors['predicted_n_digits']
        annotations = _tensors["annotations"]
        n_annotations = _tensors["n_annotations"]

        batch_size = w.shape[0]

        transformed_x = 0.5 * (x + 1.)
        transformed_y = 0.5 * (y + 1.)

        height = h * network.image_height
        width = w * network.image_width

        top = network.image_height * transformed_y - height / 2
        left = network.image_width * transformed_x - width / 2

        bottom = top + height
        right = left + width

        ground_truth_boxes = []
        predicted_boxes = []

        for idx in range(batch_size):
            _a = [
                [0, *rest]
                for (valid, cls, *rest), _
                in zip(annotations[idx], range(n_annotations[idx]))
                if valid]
            ground_truth_boxes.append(_a)

            _predicted_boxes = []

            for t in range(predicted_n_digits[idx]):
                _predicted_boxes.append(
                    [0, 1,
                     top[idx, t, 0],
                     bottom[idx, t, 0],
                     left[idx, t, 0],
                     right[idx, t, 0]])

            predicted_boxes.append(_predicted_boxes)

        return mAP(
            predicted_boxes, ground_truth_boxes, n_classes=1,
            iou_threshold=self.iou_threshold)


class AIR_Network(VariationalAutoencoder):
    max_time_steps = Param()
    run_all_time_steps = Param(help="If true, always run for `max_time_steps` and don't predict `z_pres`")
    object_shape = Param()

    scale_prior_mean = Param()
    scale_prior_std = Param()

    shift_prior_mean = Param()
    shift_prior_std = Param()

    z_pres_prior_log_odds = Param()
    z_pres_temperature = Param()
    stopping_threshold = Param()

    training_wheels = Param()
    kl_weight = Param()

    difference_air = Param()

    complete_rnn_input = Param()

    image_encoder = None
    cell = None
    output_network = None
    object_encoder = None
    object_decoder = None

    def __init__(self, env, updater, scope=None, **kwargs):
        super(AIR_Network, self).__init__(env, updater, scope=scope, **kwargs)

        ap_iou_values = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]
        self.eval_funcs = {"AP_at_point_{}".format(int(10 * v)): AIR_AP(v) for v in ap_iou_values}
        self.eval_funcs["AP"] = AIR_AP(ap_iou_values)

        self.training_wheels = build_scheduled_value(self.training_wheels, "training_wheels")

        self.scale_prior_mean = build_scheduled_value(self.scale_prior_mean, "scale_prior_mean")
        self.scale_prior_std = build_scheduled_value(self.scale_prior_std, "scale_prior_std")

        self.shift_prior_mean = build_scheduled_value(self.shift_prior_mean, "shift_prior_mean")
        self.shift_prior_std = build_scheduled_value(self.shift_prior_std, "shift_prior_std")

        self.z_pres_prior_log_odds = build_scheduled_value(self.z_pres_prior_log_odds, "z_pres_prior_log_odds")

    def apply_training_wheel(self, signal):
        return (
            self.training_wheels * tf.stop_gradient(signal)
            + (1-self.training_wheels) * signal)

    def apply_fixed_value(self, key, signal):
        value = self.fixed_values.get(key, None)
        if value is not None:
            return value * tf.ones_like(signal)
        else:
            return signal

    def build_representation(self):
        # --- process input ---

        if self.image_encoder is None:
            self.image_encoder = cfg.build_image_encoder(scope="image_encoder")
            if "image_encoder" in self.fixed_weights:
                self.image_encoder.fix_variables()

        if self.cell is None:
            self.cell = cfg.build_cell(scope="cell")
            if "cell" in self.fixed_weights:
                self.cell.fix_variables()

        if self.output_network is None:
            self.output_network = cfg.build_output_network(scope="output_network")
            if "output" in self.fixed_weights:
                self.output_network.fix_variables()

        if self.object_encoder is None:
            self.object_encoder = cfg.build_object_encoder(scope="object_encoder")
            if "object_encoder" in self.fixed_weights:
                self.object_encoder.fix_variables()

        if self.object_decoder is None:
            self.object_decoder = cfg.build_object_decoder(scope="object_decoder")
            if "object_decoder" in self.fixed_weights:
                self.object_decoder.fix_variables()

        self.target_n_digits = self._tensors["n_valid_annotations"]

        if not self.difference_air:
            encoded_inp = self.image_encoder(
                self._tensors["inp"], 0, self.is_training)
            self.encoded_inp = tf.layers.flatten(encoded_inp)

        # --- condition of while-loop ---

        def cond(step, stopping_sum, *_):
            return tf.logical_and(
                tf.less(step, self.max_time_steps),
                tf.reduce_any(tf.less(stopping_sum, self.stopping_threshold))
            )

        # --- body of while-loop ---

        def body(step, stopping_sum, prev_state,
                 running_recon, kl_loss, running_digits,
                 scale_ta, scale_kl_ta, scale_std_ta,
                 shift_ta, shift_kl_ta, shift_std_ta,
                 attr_ta, attr_kl_ta, attr_std_ta,
                 z_pres_ta, z_pres_probs_ta, z_pres_kl_ta,
                 vae_input_ta, vae_output_ta,
                 scale, shift, attr, z_pres):

            if self.difference_air:
                inp = (
                    self._tensors["inp"]
                    - tf.reshape(running_recon, (self.batch_size, *self.obs_shape))
                )
                encoded_inp = self.image_encoder(inp, 0, self.is_training)
                encoded_inp = tf.layers.flatten(encoded_inp)
            else:
                encoded_inp = self.encoded_inp

            if self.complete_rnn_input:
                rnn_input = tf.concat([encoded_inp, scale, shift, attr, z_pres], axis=1)
            else:
                rnn_input = encoded_inp

            hidden_rep, next_state = self.cell(rnn_input, prev_state)

            outputs = self.output_network(hidden_rep, 9, self.is_training)

            (scale_mean, scale_log_std,
             shift_mean, shift_log_std,
             z_pres_log_odds) = tf.split(outputs, [2, 2, 2, 2, 1], axis=1)

            # --- scale ---

            scale_std = tf.exp(scale_log_std)

            scale_mean = self.apply_fixed_value("scale_mean", scale_mean)
            scale_std = self.apply_fixed_value("scale_std", scale_std)

            scale_logits, scale_kl = normal_vae(
                scale_mean, scale_std, self.scale_prior_mean, self.scale_prior_std)
            scale_kl = tf.reduce_sum(scale_kl, axis=1, keepdims=True)
            scale = tf.nn.sigmoid(tf.clip_by_value(scale_logits, -10, 10))

            # --- shift ---

            shift_std = tf.exp(shift_log_std)

            shift_mean = self.apply_fixed_value("shift_mean", shift_mean)
            shift_std = self.apply_fixed_value("shift_std", shift_std)

            shift_logits, shift_kl = normal_vae(
                shift_mean, shift_std, self.shift_prior_mean, self.shift_prior_std)
            shift_kl = tf.reduce_sum(shift_kl, axis=1, keepdims=True)
            shift = tf.nn.tanh(tf.clip_by_value(shift_logits, -10, 10))

            # --- Extract windows from scene ---

            w, h = scale[:, 0:1], scale[:, 1:2]
            x, y = shift[:, 0:1], shift[:, 1:2]

            theta = tf.concat([w, tf.zeros_like(w), x, tf.zeros_like(h), h, y], axis=1)
            theta = tf.reshape(theta, (-1, 2, 3))

            vae_input = transformer(self._tensors["inp"], theta, self.object_shape)

            # This is a necessary reshape, as the output of transformer will have unknown dims
            vae_input = tf.reshape(vae_input, (self.batch_size, *self.object_shape, self.image_depth))

            # --- Apply Object-level VAE (object encoder/object decoder) to windows ---

            attr = self.object_encoder(vae_input, 2*self.A, self.is_training)
            attr_mean, attr_log_std = tf.split(attr, 2, axis=1)
            attr_std = tf.exp(attr_log_std)
            attr, attr_kl = normal_vae(attr_mean, attr_std, self.attr_prior_mean, self.attr_prior_std)
            attr_kl = tf.reduce_sum(attr_kl, axis=1, keepdims=True)

            vae_output = self.object_decoder(
                attr, self.object_shape[0] * self.object_shape[1] * self.image_depth, self.is_training)
            vae_output = tf.nn.sigmoid(tf.clip_by_value(vae_output, -10, 10))

            # --- Place reconstructed objects in image ---

            theta_inverse = tf.concat([1. / w, tf.zeros_like(w), -x / w, tf.zeros_like(h), 1. / h, -y / h], axis=1)
            theta_inverse = tf.reshape(theta_inverse, (-1, 2, 3))

            vae_output_transformed = transformer(
                tf.reshape(vae_output, (self.batch_size, *self.object_shape, self.image_depth,)),
                theta_inverse, self.obs_shape[:2]
            )
            vae_output_transformed = tf.reshape(
                vae_output_transformed, [self.batch_size, self.image_height * self.image_width * self.image_depth])

            # --- z_pres ---

            if self.run_all_time_steps:
                z_pres = tf.ones_like(z_pres_log_odds)
                z_pres_prob = tf.ones_like(z_pres_log_odds)
                z_pres_kl = tf.zeros_like(z_pres_log_odds)
            else:
                z_pres_log_odds = tf.clip_by_value(z_pres_log_odds, -10, 10)

                z_pres_pre_sigmoid = concrete_binary_pre_sigmoid_sample(
                    z_pres_log_odds, self.z_pres_temperature
                )
                z_pres = tf.nn.sigmoid(z_pres_pre_sigmoid)
                z_pres = (
                    self.float_is_training * z_pres
                    + (1 - self.float_is_training) * tf.round(z_pres)
                )
                z_pres_prob = tf.nn.sigmoid(z_pres_log_odds)
                z_pres_kl = concrete_binary_sample_kl(
                    z_pres_pre_sigmoid,
                    self.z_pres_prior_log_odds, self.z_pres_temperature,
                    z_pres_log_odds, self.z_pres_temperature
                )

            stopping_sum += (1.0 - z_pres)
            alive = tf.less(stopping_sum, self.stopping_threshold)
            running_digits += tf.to_int32(alive)

            # --- adjust reconstruction ---

            running_recon += tf.where(
                tf.tile(alive, (1, vae_output_transformed.shape[1])),
                z_pres * vae_output_transformed, tf.zeros_like(running_recon)
            )

            # --- add kl to loss ---

            kl_loss += tf.where(
                alive, scale_kl, tf.zeros_like(kl_loss)
            )
            kl_loss += tf.where(
                alive, shift_kl, tf.zeros_like(kl_loss)
            )
            kl_loss += tf.where(
                alive, attr_kl, tf.zeros_like(kl_loss)
            )
            kl_loss += tf.where(
                alive, z_pres_kl, tf.zeros_like(kl_loss)
            )

            # --- record values ---

            scale_ta = scale_ta.write(scale_ta.size(), scale)
            scale_kl_ta = scale_kl_ta.write(scale_kl_ta.size(), scale_kl)
            scale_std_ta = scale_std_ta.write(scale_std_ta.size(), scale_std)

            shift_ta = shift_ta.write(shift_ta.size(), shift)
            shift_kl_ta = shift_kl_ta.write(shift_kl_ta.size(), shift_kl)
            shift_std_ta = shift_std_ta.write(shift_std_ta.size(), shift_std)

            attr_ta = attr_ta.write(attr_ta.size(), attr)
            attr_kl_ta = attr_kl_ta.write(attr_kl_ta.size(), attr_kl)
            attr_std_ta = attr_std_ta.write(attr_std_ta.size(), attr_std)

            vae_input_ta = vae_input_ta.write(vae_input_ta.size(), tf.layers.flatten(vae_input))
            vae_output_ta = vae_output_ta.write(vae_output_ta.size(), vae_output)

            z_pres_ta = z_pres_ta.write(z_pres_ta.size(), z_pres)
            z_pres_probs_ta = z_pres_probs_ta.write(z_pres_probs_ta.size(), z_pres_prob)
            z_pres_kl_ta = z_pres_kl_ta.write(z_pres_kl_ta.size(), z_pres_kl)

            return (
                step + 1, stopping_sum, next_state,
                running_recon, kl_loss, running_digits,

                scale_ta, scale_kl_ta, scale_std_ta,
                shift_ta, shift_kl_ta, shift_std_ta,
                attr_ta, attr_kl_ta, attr_std_ta,
                z_pres_ta, z_pres_probs_ta, z_pres_kl_ta,
                vae_input_ta, vae_output_ta,

                scale, shift, attr, z_pres,
            )

        # --- end of while-loop body ---

        rnn_init_state = self.cell.zero_state(self.batch_size, tf.float32)

        (_, _, _, reconstruction, kl_loss, self.predicted_n_digits,
         scale, scale_kl, scale_std, shift, shift_kl, shift_std,
         attr, attr_kl, attr_std, z_pres, z_pres_probs, z_pres_kl,
         vae_input, vae_output, _, _, _, _) = tf.while_loop(
            cond, body, [
                tf.constant(0),                                 # RNN time step, initially zero
                tf.zeros((self.batch_size, 1)),                 # running sum of z_pres samples
                rnn_init_state,                                 # initial RNN state
                tf.zeros((self.batch_size, np.product(self.obs_shape))),  # reconstruction canvas, initially empty
                tf.zeros((self.batch_size, 1)),                    # running value of the loss function
                tf.zeros((self.batch_size, 1), dtype=tf.int32),    # running inferred number of digits
                tf.TensorArray(dtype=tf.float32, size=0, dynamic_size=True),
                tf.TensorArray(dtype=tf.float32, size=0, dynamic_size=True),
                tf.TensorArray(dtype=tf.float32, size=0, dynamic_size=True),
                tf.TensorArray(dtype=tf.float32, size=0, dynamic_size=True),
                tf.TensorArray(dtype=tf.float32, size=0, dynamic_size=True),
                tf.TensorArray(dtype=tf.float32, size=0, dynamic_size=True),
                tf.TensorArray(dtype=tf.float32, size=0, dynamic_size=True),
                tf.TensorArray(dtype=tf.float32, size=0, dynamic_size=True),
                tf.TensorArray(dtype=tf.float32, size=0, dynamic_size=True),
                tf.TensorArray(dtype=tf.float32, size=0, dynamic_size=True),
                tf.TensorArray(dtype=tf.float32, size=0, dynamic_size=True),
                tf.TensorArray(dtype=tf.float32, size=0, dynamic_size=True),
                tf.TensorArray(dtype=tf.float32, size=0, dynamic_size=True),
                tf.TensorArray(dtype=tf.float32, size=0, dynamic_size=True),

                tf.zeros((self.batch_size, 2)),  # scale
                tf.zeros((self.batch_size, 2)),  # shift
                tf.zeros((self.batch_size, self.A)),  # attr
                tf.zeros((self.batch_size, 1)),  # z_pres
            ]
        )

        def process_tensor_array(tensor_array, name, shape=None):
            tensor = tf.transpose(tensor_array.stack(), (1, 0, 2))

            time_pad = self.max_time_steps - tf.shape(tensor)[1]
            padding = [[0, 0], [0, time_pad]]
            padding += [[0, 0]] * (len(tensor.shape)-2)

            tensor = tf.pad(tensor, padding, name=name)

            if shape is not None:
                tensor = tf.reshape(tensor, shape)

            return tensor

        self.predicted_n_digits = self.predicted_n_digits[:, 0]
        self._tensors["predicted_n_digits"] = self.predicted_n_digits

        self._tensors['scale'] = process_tensor_array(scale, 'scale')
        self._tensors['scale_kl'] = process_tensor_array(scale_kl, 'scale_kl')
        self._tensors['scale_std'] = process_tensor_array(scale_std, 'scale_std')

        self._tensors['shift'] = process_tensor_array(shift, 'shift')
        self._tensors['shift_kl'] = process_tensor_array(shift_kl, 'shift_kl')
        self._tensors['shift_std'] = process_tensor_array(shift_std, 'shift_std')

        self._tensors['attr'] = process_tensor_array(attr, 'attr', (self.batch_size, self.max_time_steps, self.A))
        self._tensors['attr_kl'] = process_tensor_array(attr_kl, 'attr_kl')
        self._tensors['attr_std'] = process_tensor_array(attr_std, 'attr_std')

        self._tensors['z_pres'] = process_tensor_array(z_pres, 'z_pres', (self.batch_size, self.max_time_steps, 1))
        self._tensors['obj'] = tf.round(self._tensors['z_pres'])  # for `build_math_representation`
        self._tensors['z_pres_probs'] = process_tensor_array(z_pres_probs, 'z_pres_probs')
        self._tensors['z_pres_kl'] = process_tensor_array(z_pres_kl, 'z_pres_kl')

        self._tensors['vae_input'] = process_tensor_array(vae_input, 'vae_input')
        self._tensors['vae_output'] = process_tensor_array(vae_output, 'vae_output')

        reconstruction = tf.clip_by_value(reconstruction, 0.0, 1.0)

        flat_inp = tf.layers.flatten(self._tensors["inp"])

        self._tensors['per_pixel_reconstruction_loss'] = xent_loss(pred=reconstruction, label=flat_inp)
        self.losses.update(
            reconstruction=tf_mean_sum(self._tensors['per_pixel_reconstruction_loss']),
            running=self.kl_weight * tf.reduce_mean(kl_loss),
        )

        self._tensors['output'] = tf.reshape(reconstruction, (self.batch_size,) + self.obs_shape)

        count_error = 1 - tf.to_float(tf.equal(self.target_n_digits, self.predicted_n_digits))
        count_1norm = tf.abs(self.target_n_digits - self.predicted_n_digits)

        self.record_tensors(
            predicted_n_digits=self.predicted_n_digits,
            count_error=count_error,
            count_1norm=count_1norm,

            scale=self._tensors["scale"],
            x=self._tensors["shift"][:, :, 0],
            y=self._tensors["shift"][:, :, 1],
            z_pres_prob=self._tensors["z_pres_probs"],
            z_pres_kl=self._tensors["z_pres_kl"],

            scale_kl=self._tensors["scale_kl"],
            shift_kl=self._tensors["shift_kl"],
            attr_kl=self._tensors["attr_kl"],

            scale_std=self._tensors["scale_std"],
            shift_std=self._tensors["shift_std"],
            attr_std=self._tensors["attr_std"],
        )


class AIR_RenderHook(RenderHook):
    fetches = (
        "inp annotations n_annotations output scale shift "
        "predicted_n_digits vae_input vae_output background")

    def __call__(self, updater):
        fetched = self._fetch(updater)
        self._plot_reconstruction(updater, fetched)

    def _plot_reconstruction(self, updater, fetched):
        network = updater.network

        inp = fetched['inp'].reshape(self.N, *network.obs_shape)
        output = fetched['output'].reshape(self.N, *network.obs_shape)
        object_shape = network.object_shape

        vae_input = fetched['vae_input'].reshape(
            self.N, network.max_time_steps, *object_shape, network.image_depth)
        vae_output = fetched['vae_output'].reshape(
            self.N, network.max_time_steps, *object_shape, network.image_depth)

        # background = fetched['background']

        scale = fetched['scale'].reshape(self.N, network.max_time_steps, 2)
        shift = fetched['shift'].reshape(self.N, network.max_time_steps, 2)
        predicted_n_digits = fetched['predicted_n_digits']

        annotations = fetched["annotations"]
        n_annotations = fetched["n_annotations"]

        color_order = plt.rcParams['axes.prop_cycle'].by_key()['color']

        max_n_digits = max(predicted_n_digits)

        fig_width = 30
        fig, axes = plt.subplots(
            max_n_digits + 1, 2 * self.N,
            figsize=(fig_width, (max_n_digits+1) / (2*self.N) * fig_width))

        for i in range(self.N):
            ax_gt = axes[0, 2*i]
            self.imshow(ax_gt, inp[i])
            ax_gt.set_axis_off()

            ax_rec = axes[0, 2*i+1]
            self.imshow(ax_rec, output[i])
            ax_rec.set_axis_off()

            # Plot true bounding boxes
            for j in range(n_annotations[i]):
                valid, _, t, b, l, r = annotations[i][j]
                if not valid:
                    continue

                h = b - t
                w = r - l

                rect = patches.Rectangle(
                    (l, t), w, h, linewidth=1, edgecolor="white", facecolor='none')
                ax_gt.add_patch(rect)

                rect = patches.Rectangle(
                    (l, t), w, h, linewidth=1, edgecolor="white", facecolor='none')
                ax_rec.add_patch(rect)

            for t in range(max_n_digits):
                axes[t+1, 2*i].set_axis_off()
                axes[t+1, 2*i+1].set_axis_off()

                if t >= predicted_n_digits[i]:
                    axes[t+1, 2*i].set_aspect('equal')
                    axes[t+1, 2*i+1].set_aspect('equal')
                    continue

                w, h = scale[i, t, :]
                x, y = shift[i, t, :]

                transformed_x = 0.5 * (x + 1.)
                transformed_y = 0.5 * (y + 1.)

                height = h * network.image_height
                width = w * network.image_width

                top = network.image_height * transformed_y - height / 2
                left = network.image_width * transformed_x - width / 2

                rect = patches.Rectangle(
                    (left, top), width, height, linewidth=1, edgecolor=color_order[t], facecolor='none')
                ax_rec.add_patch(rect)

                rect = patches.Rectangle(
                    (left, top), width, height, linewidth=1, edgecolor=color_order[t], facecolor='none')
                ax_gt.add_patch(rect)

                ax = axes[t+1, 2*i]
                self.imshow(ax, vae_input[i, t])
                ax.set_ylabel("t={}".format(t))

                obj_rect = patches.Rectangle(
                    (1, 0), 0.2, 1, clip_on=False, transform=ax.transAxes, facecolor=color_order[t])
                ax.add_patch(obj_rect)

                ax = axes[t+1, 2*i+1]
                self.imshow(ax, vae_output[i, t])

        plt.subplots_adjust(left=0.01, right=.99, top=.99, bottom=0.01, wspace=0.14, hspace=0.1)
        self.savefig("sampled_reconstruction", fig, updater)


class AIR_ComparisonRenderHook(AIR_RenderHook):
    def _plot_reconstruction(self, updater, fetched):
        network = updater.network

        inp = fetched['inp'].reshape(self.N, *network.obs_shape)

        scale = fetched['scale'].reshape(self.N, network.max_time_steps, 2)
        shift = fetched['shift'].reshape(self.N, network.max_time_steps, 2)
        predicted_n_digits = fetched['predicted_n_digits']

        color = "xkcd:azure"

        for i in range(self.N):
            fig = plt.figure(figsize=(5, 5))
            ax = plt.gca()

            self.imshow(ax, inp[i])
            ax.set_axis_off()

            for t in range(predicted_n_digits[i]):
                w, h = scale[i, t, :]
                x, y = shift[i, t, :]

                transformed_x = 0.5 * (x + 1.)
                transformed_y = 0.5 * (y + 1.)

                height = h * network.image_height
                width = w * network.image_width

                top = network.image_height * transformed_y - height / 2
                left = network.image_width * transformed_x - width / 2

                rect = patches.Rectangle(
                    (left, top), width, height, linewidth=2,
                    edgecolor=color, facecolor='none')
                ax.add_patch(rect)

            plt.subplots_adjust(left=.01, right=.99, top=.99, bottom=0.01, wspace=0.1, hspace=0.1)
            self.savefig("ground_truth/" + str(i), fig, updater, is_dir=False)
