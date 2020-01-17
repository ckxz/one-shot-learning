import os
import sys
import numpy as np
import tensorflow as tf
from tensorflow import keras as ks
import matplotlib.pyplot as plt


import data_preproc as dp
import data_preproc_resize_244_244 as res_dp
import data_preproc_different_modes as dp_bw

def create_branch_facenet(image_shape):
    face_net = ks.models.load_model('facenet_keras.h5')
    for layer in face_net.layers[:400]:
        layer.trainable = False
    for layer in face_net.layers[400:]:
        layer.trainable = True
    return face_net


def create_branch_of_network_resnet50(image_shape):
    base_model = ks.applications.resnet50.ResNet50(input_shape=(224, 224, 3))
    for layer in base_model.layers[:70]:
        layer.trainable = False
    for layer in base_model.layers[70:]:
        layer.trainable = True
    return base_model

def create_branch_mobile_net(image_shape):
    base_model = ks.applications.mobilenet.MobileNet(weights='image_netc')
    for layer in base_model.layers:
        layer.trainable = False

    return base_model


def create_siamese_network_resnet_50(image_shape):
    # input images tensors
    image_one = ks.Input(image_shape)
    image_two = ks.Input(image_shape)

    # CNN
    base_model = create_branch_of_network_resnet50(image_shape)

    # Siamese networks predictions
    y_hat_one = base_model(image_one)
    y_hat_two = base_model(image_two)

    # Adding layer to compute siaemese networks distance
    distance_computation = ks.layers.Lambda(lambda tensors: ks.backend.abs(tensors[0] - tensors[1]))
    images_distance = distance_computation([y_hat_one, y_hat_two])
    # Dense layer
    y_hat = ks.layers.Dense(1, activation='sigmoid')(images_distance)
    siamese_net = ks.Model(inputs=[image_one, image_two], outputs=y_hat)

    return siamese_net


def create_siamese_network_facenet(image_shape):
    # input images tensors
    image_one = ks.Input(image_shape)
    image_two = ks.Input(image_shape)

    # CNN
    base_model = create_branch_facenet(image_shape)

    # Siamese networks predictions
    y_hat_one = base_model(image_one)
    y_hat_two = base_model(image_two)

    # Adding layer to compute siaemese networks distance
    distance_computation = ks.layers.Lambda(lambda tensors: ks.backend.abs(tensors[0] - tensors[1]))
    images_distance = distance_computation([y_hat_one, y_hat_two])
    # Dense layer
    y_hat = ks.layers.Dense(1, activation='sigmoid')(images_distance)
    siamese_net = ks.Model(inputs=[image_one, image_two], outputs=y_hat)

    return siamese_net


def create_branch_of_network(image_shape):
    model = ks.Sequential()
    model.add(ks.layers.Conv2D(64, kernel_size=80, activation='relu', input_shape=image_shape))
    model.add(ks.layers.MaxPool2D(pool_size=2, strides=2))
    model.add(ks.layers.Conv2D(128, kernel_size=3, activation='relu'))
    model.add(ks.layers.MaxPool2D(pool_size=2, strides=2))
    model.add(ks.layers.Conv2D(128, kernel_size=3, activation='relu'))
    model.add(ks.layers.MaxPool2D(pool_size=2, strides=2))
    model.add(ks.layers.Conv2D(256, kernel_size=3, activation='relu'))
    model.add(ks.layers.Flatten())
    model.add(ks.layers.Dense(4096, activation='sigmoid'))
    return model


def create_siamese_network(image_shape):
    # input images tensors
    image_one = ks.Input(image_shape)
    image_two = ks.Input(image_shape)
    # print(image_shape)
    # CNN
    model = create_branch_of_network(image_shape)

    # Siamese networks predictions
    y_hat_one = model(image_one)
    y_hat_two = model(image_two)

    # Adding layer to compute siaemese networks distance
    distance_computation = ks.layers.Lambda(lambda tensors: ks.backend.abs(tensors[0] - tensors[1]))
    images_distance = distance_computation([y_hat_one, y_hat_two])
    # Dense layer
    y_hat = ks.layers.Dense(1, activation='sigmoid')(images_distance)
    # inputs & outputs
    siamese_net = ks.Model(inputs=[image_one, image_two], outputs=y_hat)

    return siamese_net


def contrastive_loss(y_true, y_pred):
    margin = 1
    return ks.backend.mean(
        y_true * ks.backend.square(y_pred) +
        (1 - y_true) * ks.backend.square(ks.backend.maximum(margin - y_pred, 0))
    )


def main():
    print(f'Running:{sys.argv[1]}')
    run_mode = sys.argv[1]
    loss_function = sys.argv[2]
    model_to_run = sys.argv[3]
    ds_type = sys.argv[4]
    loss_function_siamese = ""
    if loss_function == 'binary_crossentropy':
        loss_function_siamese = loss_function
    elif loss_function == 'contrastive_loss':
        loss_function_siamese = contrastive_loss
    siamese_in_creator = dp.SiameseDatasetCreator()
    siamese_in_creator_244_244_res = res_dp.SiameseDatasetCreatorResize()

    nr_channels_res, height_res, width_res = siamese_in_creator_244_244_res.celeb_loader.dataset[0][0].shape
    if model_to_run == 'basic':
        if ds_type == 'bw':
            siamese_in_creator = dp_bw.SiameseDatasetCreator()
        nr_channels, height, width = siamese_in_creator.celeb_loader.dataset[0][0].shape
        model = create_siamese_network((height, width, nr_channels))
        
        train_siamese_data = siamese_in_creator.generate_verification_input(type_ds='train')
        test_siamese_data = siamese_in_creator.generate_verification_input(type_ds='test')

    elif model_to_run == 'res':
        model = create_siamese_network_resnet_50((height_res, width_res, nr_channels_res))
        train_siamese_data = siamese_in_creator_244_244_res.generate_verification_input(type_ds='train')
        test_siamese_data = siamese_in_creator_244_244_res.generate_verification_input(type_ds='test')
    elif model_to_run == 'facenet':
        model = create_siamese_network_facenet((height_res, width_res, nr_channels_res))
        train_siamese_data = siamese_in_creator_244_244_res.generate_verification_input(type_ds='train')
        test_siamese_data = siamese_in_creator_244_244_res.generate_verification_input(type_ds='test')

    optimizer = ks.optimizers.Adam(lr=0.00006)
    optimizer_rms = ks.optimizers.RMSprop(lr=0.001)
    model.compile(loss=loss_function_siamese,
                  optimizer=optimizer,
                  metrics=['acc'])
    
    filepath="weights-improvement-facenet.ckpt"
    # Create a callback that saves the model's weights
    cp_callback = tf.keras.callbacks.ModelCheckpoint(filepath=filepath,
                                                     save_weights_only=True,
                                                     verbose=1)
    callbacks_list = [cp_callback, tf.keras.callbacks.TensorBoard(log_dir='logs', histogram_freq=0, write_graph=True, write_images=True)]

    if run_mode == 'train':
        history = model.fit_generator(generator=train_siamese_data,
                                      steps_per_epoch=len(siamese_in_creator.celeb_loader.train_dataset),
                                      use_multiprocessing=False,
                                      verbose=1,
                                      callbacks=callbacks_list,
                                      workers=0)
        plt.plot(history.history['acc'])
        plt.title('Model accuracy')
        plt.ylabel('Accuracy')
        plt.xlabel('Epoch')
        plt.legend(['Train', 'Test'], loc='upper left')
        plt.savefig('siamese_un_normalized_imgs.png', bbox_inches='tight')

    else:
        model.load_weights("weights-improvement-img-norm_reshape.ckpt")

    scores = model.evaluate_generator(generator=test_siamese_data,
                                      steps=len(siamese_in_creator.celeb_loader.test_dataset)
                                      )
    print(scores)
    print("%s: %.2f%%" % (model.metrics_names[1], scores[1] * 100))

    return


if __name__ == '__main__':
    main()
