# WGAN-GP
Example of greyscale WGAN-GP (GAN with Wasserstein Gradient Penalty Loss). To train the GAN run train_WGAN script. 

- params['save_name'] = path to save directory/n
- params['data_path'] = dataset path. IT NEEDS TO BE A NPZ FILE [SIZE_DATASETxHEIGHT*WIDTH*1].
- params['buffer_size'] = dataset size (default 60000)
- params['batch_size'] = bacth size (default 32)
- params['noise_dim'] = noise (default 100)
- params['epochs'] = number of epochs (default 300)
- params['num_exmaple_to_generate'] = number of image generated and saved every 25 epochs to track improvements.
- params['disc_iterations'] = number of iteration of discriminator per iteration of the generator


