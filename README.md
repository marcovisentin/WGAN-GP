# WGAN-GP
Example of greyscale WGAN-GP (GAN with Wasserstein Gradient Penalty Loss). To train the GAN run train_WGAN script. 

Edit params file to personalise:

- params['save_name'] = path to save directory/n
- params['data_path'] = dataset path. IT NEEDS TO BE A NPZ FILE [SIZE_DATASETxHEIGHT*WIDTHx1].
- params['buffer_size'] = dataset size (default 60000)
- params['batch_size'] = bacth size (default 32)
- params['noise_dim'] = noise (default 100)
- params['epochs'] = number of epochs (default 3000)
- params['num_exmaple_to_generate'] = number of image generated and saved every 25 epochs to track improvements.
- params['disc_iterations'] = number of iteration of discriminator per iteration of the generator




## Result after 700 epochs on a dataset of 6000 images of abdominal MRIs: 
![image_at_epoch_0700](https://user-images.githubusercontent.com/88335919/192044187-61e15fc0-138c-4b64-86af-9091593a7685.png)
Not physiologically plausible but surprisingly accurate in the detection of some of the structures (spine, galbladder, aorta, IVC).
