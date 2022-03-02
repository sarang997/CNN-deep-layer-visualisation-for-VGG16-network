#creating a general function that creates the plot 
def visualize_layers(layer_number, test_image_path,model_h5_path,cmap =None):
       my_model = keras.models.load_model(model_h5_path)

       model = my_model.layers[0]
      

       filters,biases = model.layers[layer_number].get_weights()
       #normalizing filter weight values
       f_min, f_max = filters.min(), filters.max()
       filters = (filters - f_min) / (f_max - f_min)

       model_layer1 = Model(inputs = model.inputs, outputs = model.layers[layer_number].output)

       image = load_img(test_image_path,target_size = (224,224))
       image = img_to_array(image)
       image = expand_dims(image,axis=0)
       image =preprocess_input(image)
       feature_map1 = model_layer1.predict(image)
       fig = pyplot.figure(figsize=(30,int(feature_map1.shape[3])*5))
       for i in range(1,feature_map1.shape[3]+1):

           pyplot.subplot(int(feature_map1.shape[3]/2),2,i)
           pyplot.imshow(feature_map1[0,:,:,i-1] ,cmap= cmap)

       pyplot.show()
