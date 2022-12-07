from tensorflow.keras import layers, Model, models
import numpy as np
import cv2
import tensorflow as tf
import matplotlib.pyplot as plt
import streamlit as st
from PIL import Image
import os
import pickle
from streamlit_image_comparison import image_comparison



API_KEY = 'AIzaSyAB-hvYdWIzPRakKErGrL5eaLp6-CkqXEM'
import googlemaps
from datetime import datetime

gmaps = googlemaps.Client(key=API_KEY)


def loss(y_true, y_pred):
    def dice_loss(y_true, y_pred):
        y_pred = tf.math.sigmoid(y_pred)
        numerator = 2 * tf.reduce_sum(y_true * y_pred)
        denominator = tf.reduce_sum(y_true + y_pred)

        return 1 - numerator / denominator

    y_true = tf.cast(y_true, tf.float32)
    o = tf.nn.sigmoid_cross_entropy_with_logits(y_true, y_pred) + dice_loss(y_true, y_pred)
    return tf.reduce_mean(o)




@st.cache(allow_output_mutation=True) #cache the model at first loading
def load_model_cache():

    path_folder = os.path.dirname(__file__)#Get Current directory Path File
    VGG16_model_load = models.load_model(os.path.join(path_folder,'VGG16_model_full_Enhanced.h5'), custom_objects={'loss': loss})

    return VGG16_model_load

VGG16_model_load = load_model_cache()


def solar_roof(arr_im):
    # checking if it is a file

    # im = cv2.imread(file)
    #try:
    #im_pad = tf.image.pad_to_bounding_box(im, 0, 0, 5120, 5120)
    
    shape1 = (arr_im.shape[0]//256+1)*256
    shape2 = (arr_im.shape[1]//256+1)*256

    
    
    im_pad = np.zeros((shape1,shape2,3))
    im_pad[:arr_im.shape[0],:arr_im.shape[1],:] = arr_im/255

    M = 256
    N = 256
    i=0
    test_image = []
    test_image_merge = np.zeros((shape1,shape2,3))


    for x in range(0,im_pad.shape[0],M):
        for y in range(0,im_pad.shape[1],N):
            tile =im_pad[x:x+M,y:y+N]
            test_image_pred = VGG16_model_load.predict(np.expand_dims(tile, axis = 0))
            test_image_merge[x:x+M,y:y+N]=(test_image_pred)[0]


    cropped = np.uint8(test_image_merge[0:arr_im.shape[0],0:arr_im.shape[1],:]*255)

    return cropped



def save_object(obj, filename):
    with open(filename, 'wb') as outp:  # Overwrites any existing file.
        pickle.dump(obj, outp, pickle.HIGHEST_PROTOCOL)

# sample usage


def main():
    
    lcol1,lcol2,lcol3 = st.columns(3)  
    slb = Image.open('slb.png')

    # slb = cv2.imread('slb.png')
    # slb = slb.convert("RGB")
    wagon = Image.open('wagon.png')


    lcol1.image(slb, width=150)
    lcol3.image(wagon, width=200)
    
    
    
    st.markdown('<div style="text-align: center"><h1 style="color:rgb(0,0,255)"> JASS Solar Roof Predection </h1></div>', unsafe_allow_html=True)
    

    st.sidebar.title('Navigation')
    selection = st.sidebar.radio("Go to", ["Home","Prediction", "Load Images"])
    

    
    
    
  

    if selection == "Home":
        
        hcol1,hcol2 = st.columns(2)   
        st.markdown('<div style="text-align: center"><h3 style="color:rgb(0,0,255)"> Welcome to Satellite Solar Roof Prediction </h3></div>', unsafe_allow_html=True)
        
        
        true = cv2.imread('true.png')
        mask = cv2.imread('mask.png')
        
        hcol1.image(true)
        hcol2.image(mask)

    elif selection == "Prediction":
        
        lat = st.text_input('Please enter latitude cordinate')
                            
        long = st.text_input('Please enter longitude cordinate')
        
        
        #
        
        col1_button,col2_button = st.columns(2) 
        show = col1_button.button('show the image')                
        if lat != "" and long != "":
            
            
            
            col1,col2 = st.columns(2)     
            response = gmaps.static_map(size = (5000, 5000), zoom=16, center= (float(lat), float(long)), maptype='satellite', scale=2)

            img_file = open('test_gmap_img.jpeg', 'wb')

            for x in response:
                img_file.write(x)
            img_file.close()
            im = cv2.imread('test_gmap_img.jpeg')
            # im = Image.open('test_gmap_img.png')
            # arr_im = np.array(im)
            
            arr_im = cv2.cvtColor(im, cv2.COLOR_BGR2RGB)
            
            Mask_Prediction= solar_roof(arr_im)
            #save_object(Mask_Prediction, 'Mask_Prediction.pkl')

            pil_decoded = Image.fromarray(Mask_Prediction[:,:,0])
            
            
            if show:
                
                st.spinner(text="In progress...")
                st.image(arr_im)

#             pred = st.button('Predict my Mask image')
            
#             if pred:         
#                 st.image(arr_im)
#                 st.image(pil_decoded)
#                 st.balloons()
                
                
            menu = st.selectbox("Display options", ["","side by side","overlay", "Slider"])

            if menu == "side by side":
                
                st.spinner(text="In progress...")
                
                col1.image(arr_im)
                col2.image(pil_decoded)
                st.balloons()                

            if menu == "overlay":
                IMS = Image.fromarray(arr_im[:,:,0])
                IMS = IMS.convert("RGBA")
                IML = pil_decoded.convert("RGBA")
                new_img = Image.blend(IMS, IML, 0.4)
                st.image(new_img)
                st.balloons()                

            if menu == "Slider":
                
                st.spinner(5)

                       
            # render image-comparison
                image_comparison(
                    img1=arr_im,
                    img2=pil_decoded,
                )
                st.balloons()                
                
            
            # mask_copy = Mask_Prediction.copy()
            # mask_copy[mask_copy>0.8*255]=1
            # mask_copy[mask_copy<=0.8*255]=0
            st.text("Total Area= "+str(round(((int(arr_im.shape[0])*int(arr_im.shape[1]))*2.39*2.39*1e-6),2))+"km2")
            print(np.max(Mask_Prediction), np.min(Mask_Prediction), )
            st.text("Solar Roof Area= "+str((round(((np.sum(Mask_Prediction))/255)*2.39*2.39*1e-6*.8,2)))+"km2")

      
    elif selection == 'Load Images':
        
        file_name = st.file_uploader('Please load your aerial image here', type=["tif", "png"])

        if file_name is not None:
            im = Image.open(file_name)
            # arr_im = im
            im = im.convert("RGB")

            arr_im = np.array(im)
            st.image(im)
            Mask_Prediction= solar_roof(arr_im)
            save_object(Mask_Prediction, 'Mask_Prediction.pkl')

            pil_decoded = Image.fromarray(Mask_Prediction[:,:,0])

            if st.button('Predict my Mask image (side by side)'):
                scol1,scol2 = st.columns(2)
                scol1.image(im)
                scol2.image(pil_decoded)
                st.balloons()
                
                
            if st.button('Predict my Mask image (Slider)'):
                
                
                st.spinner(5)
                image_comparison(
                    img1=arr_im,
                    img2=pil_decoded, width=256
                )
                st.balloons()                
                


if __name__ == '__main__':
    main()

    
    