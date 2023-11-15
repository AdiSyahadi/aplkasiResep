import streamlit as st
import tensorflow as tf
from PIL import Image
import numpy as np
from matplotlib import pyplot as plt


def load_prep_image(img,  img_shape =224):
    #img = tf.io.read_file(filename)
    img = tf.io.encode_jpeg(img)
    img = tf.image.decode_image(img, channels = 3)
    img = tf.image.resize(img, size = [img_shape,img_shape])
    img = img/255.

    return img

def pred_and_plot(model, img, class_names):
    st.set_option('deprecation.showPyplotGlobalUse', False)

    img_tf = tf.expand_dims(img, axis = 0)
    #pred = model.predict(img_tf)
    #pred_class = class_names[int(tf.round(pred)[0][0])]
    pred_prob = model.predict(tf.expand_dims(img, axis=0)) # model accepts tensors of shape [None, 224, 224, 3]
    pred_class = class_names[pred_prob.argmax()] # find the predicted class 

    st.write(pred_prob)
    st.write(pred_class)
    plt.imshow(img)
    plt.title( " " + str(pred_class))
    st.pyplot()


def load_model_h5(filename):
    model = tf.keras.models.load_model(filename)
    return model

def class_names_list(filename):
    file = open(filename, 'r').read()
    class_names = file.split("\n")
    return class_names

model =  load_model_h5("./models/model.h5")
class_names = class_names_list("./models/class_names.txt")

def main():
    st.header("Hai Selamat Datang di Program Kami")
    st.write("Aplikasi ini dapat digunakan untuk mengklasifikasikan gambar makanan. program ini dilatih menggunakan Dataset 101 makanan.")

    image = Image.open('img1.png')
    st.image(image)
    
    selector = st.selectbox(
        label = "Picture or Camera input?",
        options = ["camera", "picture","aboutus"],
        )

    if selector == "picture":
        data = st.file_uploader(label="Upload image")
        st.text('Jenis file wajib JPG')
        df = pd.DataFrame(np.random.randn(50, 20), columns=("col %d" % i for i in range(20)))
        st.dataframe(df)  

    elif selector == "camera":
        data = st.camera_input("Take a picture")
        
    elif selector == "aboutus":
        image = Image.open('ourteam.png')
        st.image(image)
    try:
        st.image(data)
        
    except:
        st.warning("No image uploaded!")
        st.stop()

    if st.button(label = "Classify"):
        data = Image.open(data)
        #data.save("data.jpg")
        #img = load_prep_image("data.jpg")
        img = load_prep_image(img = data)
        pred_and_plot(model = model, img = img, class_names = class_names)
    # "with" notation
    




if __name__ == "__main__":
    main()
