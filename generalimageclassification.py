import streamlit as st
import numpy as np
import cv2
from PIL import Image 
from skimage.transform import resize
import tensorflow as tf
import visualkeras
from contextlib import contextmanager
from streamlit.report_thread import REPORT_CONTEXT_ATTR_NAME
from threading import current_thread
from io import StringIO
import sys


@contextmanager
def st_redirect(src, dst):
    placeholder = st.empty()
    output_func = getattr(placeholder, dst)

    with StringIO() as buffer:
        old_write = src.write

        def new_write(b):
            if getattr(current_thread(), REPORT_CONTEXT_ATTR_NAME, None):
                buffer.write(b)
                output_func(buffer.getvalue())
            else:
                old_write(b)

        try:
            src.write = new_write
            yield
        finally:
            src.write = old_write



@contextmanager
def st_stdout(dst):
    with st_redirect(sys.stdout, dst):
        yield

@st.cache(allow_output_mutation=True,suppress_st_warning=True)
def valid_states():
    return {"valid": None}

@st.cache(allow_output_mutation=True,suppress_st_warning=True)
def created_states():
    return {"created": None}

@st.cache(allow_output_mutation=True,suppress_st_warning=True)
def compiled_states():
    return {"compiled": None}    

@st.cache(allow_output_mutation=True,suppress_st_warning=True)
def trained_states():
    return {"trained": None}    

@st.cache(allow_output_mutation=True,suppress_st_warning=True)
def create_dataset(a,b,c,d,k):
    nalpha=len(a)
    nbeta=len(b)
    ncharlie=0
    ndelta=0
    if k==4:
        ncharlie=len(c)
        ndelta=len(d)
        images=np.array(list(a.values())+list(b.values())+list(c.values())+list(d.values()))
        labels=np.array([0]*nalpha+[1]*nbeta+[2]*ncharlie+[3]*ndelta)
    elif k==3:
        ncharlie=len(c)
        st.write(np.shape(list(a.values())))
        st.write(np.shape(list(b.values())))
        st.write(np.shape(list(c.values())))
        images=np.array(list(a.values())+list(b.values())+list(c.values()))
        labels=np.array([0]*nalpha+[1]*nbeta+[2]*ncharlie)
        
    else:
        images=np.array(list(a.values())+list(b.values()))
        labels=np.array([0]*nalpha+[1]*nbeta)
    
    ds=None
    s=""
    if (k==4 and (nalpha==0 or nbeta==0 or ncharlie==0 or ndelta==0)) or (k==3 and (nalpha==0 or nbeta==0 or ncharlie==0)) or (k==2 and (nalpha==0 or nbeta==0)):
        valid.update({"valid":False})
        
    if valid["valid"]!=False:
        ds = tf.data.Dataset.from_tensor_slices((images, labels)).batch(32)
        if k==4:
            s=str(nalpha)+" image(s) uploaded for class 1. "+str(nbeta)+" image(s) uploaded for class 2. "+str(ncharlie)+" image(s) uploaded for class 3. "+str(ndelta)+" image(s) uploaded for class 4. "
        elif k==3:
            s=str(nalpha)+" image(s) uploaded for class 1. "+str(nbeta)+" image(s) uploaded for class 2. "+str(ncharlie)+" image(s) uploaded for class 3. "
        else:
            s=str(nalpha)+" image(s) uploaded for class 1. "+str(nbeta)+" image(s) uploaded for class 2. "
        valid.update({"valid":True})
    return ds,s

@st.cache(allow_output_mutation=True,suppress_st_warning=True)
def model_creation(n,na,nf):
    if n==3:
            model = tf.keras.models.Sequential([
                tf.keras.layers.Conv2D(16, (kernel, kernel), activation='relu', input_shape=(150, 150, 3)),
                tf.keras.layers.MaxPooling2D(2, 2),
                tf.keras.layers.Conv2D(32, (kernel, kernel), activation='relu'),
                tf.keras.layers.MaxPooling2D(2, 2),
                tf.keras.layers.Conv2D(64, (kernel, kernel), activation='relu'),
                tf.keras.layers.MaxPooling2D(2, 2),
                tf.keras.layers.Flatten(),
                tf.keras.layers.Dense(nodes, activation='relu'),
                tf.keras.layers.Dense(nf, activation=na)
            ])
    elif n==2:
            model = tf.keras.models.Sequential([
                tf.keras.layers.Conv2D(16, (kernel, kernel), activation='relu', input_shape=(150, 150, 3)),
                tf.keras.layers.MaxPooling2D(2, 2),
                tf.keras.layers.Conv2D(32, (kernel, kernel), activation='relu'),
                tf.keras.layers.MaxPooling2D(2, 2),
                tf.keras.layers.Flatten(),
                tf.keras.layers.Dense(nodes, activation='relu'),
                tf.keras.layers.Dense(nf, activation=na)
            ])
    else:
            model = tf.keras.models.Sequential([
                tf.keras.layers.Conv2D(16, (kernel, kernel), activation='relu', input_shape=(150, 150, 3)),
                tf.keras.layers.MaxPooling2D(2, 2),
                tf.keras.layers.Flatten(),
                tf.keras.layers.Dense(nodes, activation='relu'),
                tf.keras.layers.Dense(nf, activation=na)
            ])
    return model

@st.cache(allow_output_mutation=True,suppress_st_warning=True)
def model_compilation(a,l,n):
    if n>2:
        f="sparse_categorical_crossentropy"
    else:
        f="binary_crossentropy"
    if algorithm=="rmsprop":
            model.compile(optimizer=tf.keras.optimizers.RMSprop(lr=l), loss=f, metrics=['acc'])
    elif algorithm=="sgd":
            model.compile(optimizer=tf.keras.optimizers.SGD(lr=l), loss=f, metrics=['acc'])
    elif algorithm=="adam":
            model.compile(optimizer=tf.keras.optimizers.Adam(lr=l), loss=f, metrics=['acc'])
    else:
            model.compile(optimizer=tf.keras.optimizers.Adagrad(lr=l), loss=f, metrics=['acc'])
    return



def load_image(image_file):
	img = Image.open(image_file)
	return img 

def threechannels(img):
    if len(img.shape) > 2 and img.shape[2] == 4:
        #convert the image from RGBA2RGB
        img = cv2.cvtColor(img, cv2.COLOR_BGRA2BGR)
    if len(img.shape)==2:
        img = cv2.cvtColor(img,cv2.COLOR_GRAY2RGB)
    return img

def summary(model: tf.keras.Model) -> str:
        summary = []
        model.summary(print_fn=lambda x: summary.append(x))
        return '\n'.join(summary)
st.title("General-Purpose Image Classifier")
st.sidebar.header("Model Creation")
st.sidebar.write("**Number of Classes**")
nclass=st.sidebar.slider("Number of categories to classify",min_value=2,max_value=4)
st.sidebar.write("**Iterations of Convolution & Pooling Layers**")
nlayers=st.sidebar.slider("Number of Convolution & Pooling Layer Combination",min_value=1,max_value=3)
st.sidebar.write("**Kernel Size**")
kernel=st.sidebar.slider("Kernel size (squared) of the Convolution Layer in pixels",min_value=2,max_value=5)
st.sidebar.write(str(kernel)+"x"+str(kernel))
st.sidebar.write("**Number of Dense Layer Nodes after Flatten Layer**")
nodes=st.sidebar.slider("Number of Dense Layer Nodes after Flatten Layer",min_value=128,max_value=1024)
st.sidebar.write("**Output Layer Activation**")
if nclass>2:
    st.sidebar.write("Softmax")
else:
    st.sidebar.write("Sigmoid")

st.sidebar.header("Model Compilation")
st.sidebar.write("**Choose an Optimizer**")
algorithm=st.sidebar.selectbox("Algorithm", ["sgd","rmsprop","adam","adagrad"])
st.sidebar.write("**Learning rate for Optimizer**")
learningrate=st.sidebar.slider("Learning rate",min_value=0.003,max_value=0.02,value=0.01,step=0.001)
st.sidebar.write("**Loss Function**")
if nclass>2:
    st.sidebar.write("Sparse Categorical Cross Entropy")
else:
    st.sidebar.write("Binary Cross Entropy")
st.sidebar.header("Model Training")
st.sidebar.write("**Number of Epochs**")
epoch=st.sidebar.slider("Epochs",min_value=20,max_value=200,value=50,step=10)
st.sidebar.header("Actions")







st.subheader("Image Upload")
status_data=st.empty()
status_data.info("Awaiting images to be uploaded for each class")
alpha_raw=st.file_uploader('Upload your images for Class 1',type=["png", "jpg"],accept_multiple_files=True)
beta_raw=st.file_uploader('Upload your images for Class 2',type=["png", "jpg"],accept_multiple_files=True)
if nclass>=3:
    charlie_raw=st.file_uploader('Upload your images for Class 3',type=["png", "jpg"],accept_multiple_files=True)
if nclass==4:
    delta_raw=st.file_uploader('Upload your images for Class 4',type=["png", "jpg"],accept_multiple_files=True)

alpha={}
beta={}
charlie={}
delta={}
if len(alpha_raw)>0:
    for i, image in enumerate(alpha_raw):
	    alpha[i] = resize(threechannels(np.array(load_image(alpha_raw[i]))), (150, 150),anti_aliasing=True)
if len(beta_raw)>0:
    for i, image in enumerate(beta_raw):
	    beta[i] = resize(threechannels(np.array(load_image(beta_raw[i]))), (150, 150),anti_aliasing=True)
if nclass>=3:
    if len(charlie_raw)>0:
        for i, image in enumerate(charlie_raw):
            charlie[i] = resize(threechannels(np.array(load_image(charlie_raw[i]))), (150, 150),anti_aliasing=True)
if nclass==4:
    if len(delta_raw)>0:
        for i, image in enumerate(delta_raw):
            delta[i] = resize(threechannels(np.array(load_image(delta_raw[i]))), (150, 150),anti_aliasing=True)

flow=st.sidebar.button('Flow images into dataset')
valid=valid_states()
message=""
dataset=None
if flow:
    status_data.info("Flowing...") 
    valid.update({"valid":None})
    dataset,message=create_dataset(alpha,beta,charlie,delta,nclass)

if valid["valid"]==True:
    dataset,message=create_dataset(alpha,beta,charlie,delta,nclass)
    status_data.success("Dataset constructed. Images have been flowed for each class. "+message+"You may now create and fit the model to the data.")

elif valid["valid"]==False:
    status_data.error("No images uploaded for at least one class")




nfinal=4
nact="softmax"
if nclass==3:
    nfinal=3
if nclass==2:
    nfinal=1
    nact="sigmoid"

create=st.sidebar.button("Create model")
st.subheader("Creating a Model")
status_model=st.empty()
status_model.info("Model not yet created")
created=created_states()
if create:
    status_model.info("Creating model...")
    model=model_creation(nlayers,nact,nfinal)
    created.update({"created":True})

    
if created["created"]==True:
    model=model_creation(nlayers,nact,nfinal)
    status_model.success("Model created")
    created.update({"created":True})
    visualkeras.layered_view(model, to_file='model.png')
    st.image('./model.png')
    st.code(summary(model))
    

st.subheader("Compiling the Model")
status_compile=st.empty()
status_compile.info("Model not yet compiled")
compile=st.sidebar.button("Compile model")
compiled=compiled_states()
if compile:
    if created["created"]!=True:
        status_compile.error("Model not yet created and could not be compiled")
    else:
        status_compile.info("Compiling...")
        model_compilation(algorithm,learningrate,nclass)
        status_compile.success("Model compiled with algorithm "+str(algorithm)+" and learning rate at "+str(learningrate))
        compiled.update({"compiled":True})

if compiled["compiled"]==True:
    status_compile.success("Model compiled with algorithm "+str(algorithm)+" and learning rate at "+str(learningrate))
    model_compilation(algorithm,learningrate,nclass)


train=st.sidebar.button("Train model")
st.subheader("Training the Model")
status_train=st.empty()
status_train.info("Model not yet trained")
trained=trained_states()
if train:
    if valid["valid"]==True and created["created"]==True and compiled["compiled"]==True:
        status_train.info("Training...")
        with st_stdout("code"):
            model.fit(dataset, epochs=epoch,verbose=1)
        status_train.success("Training complete")
        trained.update({"trained":True})

    else:
        status_train.error("Failed to train model - Please confirm that all previous steps have been completed")


st.subheader("Predicting with the Built Model") 
prediction_raw=st.file_uploader('Upload an image for the model to make a prediction',type=["png", "jpg"])
predict=st.button("Predict image class using the trained model")
status_predict=st.empty()
if predict:
    if valid["valid"]==True and created["created"]==True and compiled["compiled"]==True and trained["trained"]==True:
        if prediction_raw is not None:
            prediction_image = Image.open(prediction_raw)
            prediction_array = resize(threechannels(np.array(prediction_image)), (150, 150),anti_aliasing=True)
            prediction_array = np.expand_dims(prediction_array,axis=0)
            prediction_class=int(model.predict_classes(prediction_array))
            status_predict.info("The model predicted the image to be of class "+str(prediction_class))
        else:
            status_predict.error("No image is uploaded")
    else:
        status_predict.error("Model could not be used - Please confirm that the model has been built and trained")
        
