# -*- coding: utf-8 -*-
"""
Created on Fri Jan 17 11:23:24 2025

@author: andres cremades botella

File for creating a SHAP tutorial - Group meeting Mon Jan 20.

Base example taken from https://shap.readthedocs.io/en/latest/example_notebooks/image_examples/image_classification/Multi-input%20Gradient%20Explainer%20MNIST%20Example.html
"""
#%%
# ------------------------------------------------------------------------------------------------------------------------------------------------------------------------------
# Import the packages
# ------------------------------------------------------------------------------------------------------------------------------------------------------------------------------
import tensorflow as tf
from tensorflow.keras import Input
from tensorflow.keras.layers import Conv2D, Dense, Dropout, Flatten
import numpy as np
import matplotlib.pyplot as plt
import matplotlib

# ------------------------------------------------------------------------------------------------------------------------------------------------------------------------------
# Define the size of the images
#     - size_x : size of the picture in x
#     - size_y : size of the picture in y
#     - outcha : number of output channels
#     - fs     : font size of the plots
# ------------------------------------------------------------------------------------------------------------------------------------------------------------------------------
size_x = 28
size_y = 28
outcha = 10
fs     = 14
matplotlib.rc('font',size=fs)

# ------------------------------------------------------------------------------------------------------------------------------------------------------------------------------
# load the MNIST data
#     - x_train : training data for the input
#     - y_train : training data for the output
#     - x_test  : validation data for the input
#     - y_test  : validation data for the output
# ------------------------------------------------------------------------------------------------------------------------------------------------------------------------------
(x_train, y_train), (x_test, y_test) = tf.keras.datasets.mnist.load_data()
x_train, x_test                      = x_train / 255.0, x_test / 255.0
x_train                              = x_train.astype("float32")
x_test                               = x_test.astype("float32")
x_train                              = x_train.reshape(x_train.shape[0], size_x, size_y, 1)
x_test                               = x_test.reshape(x_test.shape[0], size_x, size_y, 1)

# ------------------------------------------------------------------------------------------------------------------------------------------------------------------------------
# Define our model
# The model takes 2 inputs: input1 and input2 which are the input image. Then a convolutional network is used
# ------------------------------------------------------------------------------------------------------------------------------------------------------------------------------
input1  = Input(shape=(size_x, size_y, 1))
input2  = Input(shape=(size_x, size_y, 1))
input2c = Conv2D(32, kernel_size=(3, 3), activation="relu")(input2)
joint   = tf.keras.layers.concatenate([Flatten()(input1), Flatten()(input2c)])
out     = Dense(outcha, activation="softmax")(Dropout(0.2)(Dense(128, activation="relu")(joint)))
model   = tf.keras.models.Model(inputs=[input1, input2], outputs=out)

model.compile(optimizer="adam", loss="sparse_categorical_crossentropy", metrics=["accuracy"])

# ------------------------------------------------------------------------------------------------------------------------------------------------------------------------------
# Fit the model
# ------------------------------------------------------------------------------------------------------------------------------------------------------------------------------
model.fit([x_train, x_train], y_train, epochs=3)

#%%
# ------------------------------------------------------------------------------------------------------------------------------------------------------------------------------
# Calculate the SHAP values: Import the shap package
# ------------------------------------------------------------------------------------------------------------------------------------------------------------------------------
import shap

#%%
# ------------------------------------------------------------------------------------------------------------------------------------------------------------------------------
# Calculate the SHAP values using the KernelExplainer
# ------------------------------------------------------------------------------------------------------------------------------------------------------------------------------

# ------------------------------------------------------------------------------------------------------------------------------------------------------------------------------
# Segmentate the domain
# ------------------------------------------------------------------------------------------------------------------------------------------------------------------------------
box_size = 7
index    = 0
mask     = np.zeros((size_x, size_y), dtype=int)
for ind_i in range(0, size_x, box_size):
    for ind_j in range(0, size_y, box_size):
        mask[ind_i:ind_i+box_size, ind_j:ind_j+box_size] = index
        index += 1

nmask = np.max(mask)

# -----------------------------------------------------------------------------------------------------------------------------------------------------------------------------
# Create a reference for the SHAP values
# -----------------------------------------------------------------------------------------------------------------------------------------------------------------------------
reference = np.zeros((size_x,size_y,1))

# -----------------------------------------------------------------------------------------------------------------------------------------------------------------------------
# Define the input to calculate the SHAP values 
# -----------------------------------------------------------------------------------------------------------------------------------------------------------------------------
Xin = x_test[0].reshape(1,size_x,size_y,1)

# -----------------------------------------------------------------------------------------------------------------------------------------------------------------------------
# Function of the model
# -----------------------------------------------------------------------------------------------------------------------------------------------------------------------------
def f(zs):
    ii  = 0
    lm  = zs.shape[0]
    out = np.zeros((lm,outcha))
    print("Starting kernel SHAP:",flush=True)
    
    for ii in np.arange(lm):
        if ii<lm-1:
            print("Calculation "+str(ii)+" of "+str(lm),end='\r',flush=True)
        else:
            print("Calculation "+str(ii)+" of "+str(lm),flush=True)
            
        zii         = zs[ii]
        model_input = mask_dom(zii)
        out[ii,:]   = model.predict([model_input,model_input])
    return out

# -----------------------------------------------------------------------------------------------------------------------------------------------------------------------------
# Function to get the index of the blocks
# -----------------------------------------------------------------------------------------------------------------------------------------------------------------------------
def get_structure_indices():
    struc_indx = []
    for ii in range(nmask+1):
        indx = np.array(np.where(mask == ii)).transpose()
        struc_indx.append(indx.astype(int))
    array_struc_indx = np.array(struc_indx, dtype=object)
    return array_struc_indx

# -----------------------------------------------------------------------------------------------------------------------------------------------------------------------------
# Mask the domain
# -----------------------------------------------------------------------------------------------------------------------------------------------------------------------------
def mask_dom(zs):
    mask_out = Xin.copy()
    if 0 not in zs:
        return mask_out            
    struc_selected                          = np.where(zs==0)[0].astype(int)
    indx                                    = np.vstack(array_struc_indx[struc_selected]).astype(int)
    mask_out[:, indx[:,0], indx[:,1], :] = reference[indx[:,0], indx[:,1], :] 
    return mask_out

# -----------------------------------------------------------------------------------------------------------------------------------------------------------------------------
# Calculate the SHAP values
# -----------------------------------------------------------------------------------------------------------------------------------------------------------------------------
array_struc_indx   = get_structure_indices()
explainer          = shap.KernelExplainer(f, np.zeros((1,nmask+1)))
kernel_shap_values = explainer.shap_values(np.ones((1,nmask+1)))
 #%%
# -----------------------------------------------------------------------------------------------------------------------------------------------------------------------------
# Create a matrix of SHAP values
# -----------------------------------------------------------------------------------------------------------------------------------------------------------------------------
dim_shap_out    = len(kernel_shap_values[0,0,:])
shap_values_mat = [[] for ind_i in np.arange(dim_shap_out)]
for ind_i in np.arange(len(kernel_shap_values[0,0,:])):
    dim_shap_0             = len(kernel_shap_values[:,0,ind_i])
    dim_shap_1             = len(kernel_shap_values[0,:,ind_i])
    shap_values_mat[ind_i] = np.zeros((size_x,size_y))
    for ind_j in np.arange(dim_shap_0):
        for ind_k in np.arange(dim_shap_1):
            ind_jk                                          = np.vstack(array_struc_indx[ind_k]).astype(int)
            shap_values_mat[ind_i][ind_jk[:,0],ind_jk[:,1]] = kernel_shap_values[ind_j,ind_k,ind_i]
#%%
# -----------------------------------------------------------------------------------------------------------------------------------------------------------------------------
# Plot the results
# -----------------------------------------------------------------------------------------------------------------------------------------------------------------------------
fig, axes = plt.subplots(1, 10, figsize=(25, 4))
for ind_i, ax in enumerate(axes.flat):
    print(ind_i)
    maxshap = np.max(abs(kernel_shap_values[:,:,ind_i]))
    ax.set_title("Compared to "+str(ind_i))
    ax.pcolor(np.flip(x_test[0,:,:,0],axis=(0)),cmap="Greys")
    ax.pcolor(np.flip(shap_values_mat[ind_i],axis=(0)),alpha=0.5,vmin=-maxshap,vmax=maxshap,cmap="bwr")
    ax.set_xticks(range(0,size_x,box_size))
    ax.set_yticks(range(0,size_y,box_size))
    ax.grid()
    ax.set_xticklabels([])
    ax.set_yticklabels([])
plt.savefig("kernel_shap.png")


#%%
# -----------------------------------------------------------------------------------------------------------------------------------------------------------------------------
# Calculate the SHAP values using the GradientExplainer
# -----------------------------------------------------------------------------------------------------------------------------------------------------------------------------

# -----------------------------------------------------------------------------------------------------------------------------------------------------------------------------
# Define the input to calculate the SHAP values 
# -----------------------------------------------------------------------------------------------------------------------------------------------------------------------------
Xin = x_test[0].reshape(1,size_x,size_y,1)

# -----------------------------------------------------------------------------------------------------------------------------------------------------------------------------
# Since we have two inputs we pass a list of inputs to the explainer. GradientExplainer will calculate a SHAP value for each input feature.
# Any required function should be included in the tensorflow or pytorch model.
# -----------------------------------------------------------------------------------------------------------------------------------------------------------------------------
explainer            = shap.GradientExplainer(model, [x_train, x_train])
gradient_shap_values = explainer.shap_values([Xin, Xin])

# -----------------------------------------------------------------------------------------------------------------------------------------------------------------------------
# Plot the explanations for all classes for the first input (this is the feed forward input)
# -----------------------------------------------------------------------------------------------------------------------------------------------------------------------------
# -----------------------------------------------------------------------------------------------------------------------------------------------------------------------------
# Plot the results
# -----------------------------------------------------------------------------------------------------------------------------------------------------------------------------
fig, axes = plt.subplots(2, 10, figsize=(25, 8))
for ind_i in np.arange(10):
    maxshap = np.max([np.max(abs(gradient_shap_values[0][0,:,:,0,ind_i])),np.max(abs(gradient_shap_values[1][0,:,:,0,ind_i]))])
    axes[0,ind_i].set_title("Compared to "+str(ind_i))
    axes[0,ind_i].pcolor(np.flip(x_test[0,:,:,0],axis=(0)),cmap="Greys")
    axes[0,ind_i].pcolor(np.flip(gradient_shap_values[0][0,:,:,0,ind_i],axis=(0)),alpha=0.5,vmin=-maxshap,vmax=maxshap,cmap="bwr")
    axes[0,ind_i].set_xticks(range(0,size_x,1))
    axes[0,ind_i].set_yticks(range(0,size_y,1))
    axes[0,ind_i].grid()
    axes[0,ind_i].set_xticklabels([])
    axes[0,ind_i].set_yticklabels([])
    axes[1,ind_i].pcolor(np.flip(x_test[0,:,:,0],axis=(0)),cmap="Greys")
    axes[1,ind_i].pcolor(np.flip(gradient_shap_values[1][0,:,:,0,ind_i],axis=(0)),alpha=0.5,vmin=-maxshap,vmax=maxshap,cmap="bwr")
    axes[1,ind_i].set_xticks(range(0,size_x,1))
    axes[1,ind_i].set_yticks(range(0,size_y,1))
    axes[1,ind_i].grid()
    axes[1,ind_i].set_xticklabels([])
    axes[1,ind_i].set_yticklabels([])
    if ind_i == 0:
        axes[0,0].set_ylabel("Input 1")
        axes[1,0].set_ylabel("Input 2")
plt.savefig("gradient_shap.png")


#%%
# -----------------------------------------------------------------------------------------------------------------------------------------------------------------------------
# Calculate the SHAP values using the DeepExplainer
# -----------------------------------------------------------------------------------------------------------------------------------------------------------------------------

# -----------------------------------------------------------------------------------------------------------------------------------------------------------------------------
# Define the input to calculate the SHAP values 
# -----------------------------------------------------------------------------------------------------------------------------------------------------------------------------
Xin = x_test[0].reshape(1,size_x,size_y,1)

# -----------------------------------------------------------------------------------------------------------------------------------------------------------------------------
# Since we have two inputs we pass a list of inputs to the explainer. GradientExplainer will calculate a SHAP value for each input feature.
# Any required function should be included in the tensorflow or pytorch model.
# -----------------------------------------------------------------------------------------------------------------------------------------------------------------------------
explainer        = shap.DeepExplainer(model, [x_train[:100], x_train[:100]])
deep_shap_values = explainer.shap_values([Xin, Xin])

# -----------------------------------------------------------------------------------------------------------------------------------------------------------------------------
# Plot the explanations for all classes for the first input (this is the feed forward input)
# -----------------------------------------------------------------------------------------------------------------------------------------------------------------------------
# -----------------------------------------------------------------------------------------------------------------------------------------------------------------------------
# Plot the results
# -----------------------------------------------------------------------------------------------------------------------------------------------------------------------------
fig, axes = plt.subplots(2, 10, figsize=(25, 8))
for ind_i in np.arange(10):
    maxshap = np.max([np.max(abs(deep_shap_values[0][0,:,:,0,ind_i])),np.max(abs(deep_shap_values[1][0,:,:,0,ind_i]))])
    axes[0,ind_i].set_title("Compared to "+str(ind_i))
    axes[0,ind_i].pcolor(np.flip(x_test[0,:,:,0],axis=(0)),cmap="Greys")
    axes[0,ind_i].pcolor(np.flip(deep_shap_values[0][0,:,:,0,ind_i],axis=(0)),alpha=0.5,vmin=-maxshap,vmax=maxshap,cmap="bwr")
    axes[0,ind_i].set_xticks(range(0,size_x,1))
    axes[0,ind_i].set_yticks(range(0,size_y,1))
    axes[0,ind_i].grid()
    axes[0,ind_i].set_xticklabels([])
    axes[0,ind_i].set_yticklabels([])
    axes[1,ind_i].pcolor(np.flip(x_test[0,:,:,0],axis=(0)),cmap="Greys")
    axes[1,ind_i].pcolor(np.flip(deep_shap_values[1][0,:,:,0,ind_i],axis=(0)),alpha=0.5,vmin=-maxshap,vmax=maxshap,cmap="bwr")
    axes[1,ind_i].set_xticks(range(0,size_x,1))
    axes[1,ind_i].set_yticks(range(0,size_y,1))
    axes[1,ind_i].grid()
    axes[1,ind_i].set_xticklabels([])
    axes[1,ind_i].set_yticklabels([])
    if ind_i == 0:
        axes[0,0].set_ylabel("Input 1")
        axes[1,0].set_ylabel("Input 2")
plt.savefig("deep_shap.png")