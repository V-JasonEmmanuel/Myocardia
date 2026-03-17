from skimage.io import imread
from skimage import color
import matplotlib.pyplot as plt
import streamlit as st
from skimage.filters import threshold_otsu,gaussian
from skimage.transform import resize
from numpy import asarray
from skimage.metrics import structural_similarity
from skimage import measure
from sklearn.decomposition import PCA
from sklearn.neighbors import KNeighborsClassifier
import joblib
from sklearn.preprocessing import MinMaxScaler
import pandas as pd
import numpy as np
import os
from natsort import natsorted
from pathlib import Path

uploaded_file = st.file_uploader("Choose a file")

if uploaded_file is not None:
  image=imread(uploaded_file)
  image_gray = color.rgb2gray(image)
  image_gray=resize(image_gray,(1572,2213))
  """#### **UPLOADED ECG IMAGE**"""
  
  #checkign if we parse the user image and similar to our format
  # locate project root (parent of this script's folder) and dataset/model locations relative to it
  project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
  dataset_dir = os.path.join(project_root, 'ECG_IMAGES_DATASET')
  if not os.path.isdir(dataset_dir):
    # try to find any folder named ECG_IMAGES_DATASET under project
    candidates = list(Path(project_root).rglob('ECG_IMAGES_DATASET'))
    dataset_dir = str(candidates[0]) if candidates else None

  def find_image_by_keyword(keyword):
    if not dataset_dir:
      return None
    matches = list(Path(dataset_dir).rglob(f'*{keyword}*.jpg'))
    if matches:
      return str(matches[0])
    # fallback: any jpg
    alljpg = list(Path(dataset_dir).rglob('*.jpg'))
    return str(alljpg[0]) if alljpg else None

  # try to find representative sample images used for similarity check
  paths = [find_image_by_keyword(k) for k in ('PMI', 'HB', 'Normal', 'MI')]
  sample_images = []
  for p in paths:
    if p and os.path.isfile(p):
      try:
        img = imread(p)
        img = color.rgb2gray(img)
        img = resize(img, (1572,2213))
        sample_images.append(img)
      except Exception:
        # ignore unreadable images
        pass

  # compute similarity only against available sample images
  sim_scores = []
  for s in sample_images:
    try:
      sim_scores.append(structural_similarity(image_gray, s))
    except Exception:
      pass

  similarity_score = max(sim_scores) if sim_scores else 0.0

  if similarity_score > 0.70:
    st.image(image)
    """#### **GRAY SCALE IMAGE**"""
    my_expander = st.expander(label='Gray SCALE IMAGE')
    with my_expander: 
      st.image(image_gray)
    """#### **DIVIDING LEADS**"""
    #dividing the ECG leads from 1-13 from the above image
    Lead_1 = image[300:600, 150:643]
    Lead_2 = image[300:600, 646:1135]
    Lead_3 = image[300:600, 1140:1625]
    Lead_4 = image[300:600, 1630:2125]
    Lead_5 = image[600:900, 150:643]
    Lead_6 = image[600:900, 646:1135]
    Lead_7 = image[600:900, 1140:1625]
    Lead_8 = image[600:900, 1630:2125]
    Lead_9 = image[900:1200, 150:643]
    Lead_10 = image[900:1200, 646:1135]
    Lead_11 = image[900:1200, 1140:1625]
    Lead_12 = image[900:1200, 1630:2125]
    Lead_13 = image[1250:1480, 150:2125]
    Leads=[Lead_1,Lead_2,Lead_3,Lead_4,Lead_5,Lead_6,Lead_7,Lead_8,Lead_9,Lead_10,Lead_11,Lead_12,Lead_13]
    #plotting lead 1-12
    fig , ax = plt.subplots(4,3)
    fig.set_size_inches(10, 10)
    x_counter=0
    y_counter=0

    for x,y in enumerate(Leads[:len(Leads)-1]):
      if (x+1)%3==0:
        ax[x_counter][y_counter].imshow(y)
        ax[x_counter][y_counter].axis('off')
        ax[x_counter][y_counter].set_title("Leads {}".format(x+1))
        x_counter+=1
        y_counter=0
      else:
        ax[x_counter][y_counter].imshow(y)
        ax[x_counter][y_counter].axis('off')
        ax[x_counter][y_counter].set_title("Leads {}".format(x+1))
        y_counter+=1
    
    fig.savefig('Leads_1-12_figure.png')
    fig1 , ax1 = plt.subplots()
    fig1.set_size_inches(10, 10)
    ax1.imshow(Lead_13)
    ax1.set_title("Leads 13")
    ax1.axis('off')
    fig1.savefig('Long_Lead_13_figure.png')
    my_expander1 = st.expander(label='DIVIDING LEAD')
    with my_expander1:
      st.image('Leads_1-12_figure.png')
      st.image('Long_Lead_13_figure.png')

    """#### **PREPROCESSED LEADS**"""
    fig2 , ax2 = plt.subplots(4,3)
    fig2.set_size_inches(10, 10)
    #setting counter for plotting based on value
    x_counter=0
    y_counter=0

    for x,y in enumerate(Leads[:len(Leads)-1]):
      #converting to gray scale
      grayscale = color.rgb2gray(y)
      #smoothing image
      blurred_image = gaussian(grayscale, sigma=0.9)
      #thresholding to distinguish foreground and background
      #using otsu thresholding for getting threshold value
      global_thresh = threshold_otsu(blurred_image)

      #creating binary image based on threshold
      binary_global = blurred_image < global_thresh
      #resize image
      binary_global = resize(binary_global, (300, 450))
      if (x+1)%3==0:
        ax2[x_counter][y_counter].imshow(binary_global,cmap="gray")
        ax2[x_counter][y_counter].axis('off')
        ax2[x_counter][y_counter].set_title("pre-processed Leads {} image".format(x+1))
        x_counter+=1
        y_counter=0
      else:
        ax2[x_counter][y_counter].imshow(binary_global,cmap="gray")
        ax2[x_counter][y_counter].axis('off')
        ax2[x_counter][y_counter].set_title("pre-processed Leads {} image".format(x+1))
        y_counter+=1
    fig2.savefig('Preprossed_Leads_1-12_figure.png')
    
    #plotting lead 13
    fig3 , ax3 = plt.subplots()
    fig3.set_size_inches(10, 10)
    #converting to gray scale
    grayscale = color.rgb2gray(Lead_13)
    #smoothing image
    blurred_image = gaussian(grayscale, sigma=0.7)
    #thresholding to distinguish foreground and background
    #using otsu thresholding for getting threshold value
    global_thresh = threshold_otsu(blurred_image)
    print(global_thresh)
    #creating binary image based on threshold
    binary_global = blurred_image < global_thresh
    ax3.imshow(binary_global,cmap='gray')
    ax3.set_title("Leads 13")
    ax3.axis('off')
    fig3.savefig('Preprossed_Leads_13_figure.png')

    my_expander2 = st.expander(label='PREPROCESSED LEAD')
    with my_expander2:
      st.image('Preprossed_Leads_1-12_figure.png')
      st.image('Preprossed_Leads_13_figure.png')
    
    """#### **EXTRACTING SIGNALS(1-13)**"""
    fig4 , ax4 = plt.subplots(4,3)
    fig4.set_size_inches(10, 10)
    x_counter=0
    y_counter=0
    for x,y in enumerate(Leads[:len(Leads)-1]):
      #converting to gray scale
      grayscale = color.rgb2gray(y)
      #smoothing image
      blurred_image = gaussian(grayscale, sigma=0.9)
      #thresholding to distinguish foreground and background
      #using otsu thresholding for getting threshold value
      global_thresh = threshold_otsu(blurred_image)

      #creating binary image based on threshold
      binary_global = blurred_image < global_thresh
      #resize image
      binary_global = resize(binary_global, (300, 450))
      #finding contours
      contours = measure.find_contours(binary_global,0.8)
      contours_shape = sorted([x.shape for x in contours])[::-1][0:1]
      for contour in contours:
        if contour.shape in contours_shape:
          test = resize(contour, (255, 2))
      if (x+1)%3==0:
        ax4[x_counter][y_counter].plot(test[:, 1], test[:, 0],linewidth=1,color='black')
        ax4[x_counter][y_counter].axis('image')
        ax4[x_counter][y_counter].set_title("Contour {} image".format(x+1))
        x_counter+=1
        y_counter=0
      else:
        ax4[x_counter][y_counter].plot(test[:, 1], test[:, 0],linewidth=1,color='black')
        ax4[x_counter][y_counter].axis('image')
        ax4[x_counter][y_counter].set_title("Contour {} image".format(x+1))
        y_counter+=1
    
      #scaling the data and testing
      lead_no=x
      scaler = MinMaxScaler()
      fit_transform_data = scaler.fit_transform(test)
      Normalized_Scaled=pd.DataFrame(fit_transform_data[:,0], columns = ['X'])
      Normalized_Scaled=Normalized_Scaled.T
      #scaled_data to CSV
      if (os.path.isfile('scaled_data_1D_{lead_no}.csv'.format(lead_no=lead_no+1))):
        Normalized_Scaled.to_csv('Scaled_1DLead_{lead_no}.csv'.format(lead_no=lead_no+1), mode='a',index=False)
      else:
        Normalized_Scaled.to_csv('Scaled_1DLead_{lead_no}.csv'.format(lead_no=lead_no+1),index=False)
    
    fig4.savefig('Contour_Leads_1-12_figure.png')
    my_expander3 = st.expander(label='CONOTUR LEADS')
    with my_expander3:
      st.image('Contour_Leads_1-12_figure.png')

    """#### **CONVERTING TO 1D SIGNAL**"""    
    #lets try combining all 12 leads
    # load 1D scaled leads from the local colabs folder (project-relative)
    colabs_dir = os.path.join(project_root, 'colabs')
    if not os.path.isdir(colabs_dir):
      # fallback: current directory
      colabs_dir = os.path.dirname(__file__)

    first_csv = os.path.join(colabs_dir, 'Scaled_1DLead_1.csv')
    if not os.path.isfile(first_csv):
      st.error(f"Required CSV not found: {first_csv}")
      st.stop()

    test_final = pd.read_csv(first_csv)
    for fname in natsorted(os.listdir(colabs_dir)):
      if fname.endswith('.csv') and fname != 'Scaled_1DLead_1.csv':
        fpath = os.path.join(colabs_dir, fname)
        try:
          df = pd.read_csv(fpath)
          test_final = pd.concat([test_final, df], axis=1, ignore_index=True)
        except Exception:
          # skip unreadable CSVs
          pass
    
    st.write(test_final)
    """#### **PASS TO ML MODEL FOR PREDICTION**"""
    # load model from project-relative Deployment folder
    model_path = os.path.join(project_root, 'Deployment', 'Heart_Disease_Prediction_using_ECG.pkl')
    if not os.path.isfile(model_path):
      # try to find a .pkl in Deployment
      dep_dir = os.path.join(project_root, 'Deployment')
      if os.path.isdir(dep_dir):
        pkls = list(Path(dep_dir).rglob('*.pkl'))
        model_path = str(pkls[0]) if pkls else None

    if not model_path or not os.path.isfile(model_path):
      st.error('Model file not found. Expected a .pkl under the project Deployment folder.')
      st.stop()

    loaded_model = joblib.load(model_path)
    result = loaded_model.predict(test_final)
    if result[0] == 0:
      st.write("You ECG corresponds to Myocardial Infarction")
    
    if result[0] == 1:
      st.write("You ECG corresponds to Abnormal Heartbeat")
    
    if result[0] == 2:
      st.write("Your ECG is Normal")
    
    if result[0] == 3:
      st.write("You ECG corresponds to History of Myocardial Infarction")
    
  else:
    st.write("Sorry Our App won't be able to parse this image format right now!!!. Pls check the image input sample section for supported images")
