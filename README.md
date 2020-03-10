# Synopsis 2020
  

# Project Introduction:

  Fuel combustion is a highly complicated process because it is affected by many physical and chemical
  factors that make the prediction and measurement of ignition delay time a complicated, slow, and 
  expensive process. Ignition delay time is one of the most commonly used control properties in engine 
  combustion research and is defined as the time taken by the fuel to auto-ignite after being injected 
  into the combustion chamber. Traditional experimental methods, such as using infrared radiation 
  detector monitors and time-based pressure sensor measurements, are highly dependent on engine 
  types and are inefficient in general applications due to the multiple variables involved and their 
  non-linear relationships. With the advancement of artificial neural networks, some complex and 
  non-linear models for ignition delay time predictions have been successfully created and have 
  shown very promising results. However, the majority of published neural network models have 
  unchangeable input layer dimensions, which does not encompass different types of fuels or engines. 
  In this project, I introduced new artificial neural network models that include expandable input 
  layer dimensions based on dataset shapes and demand-based numbers of hidden layers and neurons. 
  This project has significantly increased the capabilities of neural network models and the simulation tool, 
  which has wide applications in both rocket and Scramjet combustor designs. 
  
# How to Run the Artfifical Neural Network Tool:
    *************************
    Prerequisites;
        a. Python (version 2.7 to python 3.6, python 3.7 is not supported)
        b. Keras version 2.3.0
    *************************
  1. download the tool from: https://github.com/zhouemily/RegeneronSTS
  2. cd the path where idt.py installed
  3. ./run.sh
