# Next Gen Smart Home: From Radar to Physiological Signal

### Problem Statement
I worked with MIT Device Realization Lab and Sekisui house on developing the next generation of smart homes in Japan. Specifically, we focused on translating radar signal to physiological signal using deep learning approach.

Our project is initiated under the context of aging Japan. Sekisui House plans on using radar sensors in the ceiling of Smart Homes to monitor their senior residents' health. These sensors are great because they don’t intrude on people’s privacy, unlike cameras, but they also pick up a lot of noise, from simple movements to heartbeats, and even air particles. My job was to develop a model that would separate out the useful information from the environmental noise.

### Result

Two main models are developed in our project to predict heart rate and respiration rate from radar signal. Drawing inspiration from computer vision, we based our first model on the ResNet-18 to predict per minute heart rate and respiration rate. In model 2, by adding an encoder-decoder structure, we predict the timestamp of heart pulses. 

The first model achieved test set mean absolute error of 2.40 on heart rate bpm (R2 0.90) and 1.34 on respiration rate (R2 0.78), robust against large movements such as walking and jumping. The second model achieved test set accuracy of 88.72%, accurate to 0.1 second.

The data is private to Sekisui House and therefore not disclosed. You can see our poster <a href='https://static1.squarespace.com/static/6095834ad540ff11ba51a369/t/64de55d69a5f8d2596f9ce5b/1692292566811/MIT_Device_Realization_Lab_Rachel_Duan_Nicky_Zhang.pdf'>here</a>.