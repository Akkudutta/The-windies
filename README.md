# The-windies
This is a Streamlit WebApp developed to present my Data Science project for predicting Windmill Power Generation based on actual historical data of 2018-2019.

# The-Windies Introduction
This is a Streamlit WebApp developed to present my Data Science project for predicting Windmill Power Generation based on actual historical data of 2018-2019.

This was actually the presentation used by IIM Lucknow's EPDS-03 batch students for one of their projects. 
The project is called Windmill Power Prediction and the name of the group is The Windies.
The actual project was prepared using R Studio, codes of which you can find in the repo named Windmill RCodev2.R
I have pasted the link for the Deployed Web App along with the Kaggle Data Set which was used for this project.

This Web App was developed in Python using Streamlit. Apart from the lottie animations and Plots with descriptions, while accessing the app, you can find a tab name Predict Power in the sidebar. If click on that, there will be fields where you can fill some values and actually predict the Windmill Power Generated on your own.
This app is implemented with a Machine Learing Model - Random Forest Regressor, which has been fitted using the Data Set aquired from the Kaggel link.
There is a seperate py file named ml.py where the machine learning model has been fitted and then it is saved in a seperat sav file which can be generated by running the py file. The same sav file has been imported in the app to run the ml algorithms for predicting Power using the values enter in the fields.

# Link to Web App
https://akkudutta-the-windies-app-3xmhza.streamlit.app/
