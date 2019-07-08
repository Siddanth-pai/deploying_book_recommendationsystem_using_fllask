from flask import Flask, jsonify, request,render_template,url_for,request
import pandas as pd
from flask_bootstrap import Bootstrap 
import os
from sklearn.externals import joblib



us_canada_user_rating_pivot = pd.read_csv('/home/siddanath/python_project/recommender_api/clean_data_for_api.csv')
print(us_canada_user_rating_pivot)
