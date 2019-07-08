from flask import Flask, jsonify, request,render_template,url_for,request
import pandas as pd
from flask_bootstrap import Bootstrap 
import os
from sklearn.externals import joblib
#from sklearn.linear_model import LinearRegression

app = Flask(__name__)
Bootstrap(app)

    

#
@app.route('/')


def index(): 
   return render_template('index.html')

@app.route("/predict", methods=['POST'])
def predict():

    #print("lolllllll")
    
   # print(us_canada_user_rating_pivot)
    
    if request.method == 'POST':
        try:
            model_knn = joblib.load("./recommender_model.pkl")
            #data = request.get_json()
            #years_of_experience = float(data["yearsOfExperience"])

            us_canada_user_rating = pd.read_csv('/home/siddanath/python_project/recommender_api/testcorrect_csv.csv')
            us_canada_user_rating_pivot = us_canada_user_rating.pivot(index = 'bookTitle', columns = 'userID', values = 'bookRating').fillna(0)
            namequery = request.form['namequery']
            #data = [namequery]
           # years_of_experience = float(data)

	    #vect = cv.transform(data).toarray()
            #my_prediction = lin_reg.predict(namequery).tolist()
            #query_index = np.random.choice(us_canada_user_rating_pivot.shape[0])
            distances, indices = model_knn.kneighbors(us_canada_user_rating_pivot.loc[namequery].values.reshape(1, -1), n_neighbors = 6)
            ans = list()
            for i in range(0, len(distances.flatten())):
               if i == 0:
                  pass
                  #print('Recommendations for {0}:\n'.format("10 Lb. Penalty"))
               else:
                 #print('{0}: {1}, with distance of {2}:'.format(i, us_canada_user_rating_pivot.index[indices.flatten()[i]], distances.flatten()[i]))
                 ans.append(us_canada_user_rating_pivot.index[indices.flatten()[i]])

        except ValueError:
            return jsonify("Please enter a number.")
       # print(ans)
        return render_template('results.html',prediction = ans,name = namequery.upper())
        #return jsonify(lin_reg.predict(years_of_experience).tolist())


if __name__ == '__main__':
    app.run(debug=True)
