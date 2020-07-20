from flask import Flask, render_template, request
import pickle
app = Flask(__name__)


file = open('model.pkl','rb')
clf = pickle.load(file)
file.close()

@app.route('/', methods = ["GET","POST"])
def hello_world():
    if request.method == "POST":
        myDict = request.form
        fever = int(myDict['fever'])
        pain = int(myDict['pain'])
        age = int(myDict['age'])
        runnyNose = int(myDict['nose'])
        diffBreath = int(myDict['breath'])
        enter = [fever, pain, age, runnyNose, diffBreath]
        infProb = clf.predict_proba([enter])[0][1]
        print(infProb)
        return render_template('show.html', inf=round(infProb*100))
    return render_template('layout.html')

if __name__ == '__main__':
    app.run(debug=True)