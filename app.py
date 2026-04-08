from flask import Flask, render_template, request, redirect, session, jsonify
import pandas as pd
from sklearn.ensemble import RandomForestClassifier

app = Flask(__name__)
app.secret_key = "quickbyte"

# LOAD DATA
data = pd.read_csv("restaurant_rush.csv")
menu = pd.read_csv("menu.csv")

# TRAIN MODEL
data['rush'] = data['rush'].map({'Low':0,'Medium':1,'High':2})
model = RandomForestClassifier()
model.fit(data[['hour','day','weekend']], data['rush'])

# LOGIN
@app.route('/', methods=['GET','POST'])
def login():
    if request.method == 'POST':
        session['user'] = request.form['username']
        return redirect('/home')
    return render_template("login.html")

# HOME
@app.route('/home')
def home():
    menu_dict = {}
    for _, row in menu.iterrows():
        menu_dict.setdefault(row['restaurant'], []).append(row['food_item'])

    # ⭐ BEST RESTAURANT (LOWEST PREP TIME)
    best_restaurant = ""
    best_time = 999

    for r in menu_dict:
        min_time = menu[menu['restaurant'] == r]['prep_time'].min()
        if min_time < best_time:
            best_time = int(min_time)
            best_restaurant = r

    return render_template(
        "index.html",
        menu=menu_dict,
        best=best_restaurant,
        best_time=best_time,
        rush="Low"
    )

# DASHBOARD
@app.route('/dashboard')
def dashboard():
    menu_dict = {}
    for _, row in menu.iterrows():
        menu_dict.setdefault(row['restaurant'], []).append(row['food_item'])

    return render_template("dashboard.html",
        menu=menu_dict,
        rush=session.get("rush"),
        wait=session.get("wait"),
        best=session.get("best"),
        best_time=session.get("best_time"),
        selected=session.get("selected"),
        labels=session.get("labels"),
        values=session.get("values"),
        trend=session.get("trend")
    )

# PREDICT
@app.route('/predict', methods=['POST'])
def predict():
    hour = int(request.form['hour'])
    ampm = request.form['ampm']

    if ampm == "PM" and hour != 12:
        hour += 12
    if ampm == "AM" and hour == 12:
        hour = 0

    day = int(request.form['day'])
    weekend = 1 if day >= 5 else 0

    restaurant = request.form['restaurant']
    food = request.form['food']

    pred = model.predict(pd.DataFrame([[hour,day,weekend]],
        columns=['hour','day','weekend']))

    rush_map = {0:"Low",1:"Medium",2:"High"}
    rush = rush_map[int(pred[0])]

    row = menu[(menu['restaurant']==restaurant)&(menu['food_item']==food)]
    prep = int(row['prep_time'].values[0]) if not row.empty else 10

    factor = {"Low":2,"Medium":5,"High":10}
    wait = prep + factor[rush]

    best = ""
    min_time = 999
    labels = []
    values = []

    for r in menu['restaurant'].unique():
        items = menu[menu['restaurant']==r]
        if not items.empty:
            t = int(items['prep_time'].min()) + factor[rush]
            labels.append(r)
            values.append(t)

            if t < min_time:
                min_time = t
                best = r

    trend = [1,2,3,2]

    session['rush'] = rush
    session['wait'] = wait
    session['best'] = best
    session['best_time'] = min_time
    session['selected'] = restaurant
    session['labels'] = labels
    session['values'] = values
    session['trend'] = trend

    return jsonify({
        "rush": rush,
        "wait": wait,
        "labels": labels,
        "values": values,
        "trend": trend
    })

if __name__ == "__main__":
    app.run(debug=True)