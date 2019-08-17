from flask import Flask, redirect, url_for, request, render_template

app = Flask(__name__)



'''
app setup
'''
@app.route('/home', methods=['GET', 'POST'])
def homePageUpdate():
    if request.method == 'POST':
        num = request.form['type']
    else:
        num = request.args.get['type']
    
    return render_template('home.html', number=num)



if __name__ == '__main__':
   app.run(host='0.0.0.0', port=5000) # host 0.0.0.0 makes available externally, port defaults to 5000




