# -*- coding: utf-8 -*-
from flask import Flask, render_template, request, redirect
from sklearn.externals import joblib
import pandas as pd
import numpy as np
app = Flask(__name__)
#export FLASK_APP=web.py
@app.route('/')
def hello(name=None):
    return render_template('input.html', name=name)


@app.route('/', methods=['POST'])
def my_form_post():    
    m1 ='''<!DOCTYPE html>
<html>
    <head>
        <meta charset="utf-8">
        <title>Company Credit Ranking Prediction</title>
        <meta name ='description' content="Company Credit Ranking Prediction.">

        <style>
        body{
          font-family: "Avenir Next", Avenir, Roboto, "Century Gothic";
          color: rgb(0, 0, 0);
        }
          h1{
            height:80px; 
            padding-top: 20px;
            color: rgb(255, 255, 255);
          }

          #coffee_his{
            background-image: url("");
            overflow-y: auto;
            overflow-x: hidden;
            padding-top: 20px;
            background-size: 100% 100%;
            height: 250px;
            width:  70%;
            padding-top: 20px;
            padding-bottom: 50px;
          }

          #coffee_tittle{
            align-content: center;
            background-image: url("http://cdn.wonderfulengineering.com/wp-content/uploads/2016/04/dark-wallpaper-37-610x343.jpg");
            width: 70%;
            background-size: 250px 100%;
            background-repeat: repeat;
            font-family: "Avenir Next", Avenir, Roboto, "Century Gothic";
                      }
          #origin{
            padding-top: 20px;
            background-size: 100% 100%;
            height: 600px;
            width:  70%;
            background-image:url("");
          }

          .each_origin{
            display: flex;
            flex-direction: row-reverse;
            padding-left:10px;
            padding-top: 15px;
          }
          .oriname{
            font-weight: bold;
          }
          .origin_des{
          padding-left:25px;
          padding-right: 30px;  
          }
          #coffee_roast{
            background-image: url("https://cdn.wallpaperjam.com/content/images/c1/8d/c18d59c1ed48212e181a1534e78de3464f9175d7.png");
            padding-top: 10px;
            padding-bottom: 20px;
            background-size: 100% 100%;
            height: 500px;
            width:  70%;
            
          }
          #roaststyle{
            font-family: Georgia, Garamond, serif;
          }

          h2{
            color:rgb(0, 0,0);
          }
          h5{
            color:rgb(255,255,255);
          }
          #bottomimg{
            background-image: url("http://res.cloudinary.com/tempest/image/upload/c_limit,cs_srgb,dpr_1.0,q_80,w_680/MTI5MDIxMjE5MTUxNDI0MTMx.jpg");
            padding-top: 10px;
            padding-bottom: 20px;
            background-size: 70% 100%;
            height: 200px;
            width:  100%;
            background-repeat: repeat;
            text-align: center;
            vertical-align:middle;
            text-color:rgb(0,0,0);
          }
         #map{
          margin-top: 50%;
          text-decoration: none;
          text-transform: uppercase;
          clear:both;
          
         }
         #brandlink{
          float:right;
          background-color: none;
          width:30%;
          height: 1590px;
          background-image:url("https://wallpapertag.com/wallpaper/full/2/6/0/283037-download-free-marble-background-2048x1347-xiaomi.jpg");
          background-repeat: repeat;
         }
        </style>
    </head>
    <body>

    <div id='brandlink'>
    <h2 id='company'>&nbsp;Company Bankruptcy News</h2>
    <ul>
      <li><h3>Trustee gives Toys 'R' Us executive bonus plan a lump of coal<h3><h4>
Published 1 Dec 2017<h4><p>The government's bankruptcy watchdog is criticizing Toys 'R' Us over the retailer's executive bonus plan. Also, the plan, which would pay up to $32 million to 17 executives, "defies logic," the government group says.<p></li>
      <li><h3>Radio broadcaster Cumulus files for bankruptcy protection<h3><h4>Published 30 Nov 2017<h4><p>Radio broadcasting company Cumulus Media has filed for Chapter 11 bankruptcy protection and entered into a restructuring agreement with some of its lenders to reduce its debt by $1 billion.<p></li>
      <li><h3>Lufthansa to Buy Units of Air Berlin for $249 Million<h3><h4>Published OCT. 12, 2017<h4><p>The German flag carrier Lufthansa signed an agreement on Thursday to buy sections of Air Berlin, a low-cost carrier that had filed for insolvency this summer.<p></li>
      <li><a id='map' href="http://query.nytimes.com/gst/abstract.html?res=9F0CE7DE1E39E333A25757C1A9609C946196D6CF&legacy=true">Cut cost of becoming bankruptcy</a></li>
      <li><a id='map' href="http://fortune.com/2015/10/05/retail-bankruptcy/">Largest retailer bankruptcies of last decade</a></li>
      <li><a id='map' href="www.businessinsider.com/debt-restructuring-and-bankruptcy-may-be-biggest-growth-industry-2017-2017-2">Debt restructuring and bankruptcy could be the biggest growth industry of 2017 </a></li>
    </ul>

    
    
    </div>
    <div id='coffee_tittle'>
    <h1><center>Company Credit Ranking Prediction!</center></h1>
    </div>

    <div id='coffee_his'>
      <div class="col-md-5">
          <img class="featurette-image img-responsive center-block" img src="http://www.paladinenergy.com.au/sites/default/files/getfile%20%2812%29.jpg" alt="The situation of Bankruptcy in Dataset">
        </div>
      <div class="col-md-4">
                    <div class="custom-accordion waves-effect">
                        <!-- Start Accordion Section -->
                    <div class="panel-group" id="accordion">

                        <!-- Start Accordion 1 -->
                        <div class="panel panel-default">
                            <div class="panel-heading waves-effect">
                                <h4 class="panel-title">
                                    <a data-toggle="collapse" data-parent="#accordion" href="#collapse-1">
                                        <i class="fa fa-angle-left control-icon"></i> Who We are?
                                    </a>
                                </h4>
                            </div>
                            <div id="collapse-1" class="panel-collapse collapse in">
                                <div class="panel-body"><h4>We are a company who focus on the data research,mining and analysis on finance</h4></div>                               </div>
                        </div>
                        <!-- End Accordion 1 -->

                        <!-- Start Accordion 2 -->
                        <div class="panel panel-default">
                            <div class="panel-heading waves-effect">
                                <h4 class="panel-title">
                                    <a data-toggle="collapse" data-parent="#accordion" href="#collapse-2" class="collapsed">
                                        <i class="fa fa-angle-left control-icon"></i> What we do?
                                    </a>
                                </h4>
                            </div>
                            <div id="collapse-2" class="panel-collapse collapse">
                                <div class="panel-body"><h4>We can analyzes the financial situations of your company and show your the grade of the company credit. It will definitely help you to decide whether to make investment in a certain company or how to improve your company</h4></div>                               </div>
                        </div>
                        <!-- End Accordion 2 -->

                        <!-- Start Accordion 3 -->
                        <div class="panel panel-default">
                            <div class="panel-heading waves-effect">
                                <h4 class="panel-title">
                                    <a data-toggle="collapse" data-parent="#accordion" href="#collapse-3" class="collapsed">
                                        <i class="fa fa-angle-left control-icon"></i> Why Choose Us ?
                                    </a>
                                </h4>
                            </div>
                            <div id="collapse-3" class="panel-collapse collapse"><h4>We have the most professional team of experts from Columbia University. Some of them are adept at modeling, some of them are good at visualization and others specialize in finance.</h4>
                                <div class="panel-body">.</div>
                            </div>
                        </div>
        </div>
      </div>
   </div>
</div>

    
<div id ="origin">
      <h2 id='coffee_origin'>Results of Ranking Prediction</h2>
      
     <h3>
            Will ''' 
    m2 =''' Default
     </h3>
     
     <h3>
     Rating of Company: '''
     
    m3 = '''
     </h3>
     <h3>
     Profitability: '''
    m4 = '''
     </h3>
     <h3>
     Operating: ''' 
     
    m5 = '''
     </h3>
     <h3>
     Liquidity: '''
     
    m6 = '''
      </div>

    <div id='coffee_roast'>
    <h2 >Rating Scale for Company</h2>
    <table id='roaststyle' border="1" bordercolor='white' >
      <p>We have totally nine letter grades from AAA to D. The higher the Letter Grade, the more worthy for others to make investment. 
      </p>
     <thead>
       <tr id='stylename'>
         <th>Letter Grade</th>
         <th>Grade</th>
         <th>Capacity to Repay</th>
       </tr>
     </thead>
     <tbody>
       <tr class='tablecontent'>
         <td>AAA</td>
         <td>Investment</td>
         <td>Extremely strong</td>
       </tr>
       <tr class='tablecontent'>
         <td>AA</td>
         <td>Investment</td>
         <td>Very strong</td>
       </tr>
       <tr class='tablecontent'>
         <td>A</td>
         <td>Investment</td>
         <td>Strong</td>
       </tr>
       <tr class='tablecontent'>
         <td>BBB</td>
         <td>Investment</td>
         <td>Adequate</td>
       </tr>
       <tr class='tablecontent'>
         <td>BB</td>
         <td>Speculative</td>
         <td>Faces major future uncertainties</td>
       </tr>
       <tr class='tablecontent'>
         <td>B</td>
         <td>Speculative</td>
         <td>Faces major uncertainties</td>
       </tr>
	   <tr class='tablecontent'>
         <td>CCC</td>
         <td>Speculative</td>
         <td>Currently vulnerable</td>
       </tr>
	   <tr class='tablecontent'>
         <td>C</td>
         <td>Speculative</td>
         <td>Has filed bankruptcy petition</td>
       </tr>
	   <tr class='tablecontent'>
         <td>D</td>
         <td>Speculative</td>
         <td>In default</td>
       </tr>
     </tbody>
    </table>
  </div>

    
   <div id="bottomimg">
   <footer>
        <div class="container">
            <div class="row">
                <div class="col-md-4 col-xs-12">
                    <span class="copyright">&copy;<h5>@ Project Data Science 2017<h5></span>
                    <h5>Contact:rs3770@columbia.com &nbsp;&nbsp;&nbsp;&nbsp; Fax:212-058-079 &nbsp;&nbsp;&nbsp;&nbsp;&nbsp;</h5>  
                    <h5>Address: 116 Str. Broadway</h5>
                    <a id='map' href="https://www.google.com/maps/place/Columbia+University/@40.8075395,-73.9647614,17z/data=!3m1!4b1!4m5!3m4!1s0x89c2f63e96d30dc9:0x577933f947e52750!8m2!3d40.8075355!4d-73.9625727">Find location in Google map </a>
                </div>
             
                </div>
            </div>
    </footer>
    </div> 




    </body>
</html>
     '''

    data = pd.read_csv('data123.csv',index_col = 0)
    data2 = data[data.columns[2:-1]]
    mean = []
    std = []
    for i in data2.columns:
        mean.append(np.mean(data2[i]))
        std.append(np.std(data2[i]))
    names = []
    X = [1,1]
    for i in range(3,24):
        names.append(str(i))
    for name in names:
        text = request.form[name]
        X.append(float(text.upper()))
    X2 = [63,1]
    for j in range(2,len(X)):
        X2.append((X[j] - mean[j-2])/std[j - 2])
    
    model = joblib.load('/Users/shijiezhou/Documents/MSORII/Python/Project/RF.pkl')
    result = model.predict([X2])
    fir = ' NOT '
    if result[0] == 1:
        fir = ''
    
    prob = model.predict_proba([X2])
    s = "AAA"
    if prob[0][1] < 0.05:
        s = "AAA"
    elif prob[0][1] < 0.1:
        s = "AA"
    elif prob[0][1] < 0.15:
        s = "A"
    elif prob[0][1] < 0.2:
        s = "BBB"
    elif prob[0][1] < 0.25:
        s = "BB"
    elif prob[0][1] < 0.3:
        s = "B"
    elif prob[0][1] < 0.35:
        s = "CCC"
    elif prob[0][1] < 0.5:
        s = "C"
    else:
        s = "D"
    pro = "Medium"
    liq = "Medium"
    ope = "Medium"
    if X[2] > (mean[0] + 1 * std[0]):
        pro = "High"
    elif X[2] < (mean[0] - 1 * std[0]):
        pro = "Low"
        
        
    if X[3] > mean[1] + 1 * std[1]:
        liq = "High"
    elif X[3] < mean[1] - 1 * std[1]:
        liq = "Low"
        
        
    if X[-2] > mean[-2] + 1 * std[-2]:
        ope = "High"
    elif X[-2] < mean[-2] - 1 * std[-2]:
        ope = "Low"
                
                
    #fir = str(mean[0] + 1 * std[0])         
    m = m1 + fir + m2 + s + m3 + pro + m4 + liq + m5 + ope + m6
    f = open('templates/test4.html','w')
    f.write(m)
    f.close()
    return redirect('success',code=307)
    
    # read the posted values from the UI
#    _name = request.form['inputName']
#    _email = request.form['inputEmail']
#    _password = request.form['inputPassword']
#    return _name
#    # validate the received values
#    if _name and _email and _password:
#        print("OK")
#        return "OKOK"
#    else:
##        return "No"
#@app.route('/login', methods = ['POST'])
#def login():
#    if request.method == 'POST' and request.form.get("username") == 'admin':
#        return redirect(url_for('success',data=request.form.get("data")),code=307)
#    else:
#        return redirect(url_for('index'))
@app.route("/success", methods=['POST'])
def success(name=None):
    return render_template('test4.html', name=name)