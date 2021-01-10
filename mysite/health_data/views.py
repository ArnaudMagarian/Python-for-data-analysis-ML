from django.shortcuts import render
from django.http import HttpResponse
from pymongo import MongoClient #
from sshtunnel import SSHTunnelForwarder #
import pprint #
import json
from bson.json_util import default
from bson.code import Code

import matplotlib.pyplot as plt
import base64
from io import BytesIO

MONGO_HOST = "devicimongodb063.westeurope.cloudapp.azure.com"
MONGO_DB = "countrybase"
MONGO_USER = "administrateur"
MONGO_PASS = "fcwP6h3H"

server = SSHTunnelForwarder(
	MONGO_HOST,
	ssh_username = MONGO_USER,
	ssh_password = MONGO_PASS,
	remote_bind_address = (MONGO_DB,22)
)

server.start()

client = MongoClient(MONGO_HOST, 30000)
db= client[MONGO_DB]
pprint.pprint(db.collection_names())
pipeline = {"_id.country_name" : "France"}


year_key = db.get_collection("Year_Key")
#pprint.pprint(list(year_key.find(pipeline)))

def home(request, *args, **kwargs):
	return render(request, "homepage.html", {})


def adminview(request, *args, **kwargs):
	return render(request, "admin.html",{})
	
def statistiques(request, *args, **kwargs):
	status = db.command({"dbStats":1});
	resultat = [] 
	RS1MX = "RS1MX : " + str(status["raw"]["RS1MX/devicimongodb054:27017"]["totalSize"])
	RS1YA = "RS1YA : " + str(status["raw"]["RS1YA/devicimongodb013:27017"]["totalSize"])
	RS1GU = "RS1GU : " + str(status["raw"]["RS1GU/devicimongodb065:27017"]["totalSize"])
	RS1LE = "RS1LE : " + str(status["raw"]["RS1LE/devicimongodb096:27017"]["totalSize"])
	RS2AD = "RS2AD : " + str(status["raw"]["RS2AD/devicimongodb164:27017"]["totalSize"])
	RS2LE = "RS2LE : " + str(status["raw"]["RS2LE/devicimongodb197:27017"]["totalSize"])
	RS2MX = "RS2MX : " + str(status["raw"]["RS2MX/devicimongodb155:27017"]["totalSize"])
	RS2YA = "RS2YA : " + str(status["raw"]["RS2YA/devicimongodb114:27017"]["totalSize"])

	resultat.append(RS1MX)
	resultat.append(RS1YA)
	resultat.append(RS1GU)
	resultat.append(RS1LE)
	resultat.append(RS2AD)
	resultat.append(RS2LE)
	resultat.append(RS2MX)
	resultat.append(RS2YA)
	
	datasize = []
	RS1MX = "RS1MX : " + str(status["raw"]["RS1MX/devicimongodb054:27017"]["dataSize"])
	RS1YA = "RS1YA : " + str(status["raw"]["RS1YA/devicimongodb013:27017"]["dataSize"])
	RS1GU = "RS1GU : " + str(status["raw"]["RS1GU/devicimongodb065:27017"]["dataSize"])
	RS1LE = "RS1LE : " + str(status["raw"]["RS1LE/devicimongodb096:27017"]["dataSize"])
	RS2AD = "RS2AD : " + str(status["raw"]["RS2AD/devicimongodb164:27017"]["dataSize"])
	RS2LE = "RS2LE : " + str(status["raw"]["RS2LE/devicimongodb197:27017"]["dataSize"])
	RS2MX = "RS2MX : " + str(status["raw"]["RS2MX/devicimongodb155:27017"]["dataSize"])
	RS2YA = "RS2YA : " + str(status["raw"]["RS2YA/devicimongodb114:27017"]["dataSize"])
	
	datasize.append(RS1MX)
	datasize.append(RS1YA)
	datasize.append(RS1GU)
	datasize.append(RS1LE)
	datasize.append(RS2AD)
	datasize.append(RS2LE)
	datasize.append(RS2MX)
	datasize.append(RS2YA)

	return render(request, "statistiques.html",{"data8": resultat, "datasize" : datasize})
	
def etat(request, *args, **kwargs):
	return render(request, "etat.html",{})
	
def repartition(request, *args, **kwargs):
	status = db.command({"dbStats":1});
	compteur = 0
	for shard in list(status["raw"]) :
		compteur = compteur +1
		
	compteur = "Nombre de shards utilisés : " + str(compteur)
	resultat = [] 
	RS1MX = "RS1MX : " + str(status["raw"]["RS1MX/devicimongodb054:27017"]["objects"])
	RS1YA = "RS1YA : " + str(status["raw"]["RS1YA/devicimongodb013:27017"]["objects"])
	RS1GU = "RS1GU : " + str(status["raw"]["RS1GU/devicimongodb065:27017"]["objects"])
	RS1LE = "RS1LE : " + str(status["raw"]["RS1LE/devicimongodb096:27017"]["objects"])
	RS2AD = "RS2AD : " + str(status["raw"]["RS2AD/devicimongodb164:27017"]["objects"])
	RS2LE = "RS2LE : " + str(status["raw"]["RS2LE/devicimongodb197:27017"]["objects"])
	RS2MX = "RS2MX : " + str(status["raw"]["RS2MX/devicimongodb155:27017"]["objects"])
	RS2YA = "RS2YA : " + str(status["raw"]["RS2YA/devicimongodb114:27017"]["objects"])

	resultat.append(RS1MX)
	resultat.append(RS1YA)
	resultat.append(RS1GU)
	resultat.append(RS1LE)
	resultat.append(RS2AD)
	resultat.append(RS2LE)
	resultat.append(RS2MX)
	resultat.append(RS2YA)
	return render(request, "repartition.html",{"data8": resultat, "count" : compteur})
	
def indexes(request, *args, **kwargs):
	status = db.command({"dbStats":1});
	resultat = [] 
	RS1MX = "RS1MX : " + str(status["raw"]["RS1MX/devicimongodb054:27017"]["indexes"])
	RS1YA = "RS1YA : " + str(status["raw"]["RS1YA/devicimongodb013:27017"]["indexes"])
	RS1GU = "RS1GU : " + str(status["raw"]["RS1GU/devicimongodb065:27017"]["indexes"])
	RS1LE = "RS1LE : " + str(status["raw"]["RS1LE/devicimongodb096:27017"]["indexes"])
	RS2AD = "RS2AD : " + str(status["raw"]["RS2AD/devicimongodb164:27017"]["indexes"])
	RS2LE = "RS2LE : " + str(status["raw"]["RS2LE/devicimongodb197:27017"]["indexes"])
	RS2MX = "RS2MX : " + str(status["raw"]["RS2MX/devicimongodb155:27017"]["indexes"])
	RS2YA = "RS2YA : " + str(status["raw"]["RS2YA/devicimongodb114:27017"]["indexes"])

	resultat.append(RS1MX)
	resultat.append(RS1YA)
	resultat.append(RS1GU)
	resultat.append(RS1LE)
	resultat.append(RS2AD)
	resultat.append(RS2LE)
	resultat.append(RS2MX)
	resultat.append(RS2YA)
	return render(request, "indexes.html",{"data8": resultat})
	
def analyste(request, *args, **kwargs):
	return render(request, "analyste.html",{"data" :list(year_key.find(pipeline))})
	
#pipeline1 = {"_id.year": "2020", "country_stats.life_expectancy" : {"$ne" : "null"} }
#pipeline2 = ("country_stats.life_expectancy", -1)	
	
def standard(request, *args, **kwargs):
	return render(request, "standard.html",{})
	
def R1(request, *args, **kwargs):
	r1= year_key.find({"_id.year": "2020", "country_stats.life_expectancy" : {"$ne" : "null"}  }, {"country_stats":0,  "_id.year":0}).sort("country_stats.life_expectancy", -1).limit(1)
	new_post=[]
	for doc in list(r1):
		liste = {}
		liste["result"]= "\nLe pays avec la plus grande espérance de vie est " + doc["_id"]["country_name"]
		new_post.append(liste)
	return render(request, "R1.html",{"data1" : new_post})
	
def R2(request, *args, **kwargs):
	r2= year_key.aggregate([{"$match" : {"_id.year" : "2020", "country_stats.country_area" : {"$ne" : "null"},  "country_stats.midyear_population" : {"$ne" : "null"}}},{"$project" : {"_id.country_name": 1, "pop_km2" : { "$divide": [{"$toDecimal": "$country_stats.midyear_population"}, {"$toDecimal":"$country_stats.country_area"}]}}}, {"$sort" : {"pop_km2":1}}, {"$limit" : 20}]);
	new_post=[]
	compteur =0
	for doc in list(r2):
		compteur= compteur + 1
		liste = {}
		liste["result"]= "\n" + str(compteur) + " : " + doc["_id"]["country_name"] + "     Population au km² : " + str(doc["pop_km2"])
		new_post.append(liste)
	return render(request, "R2.html",{"data2" : new_post})

	
def R3(request, *args, **kwargs):
	map = Code("""
				function(){
					if(this._id.year == "1955" || this._id.year == "1950")
						if( this.country_stats.crude_birth_rate != "null")
							emit(this._id.country_name, {"years":[{year : this._id.year, crude_birth_rate : this.country_stats.crude_birth_rate}], diff:null });
				}
				""")
	
	reduce = Code("""
					function(key, values) {
						y = {"years":[]}; 
						val55 = 0;
						val50 = 0;
						for(i =0 ; i<values.length;i++){
							for(j=0; j<values[i].years.length ; j++){
								if(values[i].years[j].year == "1950") val50 =  values[i].years[j].crude_birth_rate;
									if(values[i].years[j].year == "1955") val55 =  values[i].years[j].crude_birth_rate;
										y.years.push(values[i].years[j]);
							}
						}
						if(val55 > 0 && val50 > 0)
							y.diff = val55 - val50;
						else
							y.diff = null;
						return y;
					}
					""")
	mr_baby_boom = db.get_collection("map_reduce_baby_boom")

	result = year_key.map_reduce( map, reduce, "map_reduce_baby_boom" )
	new_post=[]
	compteur =0
	for doc in list(mr_baby_boom.find({"value.diff" : {"$gte" : 0.1}})):
		compteur= compteur + 1
		liste = {}
		liste["result"]= "\n" +  doc['_id']+ " : " + "Différence du nombre de naissances (1950-1955) : " + str(doc['value']['diff'])
		new_post.append(liste)
	return render(request, "R3.html",{"data3" : new_post})
	
def R4(request, *args, **kwargs):
	map = Code('''function(){
					if(this._id.country_name == "Iraq" || this._id.country_name == "United States" ) 
						if(parseInt(this._id.year) > 1999 & parseInt(this._id.year) < 2016)
							if(this.country_stats.crude_birth_rate != "null" && this.country_stats.crude_death_rate != "null")
								emit(this._id.country_name, {"years":[{year : this._id.year, crude_birth_rate : this.country_stats.crude_birth_rate, crude_death_rate : this.country_stats.crude_death_rate, diff: null}] });
				}; ''')
	reduce = Code('''function(key, values){
						y = {"years":[]};  
						for(i =0 ; i<values.length;i++)
							for(j=0; j<values[i].years.length ; j++){
								values[i].years[j].diff = values[i].years[j].crude_birth_rate - values[i].years[j].crude_death_rate;
								y.years.push(values[i].years[j])           
							}
						return y ;
					};''')
	result = year_key.map_reduce(map,reduce,"map_reduce_impact_irak" )
	new_post=[]
	for doc in list(result.find()):
		liste = {}
		liste["result"]= "\n  " + doc["_id"] 
		for year in doc["value"]["years"] :
			liste["result"] = liste["result"] +  "    année : " + str(year["year"]) +  "       Taux d'accroissement naturel : " + str(year["diff"]) 
		new_post.append(liste)
		
	return render(request, "R4.html",{"data4" : new_post})

def R5(request, *args, **kwargs):
	return render(request, "R5.html",{})

def R5_1(request, *args, **kwargs):
	map = Code('''function(){
					if((this._id.year =="2010" || this._id.year == "2000") && this.country_stats.crude_birth_rate!=null )
						emit(this._id.country_name, {"years":[{year : this._id.year, crude_birth_rate : this.country_stats.crude_birth_rate}], diff:null});
				}; ''')
	reduce = Code('''function(keyCountry, values){
						y = {"years":[]}; 
						val55 = 0;
						val50 = 0;
						for(i =0 ; i<values.length;i++)
							for(j=0; j<values[i].years.length ; j++){
								if(values[i].years[j].year == "2010") val55 =  values[i].years[j].crude_birth_rate;
								if(values[i].years[j].year == "2000") val50 =  values[i].years[j].crude_birth_rate;
								y.years.push(values[i].years[j]);
							}
						if(val55 > 0 && val50 > 0)
							y.diff = val55 - val50;
						else
							y.diff = null;
						return y;
						}''')
	result = year_key.map_reduce(map,reduce,"map_reduce_max_increase" )
	new_post=[]
	for doc in list(result.find().sort("value.diff",-1).limit(1)):
		liste = {}
		liste["result"]= "\n  " + doc["_id"]  + "    différence de taux : " + str(doc["value"]["diff"]) + " |  "
		for year in doc["value"]["years"] :
			liste["result"] = liste["result"] +  "    année : " + str(year["year"]) +  "       Taux de naissance : " + str(year["crude_birth_rate"]) 
		new_post.append(liste)
		
	return render(request, "R5_1.html",{"data5" : new_post})

def R5_2(request, *args, **kwargs):
	map = Code('''function(){
					if((this._id.year =="2020" || this._id.year == "2010") && this.country_stats.crude_birth_rate!=null )
						emit(this._id.country_name, {"years":[{year : this._id.year, crude_birth_rate : this.country_stats.crude_birth_rate}], diff:null});
				}; ''')
	reduce = Code('''function(keyCountry, values){
						y = {"years":[]}; 
						val55 = 0;
						val50 = 0;
						for(i =0 ; i<values.length;i++)
							for(j=0; j<values[i].years.length ; j++){
								if(values[i].years[j].year == "2020") val55 =  values[i].years[j].crude_birth_rate;
								if(values[i].years[j].year == "2010") val50 =  values[i].years[j].crude_birth_rate;
								y.years.push(values[i].years[j]);
							}
						if(val55 > 0 && val50 > 0)
							y.diff = val55 - val50;
						else
							y.diff = null;
						return y;
						}''')
	result = year_key.map_reduce(map,reduce,"map_reduce_max_increase" )
	new_post=[]
	for doc in list(result.find().sort("value.diff",-1).limit(1)):
		liste = {}
		liste["result"]= "\n  " + doc["_id"]  + "    différence de taux : " + str(doc["value"]["diff"]) + " |  "
		for year in doc["value"]["years"] :
			liste["result"] = liste["result"] +  "    année : " + str(year["year"]) +  "       Taux de naissance : " + str(year["crude_birth_rate"]) 
		new_post.append(liste)
		
	return render(request, "R5_2.html",{"data5" : new_post})

def R5_3(request, *args, **kwargs):
	map = Code('''function(){
					if((this._id.year =="2020" || this._id.year == "2000") && this.country_stats.crude_birth_rate!=null )
						emit(this._id.country_name, {"years":[{year : this._id.year, crude_birth_rate : this.country_stats.crude_birth_rate}], diff:null});
				}; ''')
	reduce = Code('''function(keyCountry, values){
						y = {"years":[]}; 
						val55 = 0;
						val50 = 0;
						for(i =0 ; i<values.length;i++)
							for(j=0; j<values[i].years.length ; j++){
								if(values[i].years[j].year == "2020") val55 =  values[i].years[j].crude_birth_rate;
								if(values[i].years[j].year == "2000") val50 =  values[i].years[j].crude_birth_rate;
								y.years.push(values[i].years[j]);
							}
						if(val55 > 0 && val50 > 0)
							y.diff = val55 - val50;
						else
							y.diff = null;
						return y;
						}''')
	result = year_key.map_reduce(map,reduce,"map_reduce_max_increase" )
	new_post=[]
	for doc in list(result.find().sort("value.diff",-1).limit(1)):
		liste = {}
		liste["result"]= "\n  " + doc["_id"]  + "    différence de taux : " + str(doc["value"]["diff"]) + " |  "
		for year in doc["value"]["years"] :
			liste["result"] = liste["result"] +  "    année : " + str(year["year"]) +  "       Taux de naissance : " + str(year["crude_birth_rate"]) 
		new_post.append(liste)
		
	return render(request, "R5_3.html",{"data5" : new_post})
	
def R6(request, *args, **kwargs):
	return render(request, "R6.html",{})

def R6_1(request, *args, **kwargs):
	map = Code('''function(){
					if(parseInt(this._id.year) > 2010 && parseInt(this._id.year) < 2021) 
						if (this.country_stats.rate_natural_increase != "null")
							emit(this._id.country_name, {"years":[{year : this._id.year, natural_rate : parseFloat(this.country_stats.rate_natural_increase), population : this.country_stats.midyear_population}], avg_evolution:null, pop_decroiss : null});
				}; ''')
	reduce = Code('''function(keyCountry, values){
						y = {"years":[]}; 
						val19 = 0;
						val20 = 0;
						sum_natural_rate = 0;
						count = 0;
						for(i =0 ; i<values.length;i++)
							for(j=0; j<values[i].years.length ; j++){
								if(values[i].years[j].year == "2019") val19 =  values[i].years[j].population;
								if(values[i].years[j].year == "2020") val20 =  values[i].years[j].population;
								if(values[i].years[j].natural_rate != null)
									sum_natural_rate += values[i].years[j].natural_rate;
									count ++;
								y.years.push(values[i].years[j]);
							}
    
    
						if(sum_natural_rate != 0)
							y.avg_evolution = sum_natural_rate/count;
						else
							y.avg_evolution = null;
						if(val19 > 0 && val20 > 0)
							if( val20 - val19 < 0)
								y.pop_decroiss = "Décroissante";
							else
								y.pop_decroiss = "Croissante";
						else
							y.pop_decroiss = null;
						return y;
						}''')
	result = year_key.map_reduce(map,reduce,"map_reduce_pop_viel_decrois" )
	new_post=[]
	for doc in list(result.find({"value.avg_evolution" : {"$ne" : "null"}}).sort("value.avg_evolution", 1).limit(10)):
		liste = {}
		liste["result"]= "\n  " + doc["_id"]  
		for year in doc["value"]["years"] :
			liste["result"] = liste["result"]
		new_post.append(liste)
		
	return render(request, "R6_1.html",{"data6" : new_post})

def R6_2(request, *args, **kwargs):
	map = Code('''function(){
					if(parseInt(this._id.year) > 2010 && parseInt(this._id.year) < 2021) 
						if (this.country_stats.rate_natural_increase != "null")
							emit(this._id.country_name, {"years":[{year : this._id.year, natural_rate : parseFloat(this.country_stats.rate_natural_increase), population : this.country_stats.midyear_population}], avg_evolution:null, pop_decroiss : null});
				}; ''')
	reduce = Code('''function(keyCountry, values){
						y = {"years":[]}; 
						val19 = 0;
						val20 = 0;
						sum_natural_rate = 0;
						count = 0;
						for(i =0 ; i<values.length;i++)
							for(j=0; j<values[i].years.length ; j++){
								if(values[i].years[j].year == "2019") val19 =  values[i].years[j].population;
								if(values[i].years[j].year == "2020") val20 =  values[i].years[j].population;
								if(values[i].years[j].natural_rate != null)
									sum_natural_rate += values[i].years[j].natural_rate;
									count ++;
								y.years.push(values[i].years[j]);
							}
    
    
						if(sum_natural_rate != 0)
							y.avg_evolution = sum_natural_rate/count;
						else
							y.avg_evolution = null;
						if(val19 > 0 && val20 > 0)
							if( val20 - val19 < 0)
								y.pop_decroiss = "Décroissante";
							else
								y.pop_decroiss = "Croissante";
						else
							y.pop_decroiss = null;
						return y;
						}''')
	result = year_key.map_reduce(map,reduce,"map_reduce_pop_viel_decrois" )
	new_post=[]
	for doc in list(result.find({"value.avg_evolution" : {"$ne" : "null"}}).sort("value.avg_evolution", 1).limit(20)):
		liste = {}
		liste["result"]= "\n  " + doc["_id"]  
		new_post.append(liste)
		
	return render(request, "R6_2.html",{"data6" : new_post})

def R6_3(request, *args, **kwargs):
	map = Code('''function(){
					if(parseInt(this._id.year) > 2010 && parseInt(this._id.year) < 2021) 
						if (this.country_stats.rate_natural_increase != "null")
							emit(this._id.country_name, {"years":[{year : this._id.year, natural_rate : parseFloat(this.country_stats.rate_natural_increase), population : this.country_stats.midyear_population}], avg_evolution:null, pop_decroiss : null});
				}; ''')
	reduce = Code('''function(keyCountry, values){
						y = {"years":[]}; 
						val19 = 0;
						val20 = 0;
						sum_natural_rate = 0;
						count = 0;
						for(i =0 ; i<values.length;i++)
							for(j=0; j<values[i].years.length ; j++){
								if(values[i].years[j].year == "2019") val19 =  values[i].years[j].population;
								if(values[i].years[j].year == "2020") val20 =  values[i].years[j].population;
								if(values[i].years[j].natural_rate != null)
									sum_natural_rate += values[i].years[j].natural_rate;
									count ++;
								y.years.push(values[i].years[j]);
							}
    
    
						if(sum_natural_rate != 0)
							y.avg_evolution = sum_natural_rate/count;
						else
							y.avg_evolution = null;
						if(val19 > 0 && val20 > 0)
							if( val20 - val19 < 0)
								y.pop_decroiss = "Décroissante";
							else
								y.pop_decroiss = "Croissante";
						else
							y.pop_decroiss = null;
						return y;
						}''')
	result = year_key.map_reduce(map,reduce,"map_reduce_pop_viel_decrois" )
	new_post=[]
	for doc in list(result.find({"value.avg_evolution" : {"$ne" : "null"}}).sort("value.avg_evolution", 1).limit(50)):
		liste = {}
		liste["result"]= "\n  " + doc["_id"] 
		new_post.append(liste)
		
	return render(request, "R6_3.html",{"data6" : new_post})
	
def R7(request, *args, **kwargs):
	return render(request, "R7.html",{})

def get_graph():
	buffer = BytesIO()
	plt.savefig(buffer, format='png')
	buffer.seek(0)
	image_png = buffer.getvalue()
	graph = base64.b64encode(image_png)
	graph = graph.decode('utf-8')
	buffer.close()
	return graph

def get_plot(x,y, taux):
	plt.switch_backend('AGG')
	plt.figure(figsize=(10,5))
	title = 'Nombre de pays ayant un taux supérieur à ' + str(taux) + " pour 1000 naissances"
	plt.title(title)
	plt.plot(x,y)
	plt.xticks(rotation=90, fontsize = 6) 
	plt.xlabel('Années')
	plt.ylabel('Nombre de pays')
	plt.tight_layout()
	
	graph = get_graph()
	return graph
	
def R7_1(request, *args, **kwargs):
	map = Code('''function(){
					if(this.country_stats.mortality_rate_under5 != "null" )
						emit(this._id.year, {"countries" :[{country : this._id.country_name, mru :this.country_stats.Male.infant_mortality_male }] ,count:null});
				}; ''')
	reduce = Code('''function(keyCountry, values){
						y = {"countries":[]}
						var country_count = 0
						for(i =0; i< values.length; i++){
							if(values[i].countries !=null) {
								for(j=0; j<values[i].countries.length ; j++){
									if(parseFloat(values[i].countries[j].mru) >80.0) {
										country_count = country_count +1
										y.countries.push(values[i].countries[j])
									}
								}
							}
						}
						y.count = country_count    
						return y;
						}''')
	result = year_key.map_reduce(map,reduce,"map_reduce_max_increase" )
	new_post=[]
	y = []
	x= []
	for doc in list(result.find().sort("_id", 1)):
		liste = {}
		x.append(doc["_id"])
		
		liste["result"]= "\n "+  doc["_id"] +  "   " + str(doc["value"]["count"])
		for country in doc["value"]["countries"] :
			liste["result"] = liste["result"] +  "   " +  country["country"] 
		liste["result"] = liste["result"] + "" 
		y.append(doc["value"]["count"])
		new_post.append(liste)
	chart = get_plot(x,y, 80)	
	return render(request, "R7_1.html",{"data7" : new_post, "chart" : chart})
	
def R7_2(request, *args, **kwargs):
	map = Code('''function(){
					if(this.country_stats.mortality_rate_under5 != "null" )
						emit(this._id.year, {"countries" :[{country : this._id.country_name, mru :this.country_stats.Male.infant_mortality_male }] ,count:null});
				}; ''')
	reduce = Code('''function(keyCountry, values){
						y = {"countries":[]}
						var country_count = 0
						for(i =0; i< values.length; i++){
							if(values[i].countries !=null) {
								for(j=0; j<values[i].countries.length ; j++){
									if(parseFloat(values[i].countries[j].mru) >100.0) {
										country_count = country_count +1
										y.countries.push(values[i].countries[j])
									}
								}
							}
						}
						y.count = country_count    
						return y;
						}''')
	result = year_key.map_reduce(map,reduce,"map_reduce_max_increase" )
	new_post=[]
	y = []
	x= []
	for doc in list(result.find().sort("_id", 1)):
		liste = {}
		x.append(doc["_id"])
		
		liste["result"]= "\n "+  doc["_id"] +  "   " + str(doc["value"]["count"])
		for country in doc["value"]["countries"] :
			liste["result"] = liste["result"] +  "   " +  country["country"] 
		liste["result"] = liste["result"] + "" 
		y.append(doc["value"]["count"])
		new_post.append(liste)
	chart = get_plot(x,y, 80)	
	return render(request, "R7_2.html",{"data7" : new_post, "chart" : chart})

def R7_3(request, *args, **kwargs):
	map = Code('''function(){
					if(this.country_stats.mortality_rate_under5 != "null" )
						emit(this._id.year, {"countries" :[{country : this._id.country_name, mru :this.country_stats.Male.infant_mortality_male }] ,count:null});
				}; ''')
	reduce = Code('''function(keyCountry, values){
						y = {"countries":[]}
						var country_count = 0
						for(i =0; i< values.length; i++){
							if(values[i].countries !=null) {
								for(j=0; j<values[i].countries.length ; j++){
									if(parseFloat(values[i].countries[j].mru) >150.0) {
										country_count = country_count +1
										y.countries.push(values[i].countries[j])
									}
								}
							}
						}
						y.count = country_count    
						return y;
						}''')
	result = year_key.map_reduce(map,reduce,"map_reduce_max_increase" )
	new_post=[]
	y = []
	x= []
	for doc in list(result.find().sort("_id", 1)):
		liste = {}
		x.append(doc["_id"])
		
		liste["result"]= "\n "+  doc["_id"] +  "   " + str(doc["value"]["count"])
		for country in doc["value"]["countries"] :
			liste["result"] = liste["result"] +  "   " +  country["country"] 
		liste["result"] = liste["result"] + "" 
		y.append(doc["value"]["count"])
		new_post.append(liste)
	chart = get_plot(x,y, 80)	
	return render(request, "R7_3.html",{"data7" : new_post, "chart" : chart})
	
def R8(request, *args, **kwargs):
	return render(request, "R8.html",{})

def R8_1(request, *args, **kwargs):
	map = Code('''function(){
					if(this._id.country_name == "France")    
						emit(this._id.year, {"populations":[{male_pop : this.country_stats.Male, female_pop : this.country_stats.Female }],max: null}); 
				}; ''')
	reduce = Code('''function(keyCountry, values){
						y = {"populations":[]}
    index = [];
    for(i =0; i< values.length; i++)
    {
        if(values[i].populations !=null)
         {
            for(j=0; j<values[i].populations.length ; j++)
            {
                if (values[i].populations[j] !=null)
               {      max = 0.0;
                      if (values[i].populations[j].male_pop != null & values[i].populations[j].female_pop != null)
                     { 
                      y.populations.push(values[i].populations[j]);
                      index.push(parseFloat(values[i].populations[j].male_pop.population_age_0) + parseFloat(values[i].populations[j].female_pop.population_age_0))
                      index.push(parseFloat(values[i].populations[j].male_pop.population_age_1) + parseFloat(values[i].populations[j].female_pop.population_age_1))
                      index.push(parseFloat(values[i].populations[j].male_pop.population_age_2) + parseFloat(values[i].populations[j].female_pop.population_age_2))
                      index.push(parseFloat(values[i].populations[j].male_pop.population_age_3) + parseFloat(values[i].populations[j].female_pop.population_age_3))
                      index.push(parseFloat(values[i].populations[j].male_pop.population_age_4) + parseFloat(values[i].populations[j].female_pop.population_age_4))
                      index.push(parseFloat(values[i].populations[j].male_pop.population_age_5) + parseFloat(values[i].populations[j].female_pop.population_age_5))
                      index.push(parseFloat(values[i].populations[j].male_pop.population_age_6) + parseFloat(values[i].populations[j].female_pop.population_age_6))
                      index.push(parseFloat(values[i].populations[j].male_pop.population_age_7) + parseFloat(values[i].populations[j].female_pop.population_age_7))
                      index.push(parseFloat(values[i].populations[j].male_pop.population_age_8) + parseFloat(values[i].populations[j].female_pop.population_age_8))
                      index.push(parseFloat(values[i].populations[j].male_pop.population_age_9) + parseFloat(values[i].populations[j].female_pop.population_age_9))
                      index.push(parseFloat(values[i].populations[j].male_pop.population_age_10) + parseFloat(values[i].populations[j].female_pop.population_age_10))
                      index.push(parseFloat(values[i].populations[j].male_pop.population_age_11) + parseFloat(values[i].populations[j].female_pop.population_age_11))
                      index.push(parseFloat(values[i].populations[j].male_pop.population_age_12) + parseFloat(values[i].populations[j].female_pop.population_age_12))
                      index.push(parseFloat(values[i].populations[j].male_pop.population_age_13) + parseFloat(values[i].populations[j].female_pop.population_age_13))
                      index.push(parseFloat(values[i].populations[j].male_pop.population_age_14) + parseFloat(values[i].populations[j].female_pop.population_age_14))
                      index.push(parseFloat(values[i].populations[j].male_pop.population_age_15) + parseFloat(values[i].populations[j].female_pop.population_age_15))                      
                      index.push(parseFloat(values[i].populations[j].male_pop.population_age_16) + parseFloat(values[i].populations[j].female_pop.population_age_16))
                      index.push(parseFloat(values[i].populations[j].male_pop.population_age_17) + parseFloat(values[i].populations[j].female_pop.population_age_17))
                      index.push(parseFloat(values[i].populations[j].male_pop.population_age_18) + parseFloat(values[i].populations[j].female_pop.population_age_18))
                      index.push(parseFloat(values[i].populations[j].male_pop.population_age_19) + parseFloat(values[i].populations[j].female_pop.population_age_19))
                      index.push(parseFloat(values[i].populations[j].male_pop.population_age_20) + parseFloat(values[i].populations[j].female_pop.population_age_20)) 
                      index.push(parseFloat(values[i].populations[j].male_pop.population_age_21) + parseFloat(values[i].populations[j].female_pop.population_age_21)) 
                      index.push(parseFloat(values[i].populations[j].male_pop.population_age_22) + parseFloat(values[i].populations[j].female_pop.population_age_22)) 
                      index.push(parseFloat(values[i].populations[j].male_pop.population_age_23) + parseFloat(values[i].populations[j].female_pop.population_age_23)) 
                      index.push(parseFloat(values[i].populations[j].male_pop.population_age_24) + parseFloat(values[i].populations[j].female_pop.population_age_24)) 
                      index.push(parseFloat(values[i].populations[j].male_pop.population_age_25) + parseFloat(values[i].populations[j].female_pop.population_age_25)) 
                      index.push(parseFloat(values[i].populations[j].male_pop.population_age_26) + parseFloat(values[i].populations[j].female_pop.population_age_26)) 
                      index.push(parseFloat(values[i].populations[j].male_pop.population_age_27) + parseFloat(values[i].populations[j].female_pop.population_age_27)) 
                      index.push(parseFloat(values[i].populations[j].male_pop.population_age_28) + parseFloat(values[i].populations[j].female_pop.population_age_28)) 
                      index.push(parseFloat(values[i].populations[j].male_pop.population_age_29) + parseFloat(values[i].populations[j].female_pop.population_age_29)) 
                      index.push(parseFloat(values[i].populations[j].male_pop.population_age_30) + parseFloat(values[i].populations[j].female_pop.population_age_30)) 
                      index.push(parseFloat(values[i].populations[j].male_pop.population_age_31) + parseFloat(values[i].populations[j].female_pop.population_age_31)) 
                      index.push(parseFloat(values[i].populations[j].male_pop.population_age_32) + parseFloat(values[i].populations[j].female_pop.population_age_32)) 
                      index.push(parseFloat(values[i].populations[j].male_pop.population_age_33) + parseFloat(values[i].populations[j].female_pop.population_age_33)) 
                      index.push(parseFloat(values[i].populations[j].male_pop.population_age_34) + parseFloat(values[i].populations[j].female_pop.population_age_34)) 
                      index.push(parseFloat(values[i].populations[j].male_pop.population_age_35) + parseFloat(values[i].populations[j].female_pop.population_age_35)) 
                      index.push(parseFloat(values[i].populations[j].male_pop.population_age_36) + parseFloat(values[i].populations[j].female_pop.population_age_36)) 
                      index.push(parseFloat(values[i].populations[j].male_pop.population_age_37) + parseFloat(values[i].populations[j].female_pop.population_age_37)) 
                      index.push(parseFloat(values[i].populations[j].male_pop.population_age_38) + parseFloat(values[i].populations[j].female_pop.population_age_38)) 
                      index.push(parseFloat(values[i].populations[j].male_pop.population_age_39) + parseFloat(values[i].populations[j].female_pop.population_age_39)) 
                      index.push(parseFloat(values[i].populations[j].male_pop.population_age_40) + parseFloat(values[i].populations[j].female_pop.population_age_40)) 
                      index.push(parseFloat(values[i].populations[j].male_pop.population_age_41) + parseFloat(values[i].populations[j].female_pop.population_age_41)) 
                      index.push(parseFloat(values[i].populations[j].male_pop.population_age_42) + parseFloat(values[i].populations[j].female_pop.population_age_42)) 
                      index.push(parseFloat(values[i].populations[j].male_pop.population_age_43) + parseFloat(values[i].populations[j].female_pop.population_age_43)) 
                      index.push(parseFloat(values[i].populations[j].male_pop.population_age_44) + parseFloat(values[i].populations[j].female_pop.population_age_44)) 
                      index.push(parseFloat(values[i].populations[j].male_pop.population_age_45) + parseFloat(values[i].populations[j].female_pop.population_age_45)) 
                      index.push(parseFloat(values[i].populations[j].male_pop.population_age_46) + parseFloat(values[i].populations[j].female_pop.population_age_46)) 
                      index.push(parseFloat(values[i].populations[j].male_pop.population_age_47) + parseFloat(values[i].populations[j].female_pop.population_age_47)) 
                      index.push(parseFloat(values[i].populations[j].male_pop.population_age_48) + parseFloat(values[i].populations[j].female_pop.population_age_48)) 
                      index.push(parseFloat(values[i].populations[j].male_pop.population_age_49) + parseFloat(values[i].populations[j].female_pop.population_age_49)) 
                      index.push(parseFloat(values[i].populations[j].male_pop.population_age_50) + parseFloat(values[i].populations[j].female_pop.population_age_50)) 
                      index.push(parseFloat(values[i].populations[j].male_pop.population_age_51) + parseFloat(values[i].populations[j].female_pop.population_age_51)) 
                      index.push(parseFloat(values[i].populations[j].male_pop.population_age_52) + parseFloat(values[i].populations[j].female_pop.population_age_52)) 
                      index.push(parseFloat(values[i].populations[j].male_pop.population_age_53) + parseFloat(values[i].populations[j].female_pop.population_age_53)) 
                      index.push(parseFloat(values[i].populations[j].male_pop.population_age_54) + parseFloat(values[i].populations[j].female_pop.population_age_54)) 
                      index.push(parseFloat(values[i].populations[j].male_pop.population_age_55) + parseFloat(values[i].populations[j].female_pop.population_age_55)) 
                      index.push(parseFloat(values[i].populations[j].male_pop.population_age_56) + parseFloat(values[i].populations[j].female_pop.population_age_56)) 
                      index.push(parseFloat(values[i].populations[j].male_pop.population_age_57) + parseFloat(values[i].populations[j].female_pop.population_age_57)) 
                      index.push(parseFloat(values[i].populations[j].male_pop.population_age_58) + parseFloat(values[i].populations[j].female_pop.population_age_58)) 
                      index.push(parseFloat(values[i].populations[j].male_pop.population_age_59) + parseFloat(values[i].populations[j].female_pop.population_age_59)) 
                      index.push(parseFloat(values[i].populations[j].male_pop.population_age_60) + parseFloat(values[i].populations[j].female_pop.population_age_60)) 
                      index.push(parseFloat(values[i].populations[j].male_pop.population_age_61) + parseFloat(values[i].populations[j].female_pop.population_age_61)) 
                      index.push(parseFloat(values[i].populations[j].male_pop.population_age_62) + parseFloat(values[i].populations[j].female_pop.population_age_62)) 
                      index.push(parseFloat(values[i].populations[j].male_pop.population_age_63) + parseFloat(values[i].populations[j].female_pop.population_age_63)) 
                      index.push(parseFloat(values[i].populations[j].male_pop.population_age_64) + parseFloat(values[i].populations[j].female_pop.population_age_64)) 
                      index.push(parseFloat(values[i].populations[j].male_pop.population_age_65) + parseFloat(values[i].populations[j].female_pop.population_age_65)) 
                      index.push(parseFloat(values[i].populations[j].male_pop.population_age_66) + parseFloat(values[i].populations[j].female_pop.population_age_66)) 
                      index.push(parseFloat(values[i].populations[j].male_pop.population_age_67) + parseFloat(values[i].populations[j].female_pop.population_age_67)) 
                      index.push(parseFloat(values[i].populations[j].male_pop.population_age_68) + parseFloat(values[i].populations[j].female_pop.population_age_68)) 
                      index.push(parseFloat(values[i].populations[j].male_pop.population_age_69) + parseFloat(values[i].populations[j].female_pop.population_age_69)) 
                      index.push(parseFloat(values[i].populations[j].male_pop.population_age_70) + parseFloat(values[i].populations[j].female_pop.population_age_70)) 
                      index.push(parseFloat(values[i].populations[j].male_pop.population_age_71) + parseFloat(values[i].populations[j].female_pop.population_age_71)) 
                      index.push(parseFloat(values[i].populations[j].male_pop.population_age_72) + parseFloat(values[i].populations[j].female_pop.population_age_72)) 
                      index.push(parseFloat(values[i].populations[j].male_pop.population_age_73) + parseFloat(values[i].populations[j].female_pop.population_age_73)) 
                      index.push(parseFloat(values[i].populations[j].male_pop.population_age_74) + parseFloat(values[i].populations[j].female_pop.population_age_74)) 
                      index.push(parseFloat(values[i].populations[j].male_pop.population_age_75) + parseFloat(values[i].populations[j].female_pop.population_age_75)) 
                      index.push(parseFloat(values[i].populations[j].male_pop.population_age_76) + parseFloat(values[i].populations[j].female_pop.population_age_76)) 
                      index.push(parseFloat(values[i].populations[j].male_pop.population_age_77) + parseFloat(values[i].populations[j].female_pop.population_age_77)) 
                      index.push(parseFloat(values[i].populations[j].male_pop.population_age_78) + parseFloat(values[i].populations[j].female_pop.population_age_78)) 
                      index.push(parseFloat(values[i].populations[j].male_pop.population_age_79) + parseFloat(values[i].populations[j].female_pop.population_age_79)) 
                      index.push(parseFloat(values[i].populations[j].male_pop.population_age_80) + parseFloat(values[i].populations[j].female_pop.population_age_80)) 
                      index.push(parseFloat(values[i].populations[j].male_pop.population_age_81) + parseFloat(values[i].populations[j].female_pop.population_age_81)) 
                      index.push(parseFloat(values[i].populations[j].male_pop.population_age_82) + parseFloat(values[i].populations[j].female_pop.population_age_82)) 
                      index.push(parseFloat(values[i].populations[j].male_pop.population_age_83) + parseFloat(values[i].populations[j].female_pop.population_age_83)) 
                      index.push(parseFloat(values[i].populations[j].male_pop.population_age_84) + parseFloat(values[i].populations[j].female_pop.population_age_84)) 
                      index.push(parseFloat(values[i].populations[j].male_pop.population_age_85) + parseFloat(values[i].populations[j].female_pop.population_age_85)) 
                      index.push(parseFloat(values[i].populations[j].male_pop.population_age_86) + parseFloat(values[i].populations[j].female_pop.population_age_86)) 
                      index.push(parseFloat(values[i].populations[j].male_pop.population_age_87) + parseFloat(values[i].populations[j].female_pop.population_age_87)) 
                      index.push(parseFloat(values[i].populations[j].male_pop.population_age_88) + parseFloat(values[i].populations[j].female_pop.population_age_88)) 
                      index.push(parseFloat(values[i].populations[j].male_pop.population_age_89) + parseFloat(values[i].populations[j].female_pop.population_age_89)) 
                      index.push(parseFloat(values[i].populations[j].male_pop.population_age_90) + parseFloat(values[i].populations[j].female_pop.population_age_90)) 
                      index.push(parseFloat(values[i].populations[j].male_pop.population_age_91) + parseFloat(values[i].populations[j].female_pop.population_age_91)) 
                      index.push(parseFloat(values[i].populations[j].male_pop.population_age_92) + parseFloat(values[i].populations[j].female_pop.population_age_92)) 
                      index.push(parseFloat(values[i].populations[j].male_pop.population_age_93) + parseFloat(values[i].populations[j].female_pop.population_age_93)) 
                      index.push(parseFloat(values[i].populations[j].male_pop.population_age_94) + parseFloat(values[i].populations[j].female_pop.population_age_94)) 
                      index.push(parseFloat(values[i].populations[j].male_pop.population_age_95) + parseFloat(values[i].populations[j].female_pop.population_age_95)) 
                      index.push(parseFloat(values[i].populations[j].male_pop.population_age_96) + parseFloat(values[i].populations[j].female_pop.population_age_96)) 
                      index.push(parseFloat(values[i].populations[j].male_pop.population_age_97) + parseFloat(values[i].populations[j].female_pop.population_age_97)) 
                      index.push(parseFloat(values[i].populations[j].male_pop.population_age_98) + parseFloat(values[i].populations[j].female_pop.population_age_98)) 
                      index.push(parseFloat(values[i].populations[j].male_pop.population_age_99) + parseFloat(values[i].populations[j].female_pop.population_age_99)) 
                      index.push(parseFloat(values[i].populations[j].male_pop.population_age_100) + parseFloat(values[i].populations[j].female_pop.population_age_100))
                      
                      
                     }
               }
           }
                
         }
     }
     var i = index.indexOf(Math.max(...index));
    
    y.max = i;
    
    return y;
						}''')
	result = year_key.map_reduce(map,reduce,"map_reduce_fr_most_represented_age_group" )
	new_post=[]
	for doc in list(result.find({"value.max" : {"$ne" : -1.0}}).sort("_id",1)):
		liste = {}
		liste["result"]= "\n  " + doc["_id"]  + "    Catégorie d'âge la plus représentée : " + str(doc["value"]["max"])
		new_post.append(liste)
		
	return render(request, "R8_1.html",{"data8" : new_post})
	
def R8_2(request, *args, **kwargs):
	map = Code('''function(){
					if(this._id.country_name == "Pakistan")    
						emit(this._id.year, {"populations":[{male_pop : this.country_stats.Male, female_pop : this.country_stats.Female }],max: null}); 
				}; ''')
	reduce = Code('''function(keyCountry, values){
						y = {"populations":[]}
    index = [];
    for(i =0; i< values.length; i++)
    {
        if(values[i].populations !=null)
         {
            for(j=0; j<values[i].populations.length ; j++)
            {
                if (values[i].populations[j] !=null)
               {      max = 0.0;
                      if (values[i].populations[j].male_pop != null & values[i].populations[j].female_pop != null)
                     { 
                      y.populations.push(values[i].populations[j]);
                      index.push(parseFloat(values[i].populations[j].male_pop.population_age_0) + parseFloat(values[i].populations[j].female_pop.population_age_0))
                      index.push(parseFloat(values[i].populations[j].male_pop.population_age_1) + parseFloat(values[i].populations[j].female_pop.population_age_1))
                      index.push(parseFloat(values[i].populations[j].male_pop.population_age_2) + parseFloat(values[i].populations[j].female_pop.population_age_2))
                      index.push(parseFloat(values[i].populations[j].male_pop.population_age_3) + parseFloat(values[i].populations[j].female_pop.population_age_3))
                      index.push(parseFloat(values[i].populations[j].male_pop.population_age_4) + parseFloat(values[i].populations[j].female_pop.population_age_4))
                      index.push(parseFloat(values[i].populations[j].male_pop.population_age_5) + parseFloat(values[i].populations[j].female_pop.population_age_5))
                      index.push(parseFloat(values[i].populations[j].male_pop.population_age_6) + parseFloat(values[i].populations[j].female_pop.population_age_6))
                      index.push(parseFloat(values[i].populations[j].male_pop.population_age_7) + parseFloat(values[i].populations[j].female_pop.population_age_7))
                      index.push(parseFloat(values[i].populations[j].male_pop.population_age_8) + parseFloat(values[i].populations[j].female_pop.population_age_8))
                      index.push(parseFloat(values[i].populations[j].male_pop.population_age_9) + parseFloat(values[i].populations[j].female_pop.population_age_9))
                      index.push(parseFloat(values[i].populations[j].male_pop.population_age_10) + parseFloat(values[i].populations[j].female_pop.population_age_10))
                      index.push(parseFloat(values[i].populations[j].male_pop.population_age_11) + parseFloat(values[i].populations[j].female_pop.population_age_11))
                      index.push(parseFloat(values[i].populations[j].male_pop.population_age_12) + parseFloat(values[i].populations[j].female_pop.population_age_12))
                      index.push(parseFloat(values[i].populations[j].male_pop.population_age_13) + parseFloat(values[i].populations[j].female_pop.population_age_13))
                      index.push(parseFloat(values[i].populations[j].male_pop.population_age_14) + parseFloat(values[i].populations[j].female_pop.population_age_14))
                      index.push(parseFloat(values[i].populations[j].male_pop.population_age_15) + parseFloat(values[i].populations[j].female_pop.population_age_15))                      
                      index.push(parseFloat(values[i].populations[j].male_pop.population_age_16) + parseFloat(values[i].populations[j].female_pop.population_age_16))
                      index.push(parseFloat(values[i].populations[j].male_pop.population_age_17) + parseFloat(values[i].populations[j].female_pop.population_age_17))
                      index.push(parseFloat(values[i].populations[j].male_pop.population_age_18) + parseFloat(values[i].populations[j].female_pop.population_age_18))
                      index.push(parseFloat(values[i].populations[j].male_pop.population_age_19) + parseFloat(values[i].populations[j].female_pop.population_age_19))
                      index.push(parseFloat(values[i].populations[j].male_pop.population_age_20) + parseFloat(values[i].populations[j].female_pop.population_age_20)) 
                      index.push(parseFloat(values[i].populations[j].male_pop.population_age_21) + parseFloat(values[i].populations[j].female_pop.population_age_21)) 
                      index.push(parseFloat(values[i].populations[j].male_pop.population_age_22) + parseFloat(values[i].populations[j].female_pop.population_age_22)) 
                      index.push(parseFloat(values[i].populations[j].male_pop.population_age_23) + parseFloat(values[i].populations[j].female_pop.population_age_23)) 
                      index.push(parseFloat(values[i].populations[j].male_pop.population_age_24) + parseFloat(values[i].populations[j].female_pop.population_age_24)) 
                      index.push(parseFloat(values[i].populations[j].male_pop.population_age_25) + parseFloat(values[i].populations[j].female_pop.population_age_25)) 
                      index.push(parseFloat(values[i].populations[j].male_pop.population_age_26) + parseFloat(values[i].populations[j].female_pop.population_age_26)) 
                      index.push(parseFloat(values[i].populations[j].male_pop.population_age_27) + parseFloat(values[i].populations[j].female_pop.population_age_27)) 
                      index.push(parseFloat(values[i].populations[j].male_pop.population_age_28) + parseFloat(values[i].populations[j].female_pop.population_age_28)) 
                      index.push(parseFloat(values[i].populations[j].male_pop.population_age_29) + parseFloat(values[i].populations[j].female_pop.population_age_29)) 
                      index.push(parseFloat(values[i].populations[j].male_pop.population_age_30) + parseFloat(values[i].populations[j].female_pop.population_age_30)) 
                      index.push(parseFloat(values[i].populations[j].male_pop.population_age_31) + parseFloat(values[i].populations[j].female_pop.population_age_31)) 
                      index.push(parseFloat(values[i].populations[j].male_pop.population_age_32) + parseFloat(values[i].populations[j].female_pop.population_age_32)) 
                      index.push(parseFloat(values[i].populations[j].male_pop.population_age_33) + parseFloat(values[i].populations[j].female_pop.population_age_33)) 
                      index.push(parseFloat(values[i].populations[j].male_pop.population_age_34) + parseFloat(values[i].populations[j].female_pop.population_age_34)) 
                      index.push(parseFloat(values[i].populations[j].male_pop.population_age_35) + parseFloat(values[i].populations[j].female_pop.population_age_35)) 
                      index.push(parseFloat(values[i].populations[j].male_pop.population_age_36) + parseFloat(values[i].populations[j].female_pop.population_age_36)) 
                      index.push(parseFloat(values[i].populations[j].male_pop.population_age_37) + parseFloat(values[i].populations[j].female_pop.population_age_37)) 
                      index.push(parseFloat(values[i].populations[j].male_pop.population_age_38) + parseFloat(values[i].populations[j].female_pop.population_age_38)) 
                      index.push(parseFloat(values[i].populations[j].male_pop.population_age_39) + parseFloat(values[i].populations[j].female_pop.population_age_39)) 
                      index.push(parseFloat(values[i].populations[j].male_pop.population_age_40) + parseFloat(values[i].populations[j].female_pop.population_age_40)) 
                      index.push(parseFloat(values[i].populations[j].male_pop.population_age_41) + parseFloat(values[i].populations[j].female_pop.population_age_41)) 
                      index.push(parseFloat(values[i].populations[j].male_pop.population_age_42) + parseFloat(values[i].populations[j].female_pop.population_age_42)) 
                      index.push(parseFloat(values[i].populations[j].male_pop.population_age_43) + parseFloat(values[i].populations[j].female_pop.population_age_43)) 
                      index.push(parseFloat(values[i].populations[j].male_pop.population_age_44) + parseFloat(values[i].populations[j].female_pop.population_age_44)) 
                      index.push(parseFloat(values[i].populations[j].male_pop.population_age_45) + parseFloat(values[i].populations[j].female_pop.population_age_45)) 
                      index.push(parseFloat(values[i].populations[j].male_pop.population_age_46) + parseFloat(values[i].populations[j].female_pop.population_age_46)) 
                      index.push(parseFloat(values[i].populations[j].male_pop.population_age_47) + parseFloat(values[i].populations[j].female_pop.population_age_47)) 
                      index.push(parseFloat(values[i].populations[j].male_pop.population_age_48) + parseFloat(values[i].populations[j].female_pop.population_age_48)) 
                      index.push(parseFloat(values[i].populations[j].male_pop.population_age_49) + parseFloat(values[i].populations[j].female_pop.population_age_49)) 
                      index.push(parseFloat(values[i].populations[j].male_pop.population_age_50) + parseFloat(values[i].populations[j].female_pop.population_age_50)) 
                      index.push(parseFloat(values[i].populations[j].male_pop.population_age_51) + parseFloat(values[i].populations[j].female_pop.population_age_51)) 
                      index.push(parseFloat(values[i].populations[j].male_pop.population_age_52) + parseFloat(values[i].populations[j].female_pop.population_age_52)) 
                      index.push(parseFloat(values[i].populations[j].male_pop.population_age_53) + parseFloat(values[i].populations[j].female_pop.population_age_53)) 
                      index.push(parseFloat(values[i].populations[j].male_pop.population_age_54) + parseFloat(values[i].populations[j].female_pop.population_age_54)) 
                      index.push(parseFloat(values[i].populations[j].male_pop.population_age_55) + parseFloat(values[i].populations[j].female_pop.population_age_55)) 
                      index.push(parseFloat(values[i].populations[j].male_pop.population_age_56) + parseFloat(values[i].populations[j].female_pop.population_age_56)) 
                      index.push(parseFloat(values[i].populations[j].male_pop.population_age_57) + parseFloat(values[i].populations[j].female_pop.population_age_57)) 
                      index.push(parseFloat(values[i].populations[j].male_pop.population_age_58) + parseFloat(values[i].populations[j].female_pop.population_age_58)) 
                      index.push(parseFloat(values[i].populations[j].male_pop.population_age_59) + parseFloat(values[i].populations[j].female_pop.population_age_59)) 
                      index.push(parseFloat(values[i].populations[j].male_pop.population_age_60) + parseFloat(values[i].populations[j].female_pop.population_age_60)) 
                      index.push(parseFloat(values[i].populations[j].male_pop.population_age_61) + parseFloat(values[i].populations[j].female_pop.population_age_61)) 
                      index.push(parseFloat(values[i].populations[j].male_pop.population_age_62) + parseFloat(values[i].populations[j].female_pop.population_age_62)) 
                      index.push(parseFloat(values[i].populations[j].male_pop.population_age_63) + parseFloat(values[i].populations[j].female_pop.population_age_63)) 
                      index.push(parseFloat(values[i].populations[j].male_pop.population_age_64) + parseFloat(values[i].populations[j].female_pop.population_age_64)) 
                      index.push(parseFloat(values[i].populations[j].male_pop.population_age_65) + parseFloat(values[i].populations[j].female_pop.population_age_65)) 
                      index.push(parseFloat(values[i].populations[j].male_pop.population_age_66) + parseFloat(values[i].populations[j].female_pop.population_age_66)) 
                      index.push(parseFloat(values[i].populations[j].male_pop.population_age_67) + parseFloat(values[i].populations[j].female_pop.population_age_67)) 
                      index.push(parseFloat(values[i].populations[j].male_pop.population_age_68) + parseFloat(values[i].populations[j].female_pop.population_age_68)) 
                      index.push(parseFloat(values[i].populations[j].male_pop.population_age_69) + parseFloat(values[i].populations[j].female_pop.population_age_69)) 
                      index.push(parseFloat(values[i].populations[j].male_pop.population_age_70) + parseFloat(values[i].populations[j].female_pop.population_age_70)) 
                      index.push(parseFloat(values[i].populations[j].male_pop.population_age_71) + parseFloat(values[i].populations[j].female_pop.population_age_71)) 
                      index.push(parseFloat(values[i].populations[j].male_pop.population_age_72) + parseFloat(values[i].populations[j].female_pop.population_age_72)) 
                      index.push(parseFloat(values[i].populations[j].male_pop.population_age_73) + parseFloat(values[i].populations[j].female_pop.population_age_73)) 
                      index.push(parseFloat(values[i].populations[j].male_pop.population_age_74) + parseFloat(values[i].populations[j].female_pop.population_age_74)) 
                      index.push(parseFloat(values[i].populations[j].male_pop.population_age_75) + parseFloat(values[i].populations[j].female_pop.population_age_75)) 
                      index.push(parseFloat(values[i].populations[j].male_pop.population_age_76) + parseFloat(values[i].populations[j].female_pop.population_age_76)) 
                      index.push(parseFloat(values[i].populations[j].male_pop.population_age_77) + parseFloat(values[i].populations[j].female_pop.population_age_77)) 
                      index.push(parseFloat(values[i].populations[j].male_pop.population_age_78) + parseFloat(values[i].populations[j].female_pop.population_age_78)) 
                      index.push(parseFloat(values[i].populations[j].male_pop.population_age_79) + parseFloat(values[i].populations[j].female_pop.population_age_79)) 
                      index.push(parseFloat(values[i].populations[j].male_pop.population_age_80) + parseFloat(values[i].populations[j].female_pop.population_age_80)) 
                      index.push(parseFloat(values[i].populations[j].male_pop.population_age_81) + parseFloat(values[i].populations[j].female_pop.population_age_81)) 
                      index.push(parseFloat(values[i].populations[j].male_pop.population_age_82) + parseFloat(values[i].populations[j].female_pop.population_age_82)) 
                      index.push(parseFloat(values[i].populations[j].male_pop.population_age_83) + parseFloat(values[i].populations[j].female_pop.population_age_83)) 
                      index.push(parseFloat(values[i].populations[j].male_pop.population_age_84) + parseFloat(values[i].populations[j].female_pop.population_age_84)) 
                      index.push(parseFloat(values[i].populations[j].male_pop.population_age_85) + parseFloat(values[i].populations[j].female_pop.population_age_85)) 
                      index.push(parseFloat(values[i].populations[j].male_pop.population_age_86) + parseFloat(values[i].populations[j].female_pop.population_age_86)) 
                      index.push(parseFloat(values[i].populations[j].male_pop.population_age_87) + parseFloat(values[i].populations[j].female_pop.population_age_87)) 
                      index.push(parseFloat(values[i].populations[j].male_pop.population_age_88) + parseFloat(values[i].populations[j].female_pop.population_age_88)) 
                      index.push(parseFloat(values[i].populations[j].male_pop.population_age_89) + parseFloat(values[i].populations[j].female_pop.population_age_89)) 
                      index.push(parseFloat(values[i].populations[j].male_pop.population_age_90) + parseFloat(values[i].populations[j].female_pop.population_age_90)) 
                      index.push(parseFloat(values[i].populations[j].male_pop.population_age_91) + parseFloat(values[i].populations[j].female_pop.population_age_91)) 
                      index.push(parseFloat(values[i].populations[j].male_pop.population_age_92) + parseFloat(values[i].populations[j].female_pop.population_age_92)) 
                      index.push(parseFloat(values[i].populations[j].male_pop.population_age_93) + parseFloat(values[i].populations[j].female_pop.population_age_93)) 
                      index.push(parseFloat(values[i].populations[j].male_pop.population_age_94) + parseFloat(values[i].populations[j].female_pop.population_age_94)) 
                      index.push(parseFloat(values[i].populations[j].male_pop.population_age_95) + parseFloat(values[i].populations[j].female_pop.population_age_95)) 
                      index.push(parseFloat(values[i].populations[j].male_pop.population_age_96) + parseFloat(values[i].populations[j].female_pop.population_age_96)) 
                      index.push(parseFloat(values[i].populations[j].male_pop.population_age_97) + parseFloat(values[i].populations[j].female_pop.population_age_97)) 
                      index.push(parseFloat(values[i].populations[j].male_pop.population_age_98) + parseFloat(values[i].populations[j].female_pop.population_age_98)) 
                      index.push(parseFloat(values[i].populations[j].male_pop.population_age_99) + parseFloat(values[i].populations[j].female_pop.population_age_99)) 
                      index.push(parseFloat(values[i].populations[j].male_pop.population_age_100) + parseFloat(values[i].populations[j].female_pop.population_age_100))
                      
                      
                     }
               }
           }
                
         }
     }
     var i = index.indexOf(Math.max(...index));
    
    y.max = i;
    
    return y;
						}''')
	result = year_key.map_reduce(map,reduce,"map_reduce_fr_most_represented_age_group" )
	new_post=[]
	for doc in list(result.find({"value.max" : {"$ne" : -1.0}}).sort("_id",1)):
		liste = {}
		liste["result"]= "\n  " + doc["_id"]  + "    Catégorie d'âge la plus représentée : " + str(doc["value"]["max"])
		new_post.append(liste)
		
	return render(request, "R8_2.html",{"data8" : new_post})
	
def R8_3(request, *args, **kwargs):
	map = Code('''function(){
					if(this._id.country_name == "Nepal")    
						emit(this._id.year, {"populations":[{male_pop : this.country_stats.Male, female_pop : this.country_stats.Female }],max: null}); 
				}; ''')
	reduce = Code('''function(keyCountry, values){
						y = {"populations":[]}
    index = [];
    for(i =0; i< values.length; i++)
    {
        if(values[i].populations !=null)
         {
            for(j=0; j<values[i].populations.length ; j++)
            {
                if (values[i].populations[j] !=null)
               {      max = 0.0;
                      if (values[i].populations[j].male_pop != null & values[i].populations[j].female_pop != null)
                     { 
                      y.populations.push(values[i].populations[j]);
                      index.push(parseFloat(values[i].populations[j].male_pop.population_age_0) + parseFloat(values[i].populations[j].female_pop.population_age_0))
                      index.push(parseFloat(values[i].populations[j].male_pop.population_age_1) + parseFloat(values[i].populations[j].female_pop.population_age_1))
                      index.push(parseFloat(values[i].populations[j].male_pop.population_age_2) + parseFloat(values[i].populations[j].female_pop.population_age_2))
                      index.push(parseFloat(values[i].populations[j].male_pop.population_age_3) + parseFloat(values[i].populations[j].female_pop.population_age_3))
                      index.push(parseFloat(values[i].populations[j].male_pop.population_age_4) + parseFloat(values[i].populations[j].female_pop.population_age_4))
                      index.push(parseFloat(values[i].populations[j].male_pop.population_age_5) + parseFloat(values[i].populations[j].female_pop.population_age_5))
                      index.push(parseFloat(values[i].populations[j].male_pop.population_age_6) + parseFloat(values[i].populations[j].female_pop.population_age_6))
                      index.push(parseFloat(values[i].populations[j].male_pop.population_age_7) + parseFloat(values[i].populations[j].female_pop.population_age_7))
                      index.push(parseFloat(values[i].populations[j].male_pop.population_age_8) + parseFloat(values[i].populations[j].female_pop.population_age_8))
                      index.push(parseFloat(values[i].populations[j].male_pop.population_age_9) + parseFloat(values[i].populations[j].female_pop.population_age_9))
                      index.push(parseFloat(values[i].populations[j].male_pop.population_age_10) + parseFloat(values[i].populations[j].female_pop.population_age_10))
                      index.push(parseFloat(values[i].populations[j].male_pop.population_age_11) + parseFloat(values[i].populations[j].female_pop.population_age_11))
                      index.push(parseFloat(values[i].populations[j].male_pop.population_age_12) + parseFloat(values[i].populations[j].female_pop.population_age_12))
                      index.push(parseFloat(values[i].populations[j].male_pop.population_age_13) + parseFloat(values[i].populations[j].female_pop.population_age_13))
                      index.push(parseFloat(values[i].populations[j].male_pop.population_age_14) + parseFloat(values[i].populations[j].female_pop.population_age_14))
                      index.push(parseFloat(values[i].populations[j].male_pop.population_age_15) + parseFloat(values[i].populations[j].female_pop.population_age_15))                      
                      index.push(parseFloat(values[i].populations[j].male_pop.population_age_16) + parseFloat(values[i].populations[j].female_pop.population_age_16))
                      index.push(parseFloat(values[i].populations[j].male_pop.population_age_17) + parseFloat(values[i].populations[j].female_pop.population_age_17))
                      index.push(parseFloat(values[i].populations[j].male_pop.population_age_18) + parseFloat(values[i].populations[j].female_pop.population_age_18))
                      index.push(parseFloat(values[i].populations[j].male_pop.population_age_19) + parseFloat(values[i].populations[j].female_pop.population_age_19))
                      index.push(parseFloat(values[i].populations[j].male_pop.population_age_20) + parseFloat(values[i].populations[j].female_pop.population_age_20)) 
                      index.push(parseFloat(values[i].populations[j].male_pop.population_age_21) + parseFloat(values[i].populations[j].female_pop.population_age_21)) 
                      index.push(parseFloat(values[i].populations[j].male_pop.population_age_22) + parseFloat(values[i].populations[j].female_pop.population_age_22)) 
                      index.push(parseFloat(values[i].populations[j].male_pop.population_age_23) + parseFloat(values[i].populations[j].female_pop.population_age_23)) 
                      index.push(parseFloat(values[i].populations[j].male_pop.population_age_24) + parseFloat(values[i].populations[j].female_pop.population_age_24)) 
                      index.push(parseFloat(values[i].populations[j].male_pop.population_age_25) + parseFloat(values[i].populations[j].female_pop.population_age_25)) 
                      index.push(parseFloat(values[i].populations[j].male_pop.population_age_26) + parseFloat(values[i].populations[j].female_pop.population_age_26)) 
                      index.push(parseFloat(values[i].populations[j].male_pop.population_age_27) + parseFloat(values[i].populations[j].female_pop.population_age_27)) 
                      index.push(parseFloat(values[i].populations[j].male_pop.population_age_28) + parseFloat(values[i].populations[j].female_pop.population_age_28)) 
                      index.push(parseFloat(values[i].populations[j].male_pop.population_age_29) + parseFloat(values[i].populations[j].female_pop.population_age_29)) 
                      index.push(parseFloat(values[i].populations[j].male_pop.population_age_30) + parseFloat(values[i].populations[j].female_pop.population_age_30)) 
                      index.push(parseFloat(values[i].populations[j].male_pop.population_age_31) + parseFloat(values[i].populations[j].female_pop.population_age_31)) 
                      index.push(parseFloat(values[i].populations[j].male_pop.population_age_32) + parseFloat(values[i].populations[j].female_pop.population_age_32)) 
                      index.push(parseFloat(values[i].populations[j].male_pop.population_age_33) + parseFloat(values[i].populations[j].female_pop.population_age_33)) 
                      index.push(parseFloat(values[i].populations[j].male_pop.population_age_34) + parseFloat(values[i].populations[j].female_pop.population_age_34)) 
                      index.push(parseFloat(values[i].populations[j].male_pop.population_age_35) + parseFloat(values[i].populations[j].female_pop.population_age_35)) 
                      index.push(parseFloat(values[i].populations[j].male_pop.population_age_36) + parseFloat(values[i].populations[j].female_pop.population_age_36)) 
                      index.push(parseFloat(values[i].populations[j].male_pop.population_age_37) + parseFloat(values[i].populations[j].female_pop.population_age_37)) 
                      index.push(parseFloat(values[i].populations[j].male_pop.population_age_38) + parseFloat(values[i].populations[j].female_pop.population_age_38)) 
                      index.push(parseFloat(values[i].populations[j].male_pop.population_age_39) + parseFloat(values[i].populations[j].female_pop.population_age_39)) 
                      index.push(parseFloat(values[i].populations[j].male_pop.population_age_40) + parseFloat(values[i].populations[j].female_pop.population_age_40)) 
                      index.push(parseFloat(values[i].populations[j].male_pop.population_age_41) + parseFloat(values[i].populations[j].female_pop.population_age_41)) 
                      index.push(parseFloat(values[i].populations[j].male_pop.population_age_42) + parseFloat(values[i].populations[j].female_pop.population_age_42)) 
                      index.push(parseFloat(values[i].populations[j].male_pop.population_age_43) + parseFloat(values[i].populations[j].female_pop.population_age_43)) 
                      index.push(parseFloat(values[i].populations[j].male_pop.population_age_44) + parseFloat(values[i].populations[j].female_pop.population_age_44)) 
                      index.push(parseFloat(values[i].populations[j].male_pop.population_age_45) + parseFloat(values[i].populations[j].female_pop.population_age_45)) 
                      index.push(parseFloat(values[i].populations[j].male_pop.population_age_46) + parseFloat(values[i].populations[j].female_pop.population_age_46)) 
                      index.push(parseFloat(values[i].populations[j].male_pop.population_age_47) + parseFloat(values[i].populations[j].female_pop.population_age_47)) 
                      index.push(parseFloat(values[i].populations[j].male_pop.population_age_48) + parseFloat(values[i].populations[j].female_pop.population_age_48)) 
                      index.push(parseFloat(values[i].populations[j].male_pop.population_age_49) + parseFloat(values[i].populations[j].female_pop.population_age_49)) 
                      index.push(parseFloat(values[i].populations[j].male_pop.population_age_50) + parseFloat(values[i].populations[j].female_pop.population_age_50)) 
                      index.push(parseFloat(values[i].populations[j].male_pop.population_age_51) + parseFloat(values[i].populations[j].female_pop.population_age_51)) 
                      index.push(parseFloat(values[i].populations[j].male_pop.population_age_52) + parseFloat(values[i].populations[j].female_pop.population_age_52)) 
                      index.push(parseFloat(values[i].populations[j].male_pop.population_age_53) + parseFloat(values[i].populations[j].female_pop.population_age_53)) 
                      index.push(parseFloat(values[i].populations[j].male_pop.population_age_54) + parseFloat(values[i].populations[j].female_pop.population_age_54)) 
                      index.push(parseFloat(values[i].populations[j].male_pop.population_age_55) + parseFloat(values[i].populations[j].female_pop.population_age_55)) 
                      index.push(parseFloat(values[i].populations[j].male_pop.population_age_56) + parseFloat(values[i].populations[j].female_pop.population_age_56)) 
                      index.push(parseFloat(values[i].populations[j].male_pop.population_age_57) + parseFloat(values[i].populations[j].female_pop.population_age_57)) 
                      index.push(parseFloat(values[i].populations[j].male_pop.population_age_58) + parseFloat(values[i].populations[j].female_pop.population_age_58)) 
                      index.push(parseFloat(values[i].populations[j].male_pop.population_age_59) + parseFloat(values[i].populations[j].female_pop.population_age_59)) 
                      index.push(parseFloat(values[i].populations[j].male_pop.population_age_60) + parseFloat(values[i].populations[j].female_pop.population_age_60)) 
                      index.push(parseFloat(values[i].populations[j].male_pop.population_age_61) + parseFloat(values[i].populations[j].female_pop.population_age_61)) 
                      index.push(parseFloat(values[i].populations[j].male_pop.population_age_62) + parseFloat(values[i].populations[j].female_pop.population_age_62)) 
                      index.push(parseFloat(values[i].populations[j].male_pop.population_age_63) + parseFloat(values[i].populations[j].female_pop.population_age_63)) 
                      index.push(parseFloat(values[i].populations[j].male_pop.population_age_64) + parseFloat(values[i].populations[j].female_pop.population_age_64)) 
                      index.push(parseFloat(values[i].populations[j].male_pop.population_age_65) + parseFloat(values[i].populations[j].female_pop.population_age_65)) 
                      index.push(parseFloat(values[i].populations[j].male_pop.population_age_66) + parseFloat(values[i].populations[j].female_pop.population_age_66)) 
                      index.push(parseFloat(values[i].populations[j].male_pop.population_age_67) + parseFloat(values[i].populations[j].female_pop.population_age_67)) 
                      index.push(parseFloat(values[i].populations[j].male_pop.population_age_68) + parseFloat(values[i].populations[j].female_pop.population_age_68)) 
                      index.push(parseFloat(values[i].populations[j].male_pop.population_age_69) + parseFloat(values[i].populations[j].female_pop.population_age_69)) 
                      index.push(parseFloat(values[i].populations[j].male_pop.population_age_70) + parseFloat(values[i].populations[j].female_pop.population_age_70)) 
                      index.push(parseFloat(values[i].populations[j].male_pop.population_age_71) + parseFloat(values[i].populations[j].female_pop.population_age_71)) 
                      index.push(parseFloat(values[i].populations[j].male_pop.population_age_72) + parseFloat(values[i].populations[j].female_pop.population_age_72)) 
                      index.push(parseFloat(values[i].populations[j].male_pop.population_age_73) + parseFloat(values[i].populations[j].female_pop.population_age_73)) 
                      index.push(parseFloat(values[i].populations[j].male_pop.population_age_74) + parseFloat(values[i].populations[j].female_pop.population_age_74)) 
                      index.push(parseFloat(values[i].populations[j].male_pop.population_age_75) + parseFloat(values[i].populations[j].female_pop.population_age_75)) 
                      index.push(parseFloat(values[i].populations[j].male_pop.population_age_76) + parseFloat(values[i].populations[j].female_pop.population_age_76)) 
                      index.push(parseFloat(values[i].populations[j].male_pop.population_age_77) + parseFloat(values[i].populations[j].female_pop.population_age_77)) 
                      index.push(parseFloat(values[i].populations[j].male_pop.population_age_78) + parseFloat(values[i].populations[j].female_pop.population_age_78)) 
                      index.push(parseFloat(values[i].populations[j].male_pop.population_age_79) + parseFloat(values[i].populations[j].female_pop.population_age_79)) 
                      index.push(parseFloat(values[i].populations[j].male_pop.population_age_80) + parseFloat(values[i].populations[j].female_pop.population_age_80)) 
                      index.push(parseFloat(values[i].populations[j].male_pop.population_age_81) + parseFloat(values[i].populations[j].female_pop.population_age_81)) 
                      index.push(parseFloat(values[i].populations[j].male_pop.population_age_82) + parseFloat(values[i].populations[j].female_pop.population_age_82)) 
                      index.push(parseFloat(values[i].populations[j].male_pop.population_age_83) + parseFloat(values[i].populations[j].female_pop.population_age_83)) 
                      index.push(parseFloat(values[i].populations[j].male_pop.population_age_84) + parseFloat(values[i].populations[j].female_pop.population_age_84)) 
                      index.push(parseFloat(values[i].populations[j].male_pop.population_age_85) + parseFloat(values[i].populations[j].female_pop.population_age_85)) 
                      index.push(parseFloat(values[i].populations[j].male_pop.population_age_86) + parseFloat(values[i].populations[j].female_pop.population_age_86)) 
                      index.push(parseFloat(values[i].populations[j].male_pop.population_age_87) + parseFloat(values[i].populations[j].female_pop.population_age_87)) 
                      index.push(parseFloat(values[i].populations[j].male_pop.population_age_88) + parseFloat(values[i].populations[j].female_pop.population_age_88)) 
                      index.push(parseFloat(values[i].populations[j].male_pop.population_age_89) + parseFloat(values[i].populations[j].female_pop.population_age_89)) 
                      index.push(parseFloat(values[i].populations[j].male_pop.population_age_90) + parseFloat(values[i].populations[j].female_pop.population_age_90)) 
                      index.push(parseFloat(values[i].populations[j].male_pop.population_age_91) + parseFloat(values[i].populations[j].female_pop.population_age_91)) 
                      index.push(parseFloat(values[i].populations[j].male_pop.population_age_92) + parseFloat(values[i].populations[j].female_pop.population_age_92)) 
                      index.push(parseFloat(values[i].populations[j].male_pop.population_age_93) + parseFloat(values[i].populations[j].female_pop.population_age_93)) 
                      index.push(parseFloat(values[i].populations[j].male_pop.population_age_94) + parseFloat(values[i].populations[j].female_pop.population_age_94)) 
                      index.push(parseFloat(values[i].populations[j].male_pop.population_age_95) + parseFloat(values[i].populations[j].female_pop.population_age_95)) 
                      index.push(parseFloat(values[i].populations[j].male_pop.population_age_96) + parseFloat(values[i].populations[j].female_pop.population_age_96)) 
                      index.push(parseFloat(values[i].populations[j].male_pop.population_age_97) + parseFloat(values[i].populations[j].female_pop.population_age_97)) 
                      index.push(parseFloat(values[i].populations[j].male_pop.population_age_98) + parseFloat(values[i].populations[j].female_pop.population_age_98)) 
                      index.push(parseFloat(values[i].populations[j].male_pop.population_age_99) + parseFloat(values[i].populations[j].female_pop.population_age_99)) 
                      index.push(parseFloat(values[i].populations[j].male_pop.population_age_100) + parseFloat(values[i].populations[j].female_pop.population_age_100))
                      
                      
                     }
               }
           }
                
         }
     }
     var i = index.indexOf(Math.max(...index));
    
    y.max = i;
    
    return y;
						}''')
	result = year_key.map_reduce(map,reduce,"map_reduce_fr_most_represented_age_group" )
	new_post=[]
	for doc in list(result.find({"value.max" : {"$ne" : -1.0}}).sort("_id",1)):
		liste = {}
		liste["result"]= "\n  " + doc["_id"]  + "    Catégorie d'âge la plus représentée : " + str(doc["value"]["max"])
		new_post.append(liste)
		
	return render(request, "R8_3.html",{"data8" : new_post})
	