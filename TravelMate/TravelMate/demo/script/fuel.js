var min = Math.floor(Math.random() * 30) + 1.31;
var maz = Math.floor(Math.random() * 50) + 1.23;

var carFuel = Math.floor(Math.random() * ((maz / min) + 1.54)) + min;
document.getElementById("showfuelcomsuption").innerHTML = carFuel;

var station = Math.floor(Math.random() * 10) + 1;
document.getElementById("station").innerHTML = station;


var tempcity1 = localStorage.getItem("city1");
var tempcity2 = localStorage.getItem("city2");

var city1 = tempcity1.toLowerCase();
var city2 = tempcity2.toLowerCase();


document.getElementById("city1").innerHTML = city1;
document.getElementById("city2").innerHTML = city2;

var states = {
    'agra': '96.35',
    'ahmedabad': '96.42',
    'allahabad': '96.65',
    'aurangabad': '112.97',
    'bangalore': '101.94',
    'bhopal': '108.65',
    'bhubaneswar': '103.19',
    'chandigarh': '96.20',
    'chennai': '102.63',
    'coimbatore': '103.11',
    'dehradun': '95.28',
    'delhi': '95.41',
    'erode': '103.12',
    'gurgaon': '97.18',
    'guwahati': '96.01',
    'hyderabad': '109.66',
    'indore': '108.68',
    'jaipur': '108.48',
    'jammu': '97.50',
    'jamshedpur': '99.77',
    'kanpur': '96.26',
    'kolhapur': '111.44',
    'kolkata': '104.03',
    'kozhikode': '105.89',
    'lucknow': '96.57',
    'ludhiana': '96.81',
    'madurai': '103.81',
    'mangalore': '101.13',
    'mumbai': '111.35',
    'mysore': '101.46',
    'nagpur': '97.04',
    'nashik': '111.74',
    'patna': '107.24',
    'pune': '110.88',
    'raipur': '102.45',
    'rajkot': '96.19',
    'ranchi': '99.84',
    'salem': '103.39',
    'shimla': '97.58',
    'srinagar': '102.11',
    'surat': '96.31',
    'thane': '111.49',
    'trichy': '103.08',
    'vadodara': '96.08',
    'varanasi': '97.39',
    'visakhapatnam': '110.48',
    'faridabad': '97.49',
    'ghaziabad': '96.58',
    'noida': '96.79',
    'thiruvananthapuram': '107.71',
};


var price1 = 0;
var price2 = 0;

if (city1 in states) {
    document.getElementById("price1").innerHTML = states[city1];
    price1 = states[city1];
} else {
    document.getElementById("price1").innerHTML = "Invalid City Found";
}

if (city2 in states) {
    document.getElementById("price2").innerHTML = states[city2];
    price2 = states[city2];
} else {
    document.getElementById("price2").innerHTML = "Invalid City Found";
}


var diff = 0;
if (price1 > price2) {
    diff = price2 - price1;
    document.getElementById("fueldiff").innerHTML = diff.toFixed(2) + " loss";
    document.getElementById("fueldiff").style.color = "red";
} else if (price2 > price1) {
    diff = price1 - price2;
    document.getElementById("fueldiff").innerHTML = diff.toFixed(2) + " profit";
    document.getElementById("fueldiff").style.color = "green";
} else {
    document.getElementById("fueldiff").innerHTML = "neither profit nor loss";
}
