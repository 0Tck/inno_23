window.onload = function () {
    if (navigator.geolocation) {
        navigator.geolocation.watchPosition(showPosition);
    } else {
        alert("location access denied");
    }

    document.getElementById("showMap").src = "https://www.google.com/maps/d/embed?mid=1IEVwjHNKZcYI5LKHZfRSFXEKyzg&ehbc=2E312F";

    //document.getElementById("showMap").src = "https://www.google.com/maps/embed/v1/place?key=AIzaSyBJNY5Iw6lBN66wXCTmI1_gtV5u6pg2j7A&q=Space+Needle,Seattle+WA";
}


function showPosition(position) {
    var x = document.getElementById("demo");
    x.innerHTML = "User Latitude: " + position.coords.latitude +
        "<br> User Longitude: " + position.coords.longitude;
}

var time;
var distance = 33;
var addup = Math.floor(Math.random() * 4) + 1;
var minutes = Math.floor(Math.random() * 60) + 1;
var seconds = Math.floor(Math.random() * 60) + 1;
var d = new Date(); // for now
var hours = d.getHours(); // => 9
var tempcity1 = localStorage.getItem("city1");
var tempcity2 = localStorage.getItem("city2");
var location1 = tempcity1.toLowerCase();
var location2 = tempcity2.toLowerCase();
document.getElementById("userLocation").innerHTML = tempcity1;
//console.log(d + addup);



console.log(location1);

if (location1 == "delhi") {
    showTravelDetails(location2);
    console.log("city detected");
} else if (location2 == "delhi") {
    showTravelDetails(location1);
    console.log("city detected");
} else {
    console.log("No City Detected");
}

function showTravelDetails(cityName) {
    console.log("the city " + cityName);
    switch (cityName) {
        case "bangalore":
            document.getElementById("travelDistance").innerHTML = "2175 Km";
            document.getElementById("travelDuration").innerHTML = "Car - 37 hours<br>Train - 40 hours";
            break;
        case "pune":
            document.getElementById("travelDistance").innerHTML = "1475 Km";
            document.getElementById("travelDuration").innerHTML = "Car - 27 hours<br>Train - 24 hours";
            break;
        case "hydrabad":
            document.getElementById("travelDistance").innerHTML = "1583 Km";
            document.getElementById("travelDuration").innerHTML = "Car - 28 hours<br>Train - 32 hours";
            break;
        case "mumbai":
            document.getElementById("travelDistance").innerHTML = "1448 Km";
            document.getElementById("travelDuration").innerHTML = "Car - 26 hours<br>Train - 20 hours";
            break;
        case "dehradun":
            document.getElementById("travelDistance").innerHTML = "240 Km";
            document.getElementById("travelDuration").innerHTML = "Car - 5 hours<br>Train - 10 hours";
            break;
        case "chandigarh":
            document.getElementById("travelDistance").innerHTML = "243 Km";
            document.getElementById("travelDuration").innerHTML = "Car - 5 hours<br>Train - 5 hours";
            break;
        case "agra":
            document.getElementById("travelDistance").innerHTML = "243 Km";
            document.getElementById("travelDuration").innerHTML = "Car - 4 hours<br>Train - 5 hours";
            break;
        case "jammu":
            document.getElementById("travelDistance").innerHTML = "586 Km";
            document.getElementById("travelDuration").innerHTML = "Car - 11 hours<br>Train - 11 hours";
            break;
        case "ludhiana":
            document.getElementById("travelDistance").innerHTML = "309 Km"; 
            document.getElementById("travelDuration").innerHTML = "Car - 5 hours<br>Train - 6 hours";
            break;
        case "indore":
            document.getElementById("travelDistance").innerHTML = "860 Km";
            document.getElementById("travelDuration").innerHTML = "Car - 15 hours<br>Train - 14 hours";
            break;
        default:
            alert("Not A Valid City");
            break;
    }
}



var map = tt.map({
    key: 'IH5ICZ4povMd8ytGHhialJxGLh5qlKYW',
    container: 'map',
    dragPan: !isMobileOrTablet()
});
map.addControl(new tt.FullscreenControl());
map.addControl(new tt.NavigationControl());
