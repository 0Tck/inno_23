window.onload = function () {
    document.getElementById("restro").style.borderBottom = "4px solid #66045E";
    //document.getElementById("hotelshow").style.display = "none";


}

document.getElementById("restro").onclick = function () {
    document.getElementById("hotel").style.borderBottom = "none";
    document.getElementById("restro").style.borderBottom = "4px solid #66045E";
    document.getElementById("hotelshow").style.display = "none";
    document.getElementById("restoshow").style.display = "flex";
}

document.getElementById("hotel").onclick = function () {
    document.getElementById("restro").style.borderBottom = "none";
    document.getElementById("hotel").style.borderBottom = "4px solid #66045E";
    document.getElementById("restoshow").style.display = "none";
    document.getElementById("hotelshow").style.display = "flex";
}


