var jQueryScript = document.createElement('script');
jQueryScript.setAttribute('src', 'https://cdn.jsdelivr.net/npm/sweetalert2@11.3.6/dist/sweetalert2.all.min.js');
document.head.appendChild(jQueryScript);

document.getElementById("sos").onclick = function () {
    Swal.fire({
        title: '<strong>Emergency <u>Help</u></strong>',
        icon: 'info',
        html: '<p>Select one Emergency Service</p>' + '<button onclick="changeit1()" class="emergency1">Police</button> <br> <button onclick="changeit2()" class="emergency2">Fire</button> <br> <button onclick="changeit3()" class="emergency3">Ambulance</button>',
        showCloseButton: true,
        showCancelButton: true,
        focusConfirm: false,
        confirmButtonText: '<i class="fas fa-check-circle"></i> Confirm!',
        confirmButtonAriaLabel: 'Thumbs up, great!',
        cancelButtonText: '<i class="fas fa-ban"></i> Cancel',
        cancelButtonAriaLabel: 'Thumbs down'
    }).then((result) => {
        let timerInterval
        Swal.fire({
            title: 'Sending Emergency Message',
            html: 'Your message is delivering in <b></b> milliseconds.',
            timer: 5000,
            timerProgressBar: true,
            didOpen: () => {
                Swal.showLoading()
                const b = Swal.getHtmlContainer().querySelector('b')
                timerInterval = setInterval(() => {
                    b.textContent = Swal.getTimerLeft()
                }, 10)
            },
            willClose: () => {
                clearInterval(timerInterval)
            }
        }).then((result) => {
            /* Read more about handling dismissals below */
            if (result.dismiss === Swal.DismissReason.timer) {
                console.log('I was closed by the timer')
            }
        })
    })
}


function changeit1() {
    document.querySelector(".emergency1").style.backgroundColor = "#F9FBDC";
    document.querySelector(".emergency1").style.color = "#000000";
}

function changeit2() {
    document.querySelector(".emergency2").style.backgroundColor = "#F62817";
}

function changeit3() {
    document.querySelector(".emergency3").style.backgroundColor = "#f3f3f3";
    document.querySelector(".emergency3").style.color = "#333";
}

function start() {
    location.replace("first-form.html");
}

function next1() {
    location.replace("second-form.html");
}

function next2() {
    location.replace("third-form.html");
}

function next3() {
    var budgetValue = document.getElementById("budget").value;
    var noOfPeople = document.getElementById("no_of_people").value;
    
    if(budgetValue <= 0 || noOfPeople == "select")
        {
            alert("Enter Correct Info only!!");
        }
    else{
        location.replace("journey.html");
        localStorage.setItem("budgetValue", budgetValue);
        localStorage.setItem("noOfPeople", noOfPeople);
    }
    
    
}

function showWorks() {
    location.replace("works.html");
}

function help() {
    location.replace("help.html");
}

function checkValue() {
    var valueofpersons = document.getElementById("no_of_people").value;
    if (valueofpersons <= 0) {
        alert("wrong Input");
    }
}

document.getElementById("returnBtn").onclick = function () {
    location.replace("journey.html");
}

function gotomaps() {

    var city1 = document.getElementById("location").value;
    var city2 = document.getElementById("destination").value;

    if (city1 == "" || city2 == "") {
        alert("Enter Both City Name First")
    } else {
        location.replace("map.html");
        localStorage.setItem("city1", city1);
        localStorage.setItem("city2", city2);
    }


}
