var tcost = Math.floor(Math.random() * 2000) + 1000;;
var hcost = Math.floor(Math.random() * 2000) + 500;;
var fcost = Math.floor(Math.random() * 1000) + 100;;
var ocost = Math.floor(Math.random() * 500) + 100;;

var dividedCost = localStorage.getItem("noOfPeople");

console.log(dividedCost);

document.getElementById("transportCost").innerHTML = tcost;
document.getElementById("hotelCost").innerHTML = hcost;
document.getElementById("foodCost").innerHTML = fcost;
document.getElementById("otherCost").innerHTML = ocost;

document.getElementById("totalCost").innerHTML = tcost + hcost + fcost + ocost;
document.getElementById("splitCost").innerHTML = (tcost + hcost + fcost + ocost) / dividedCost;



var ctx = document.getElementById('myChart');
var myChart = new Chart(ctx, {
    type: 'pie',
    data: {
        labels: ['Transport', 'Hotel', 'Food', 'Others'],
        datasets: [{
            label: '# of Votes',
            data: [tcost, hcost, fcost, ocost],
            backgroundColor: [
                'rgba(255, 99, 132, 1)',
                'rgba(54, 162, 235, 1)',
                'rgba(255, 206, 86, 1)',
                'rgba(75, 192, 192, 1)',
                'rgba(153, 102, 255, 1)',
                'rgba(255, 159, 64, 1)'
            ],
            borderColor: [
                'rgba(255, 99, 132, 1)',
                'rgba(54, 162, 235, 1)',
                'rgba(255, 206, 86, 1)',
                'rgba(75, 192, 192, 1)',
                'rgba(153, 102, 255, 1)',
                'rgba(255, 159, 64, 1)'
            ],
            borderWidth: 1
        }]
    },
    options: {
        animations: {
            tension: {
                duration: 1000,
                easing: 'linear',
                from: 1,
                to: 0,
                loop: true
            }
        },
        scales: {
            y: {
                beginAtZero: true
            }
        }
    }
});
